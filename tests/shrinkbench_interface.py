# autopep8: off
import os
import sys
import argparse
from copy import deepcopy

# Assuming this script is in a folder like 'strategies/' and the project root is '../'
# Adjust the path as necessary for your project structure.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# --- ShrinkBench Imports ---
from shrinkbench.strategies import VisionPruning
from shrinkbench.compress import compress
from shrinkbench.util import evaluate

# --- Your Project's Imports ---
# Make sure these paths are correct relative to your project structure
from causalpruner import (
    CausalWeightsTrainerConfig,
    SGDPruner,
    SGDPrunerConfig,
)
from datasets import get_dataset
from models import get_model


# =================================================================================
# SECTION 1: The ShrinkBench Strategy Class
# =================================================================================

class GlobalCausalPruning(VisionPruning):
    
    def __init__(self, model, fraction, sgd_pruner_config):
        """
        A ShrinkBench strategy that uses the refactored SGDPruner.
        
        Args:
            model (nn.Module): The model to be pruned.
            fraction (float): The total fraction of weights to prune.
            sgd_pruner_config (SGDPrunerConfig): A fully populated config object.
                                                 This is the key to linking everything.
        """
        super().__init__(model, fraction)
        
        self.sgd_pruner_config = deepcopy(sgd_pruner_config)
        # --- KEY MODIFICATIONS TO THE CONFIG ---
        self.sgd_pruner_config.return_masks_only = True # 1. Set the new flag
        self.sgd_pruner_config.model = self.model      # 2. Ensure it has the correct model reference
        self.sgd_pruner_config.trainer_config.prune_amount = self.fraction # 3. Set prune amount
        
    def model_masks(self):
        """
        Computes the final masks by running the iterative SGDPruner process.
        """
        print(f"\nStarting Global Causal Pruning for {self.sgd_pruner_config.num_prune_iterations} iterations...")
        
        # Instantiate the pruner. It's now configured to return masks.
        pruner = SGDPruner(self.sgd_pruner_config)
        
        masks = None
        for i in range(self.sgd_pruner_config.num_prune_iterations):
            print(f"  Pruning Iteration {i + 1}/{self.sgd_pruner_config.num_prune_iterations}")
            
            # This single call now does all the work AND returns the result
            masks = pruner.run_prune_iteration()
        
        print("Global Causal Pruning finished.")
        
        # The returned masks are keyed by module name (e.g., 'layer1.conv1')
        # ShrinkBench expects keys to be parameter names (e.g., 'layer1.conv1.weight')
        final_masks = {f"{name}.weight": mask for name, mask in masks.items()}

        # Filter to only include parameters that are actually in the model
        model_params = self.params().keys()
        final_masks = {name: mask for name, mask in final_masks.items() if name in model_params}

        return final_masks


# =================================================================================
# SECTION 2: Main function and Argument Parser
# =================================================================================

def main(args):
    """
    Sets up and runs the Causal Pruning experiment using the ShrinkBench framework.
    """
    print("=" * 60)
    print("Running Causal Pruning with ShrinkBench")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Target Sparsity: {args.total_prune_amount * 100:.1f}%")
    print(f"Prune Iterations: {args.num_prune_iterations}")
    print("=" * 60)

    fabric = Fabric(accelerator="auto", devices=1)
    fabric.launch()

    model = get_model(args.model, args.dataset)
    train_dataset, test_dataset, _ = get_dataset(args.dataset, args.model, args.dataset_root_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    model = fabric.setup(model)

    # print("\nEvaluating dense model...")
    # dense_loss, dense_acc = evaluate(model, test_loader, fabric.device)
    # print(f"Dense Model Accuracy: {dense_acc * 100:.2f}%")

    causal_config = CausalWeightsTrainerConfig(
        fabric=fabric,
        init_lr=args.causal_pruner_init_lr,
        l1_regularization_coeff=args.causal_pruner_l1_regularization_coeff,
        prune_amount=args.total_prune_amount,
        max_iter=args.causal_pruner_max_iter,
        loss_tol=args.causal_pruner_loss_tol,
        num_iter_no_change=args.causal_pruner_num_iter_no_change,
        batch_size=args.causal_pruner_batch_size,
        num_dataloader_workers=args.num_causal_pruner_dataloader_workers,
        pin_memory=True,
        backend="torch",
    )

    sgd_config = SGDPrunerConfig(
        fabric=fabric,
        model=model,
        pruner="SGDPruner",
        checkpoint_dir=args.checkpoint_dir,
        start_clean=True,
        reset_weights=False,
        reset_params=False,
        num_prune_iterations=args.num_prune_iterations,
        num_prune_epochs=args.num_prune_epochs,
        prune_dataloader=train_loader,
        prune_optimizer_lr=args.causal_pruner_train_lr,
        verbose=False,
        threaded_checkpoint_writer=True,
        delete_checkpoint_dir_after_training=True,
        trainer_config=causal_config,
    )

    strategy = GlobalCausalPruning(model,
                                   fraction=args.total_prune_amount,
                                   sgd_pruner_config=sgd_config)
    
    pruned_model, stats = compress(model, strategy, test_loader, fabric.device)

    print("\n--- Pruning Results ---")
    print(stats)

    print("\nEvaluating pruned model (before fine-tuning)...")
    pruned_loss, pruned_acc = evaluate(pruned_model, test_loader, fabric.device)
    print(f"Pruned Model Accuracy: {pruned_acc * 100:.2f}%")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal Pruning with ShrinkBench")

    parser.add_argument("--model", type=str, default="resnet20", help="Model name",
                        choices=["resnet20", "resnet50_torch", "mobilenet_untrained"])
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name",
                        choices=["cifar10", "tinyimagenet", "imagenet"])
    parser.add_argument("--total_prune_amount", type=float, default=0.8,
                        help="Total prune amount after all the iterations (e.g., 0.8 for 80%)")
    parser.add_argument("--num_prune_iterations", type=int, default=8,
                        help="Number of iterations to reach the target pruning amount")
    

    parser.add_argument("--dataset_root_dir", type=str, default="../data",
                        help="Directory to download datasets")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for data loaders")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/shrinkbench_causal_temp",
                        help="Temporary directory for pruner to write data files")

 
    parser.add_argument("--num_prune_epochs", type=int, default=1,
                        help="Number of epochs for data gathering in each prune iteration")
    parser.add_argument("--causal_pruner_train_lr", type=float, default=1e-4,
                        help="Learning rate for the SGD optimizer during data gathering")
    parser.add_argument("--causal_pruner_init_lr", type=float, default=0.01,
                        help="Initial learning rate for the CausalWeightsTrainer")
    parser.add_argument("--causal_pruner_l1_regularization_coeff", type=float, default=1e-4,
                        help="Causal Pruner L1 regularization coefficient")
    parser.add_argument("--causal_pruner_max_iter", type=int, default=30,
                        help="Max iterations for CausalWeightsTrainer fitting")
    parser.add_argument("--causal_pruner_loss_tol", type=float, default=1e-7,
                        help="Loss tolerance for CausalWeightsTrainer early stopping")
    parser.add_argument("--causal_pruner_num_iter_no_change", type=int, default=3,
                        help="Patience for CausalWeightsTrainer early stopping")
    parser.add_argument("--causal_pruner_batch_size", type=int, default=128,
                        help="Batch size for CausalWeightsTrainer fitting")
    parser.add_argument("--num_causal_pruner_dataloader_workers", type=int, default=4,
                        help="Dataloader workers for CausalWeightsTrainer")

    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()
    main(args)