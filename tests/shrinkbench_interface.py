# autopep8: off
import os
import sys
import argparse
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# --- ShrinkBench Imports ---
from shrinkbench.strategies import VisionPruning
from shrinkbench.compress import compress
from shrinkbench.util import evaluate

from causalpruner import (
    CausalWeightsTrainerConfig,
    SGDPruner,
    SGDPrunerConfig,
)
from datasets import get_dataset
from models import get_model

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
        
     
        pruner = SGDPruner(self.sgd_pruner_config)
        
        masks = None
        for i in range(self.sgd_pruner_config.num_prune_iterations):
            print(f"  Pruning Iteration {i + 1}/{self.sgd_pruner_config.num_prune_iterations}")

            masks = pruner.run_prune_iteration()
        
        print("Global Causal Pruning finished.")
        
        # The returned masks are keyed by module name (e.g., 'layer1.conv1')
        #ShrinkBench expects keys to be parameter names 
        final_masks = {f"{name}.weight": mask for name, mask in masks.items()}

        # Filter to only include parameters that are actually in the model
        model_params = self.params().keys()
        final_masks = {name: mask for name, mask in final_masks.items() if name in model_params}

        return final_masks


