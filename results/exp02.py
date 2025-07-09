import sys
import os
import itertools
import pdb
import copy
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tests.models import get_model
from tests.datasets import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(total=len(testloader), leave=False)
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.update(1)
            pbar.set_description(f"Acc: {correct / total:.4f}")
        pbar.close()
    return correct / total


def compute_sparsity(model):
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total


def load_weights(model, path, flag_noprune: bool = False):
    try:
        state_dict = torch.load(path)["model"]
    except:
        state_dict = torch.load(path)

    if flag_noprune:
        model.load_state_dict(state_dict)
        return model

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.0)
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.0)

    model.load_state_dict(state_dict)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                prune.remove(module, "weight")
        if isinstance(module, nn.Linear):
            if prune.is_pruned(module):
                prune.remove(module, "weight")
    return model


def load_weights_v2(model, path, flag_fc_prune: bool = False):
    try:
        state_dict = torch.load(path)["model"]
    except:
        state_dict = torch.load(path)

    # Apply pruning structure to match the checkpoint
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.0)
        if isinstance(module, nn.Linear) and flag_fc_prune:
            prune.l1_unstructured(module, name="weight", amount=0.0)

    # Load the checkpoint state dict (this includes weight_orig and weight_mask)
    model.load_state_dict(state_dict, strict=False)

    return model


def apply_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    return model


def init_random_direction(state_dict: dict) -> dict:
    new_state_dict = copy.deepcopy(state_dict)
    for name, param in state_dict.items():
        if param.dim() <= 1:
            continue
        elif "mask" in name:
            continue
        else:
            random_weights = torch.randn_like(param)
            nn.init.normal_(random_weights)
            # for d, w in zip(random_weights, param):
            #     d.mul_(w.norm() / (d.norm() + 1e-10))
            new_state_dict[name].copy_(random_weights)

    return new_state_dict


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> float:
    """
    Evaluate the average loss of a model on a test dataset.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader containing test data
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        device: Device to run evaluation on ('cpu' or 'cuda')

    Returns:
        Average loss over the test dataset
    """
    model.eval()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = loss_fn(output, target)

        # Accumulate loss (multiply by batch size for proper averaging)
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Return average loss
    return total_loss / total_samples


def add_noise_to_model(
    base_model: nn.Module,
    base_state_dict: dict,
    noise_level: float,
) -> nn.Module:

    new_state_dict = copy.deepcopy(base_state_dict)
    random_state_dict = init_random_direction(base_state_dict)

    # Interpolate parameters
    for name in new_state_dict.keys():
        # if new_state_dict[name].dim() <= 1:
        #     continue
        new_state_dict[name].copy_(new_state_dict[name] + noise_level * random_state_dict[name])

    base_model.load_state_dict(new_state_dict)
    return base_model


def get_paths(model_name: str, prune_amt: float) -> str:
    if model_name == "lenet":
        causal_path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_causalpruner_5_10_1_0.001_{prune_amt}/model.trained.ckpt"
        mag_path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        base_path = (
            "/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_noprune/model.trained.ckpt"
        )

        return {
            "causal": causal_path,
            "mag": mag_path,
            "base": base_path,
        }

    elif model_name == "resnet20":
        causal_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_causalpruner_1_10_1_0.001_{prune_amt}/model.trained.ckpt"
        mag_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        base_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_noprune/model.trained.ckpt"
        return {
            "causal": causal_path,
            "mag": mag_path,
            "base": base_path,
        }

    elif model_name == "resnet18":
        causal_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_causalpruner_20_10_10_0.001_0.9/model.trained.ckpt"
        mag_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_magpruner_0.9/model.trained.ckpt"
        base_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_noprune/model.trained.ckpt"
        return {
            "causal": causal_path,
            "mag": mag_path,
            "base": base_path,
        }


def noise_tolerance_experiment():
    model_name = "resnet18"
    prune_amt = 0.9
    dict_paths = get_paths(model_name, prune_amt)

    pbar = tqdm(total=len(dict_paths) * 20 * 10, leave=False)
    with open(f"results/exp02_results_{model_name}.csv", "w") as f:
        f.write("path_name,noise_level,loss,relative_loss\n")
    for path_name, path_select in dict_paths.items():
        print(f"Evaluating {path_name}")
        flag_noprune = True if path_name == "base" else False
        model = get_model("resnet18", "cifar10", checkpoint_dir=None)
        model = load_weights(model, path_select, flag_noprune).to(device)
        base_state_dict = copy.deepcopy(model.state_dict())

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet18", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=4
        )

        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        print(f"acc : {acc} sparsity : {sparsity}")

        base_loss = evaluate_loss(model, testloader)
        print(f"Base loss: {base_loss}")

        for noise_level in np.logspace(-5, -1, 20):
            for _ in range(10):
                model = add_noise_to_model(model, base_state_dict, noise_level)
                loss = evaluate_loss(model, testloader)
                with open(f"results/exp02_results_{model_name}.csv", "a") as f:
                    f.write(f"{path_name},{noise_level},{loss},{loss / base_loss}\n")
                pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    noise_tolerance_experiment()
