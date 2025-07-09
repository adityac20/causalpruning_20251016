import sys
import os
import itertools
import os
import glob
import pdb


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tests.models import get_model
from tests.datasets import get_dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(total=len(testloader))
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.update(1)
    pbar.close()
    return correct / total


def compute_sparsity(model):
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total


def load_pruned_weights(model, path):
    try:
        model.load_state_dict(torch.load(path))
        return model
    except:
        pass

    try:
        model.load_state_dict(torch.load(path)["model"])
        return model
    except:
        pass

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.0)

    model.load_state_dict(torch.load(path)["model"])

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if prune.is_pruned(module):
                prune.remove(module, "weight")

    return model


def find_resnet20_causalpruner_0_8_models(base_dir="."):
    """
    Find paths to model.trained.ckpt files in folders containing
    'resnet20', 'causalpruner', and '_0.8' in their names.

    Args:
        base_dir (str): Base directory to search from (default: current directory)

    Returns:
        list: List of full paths to model.trained.ckpt files
    """
    target_dirs = [
        "checkpoints",
        "checkpoints_ablation",
        "checkpoints_all_results",
        "checkpoints_all_results2",
    ]

    # Add checkpoints if it exists
    checkpoints_path = os.path.join(base_dir, "checkpoints")
    if os.path.exists(checkpoints_path):
        target_dirs.append("checkpoints")

    found_paths = []

    for target_dir in target_dirs:
        search_path = os.path.join(base_dir, target_dir)

        if not os.path.exists(search_path):
            continue

        # Recursively search through all subdirectories
        for root, dirs, files in os.walk(search_path):
            # Check if current directory name contains all required strings
            dir_name = os.path.basename(root)

            if all(keyword in dir_name for keyword in ["resnet20", "causalpruner", "_0.9"]):
                # Check if model.trained.ckpt exists in this directory
                model_path = os.path.join(root, "model.trained.ckpt")
                if os.path.exists(model_path):
                    found_paths.append(model_path)

    return found_paths


def get_model_paths_dict():
    path1 = "/home/think-server/Documents/cpn/checkpoints_all_results2/resnet20_cifar10_causalpruner_1_1_1_0.001_0.8/model.trained.ckpt"
    path2 = "./checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_causalpruner_1_10_1_0.001_0.9/model.trained.ckpt"
    path3 = "/home/think-server/Documents/cpn/checkpoints_all_results/mobilenet_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.trained.ckpt"
    path4 = "/home/think-server/Documents/cpn/checkpoints_all_results/resnet50_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.trained2.ckpt"
    dict_models = {
        ("resnet20", "cifar10", 0.8): path1,
        ("resnet20", "cifar10", 0.9): path2,
        ("mobilenet_untrained", "imagenet", 0.8): path3,
        ("resnet50_untrained", "imagenet", 0.8): path4,
    }
    return dict_models


# Example usage:
if __name__ == "__main__":
    best_acc = 0
    best_path = None
    path1 = "/home/think-server/Documents/cpn/checkpoints_all_results2/resnet20_cifar10_causalpruner_1_1_1_0.001_0.8/model.trained.ckpt"
    path2 = "./checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_causalpruner_1_10_1_0.001_0.9/model.trained.ckpt"

    dict_models = get_model_paths_dict()
    # for (model_name, dataset_name, prune_amount), path in dict_models.items():
    model_name = "resnet18"
    dataset_name = "cifar10"
    prune_amount = 0.9
    causal_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_causalpruner_20_10_10_0.001_0.9/model.trained.ckpt"
    mag_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_magpruner_0.9/model.trained.ckpt"
    base_path = (
        "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_noprune/model.trained.ckpt"
    )
    path = causal_path
    model = get_model(model_name, dataset_name, checkpoint_dir=None)
    model = load_pruned_weights(model, path).to(device)
    # Get the datasets
    tmp = get_dataset(dataset_name, model_name, "./data")
    trainset = tmp[0]
    testset = tmp[1]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    acc = get_accuracy(model, testloader)
    sparsity = compute_sparsity(model)
    print(
        f"model : {model_name} dataset : {dataset_name} prune_amount : {prune_amount} acc : {acc} sparsity : {sparsity}"
    )
