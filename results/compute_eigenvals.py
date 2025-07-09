import sys
import os
import itertools
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset

from tests.models import get_model
from tests.datasets import get_dataset

from hessian_eigenthings import compute_hessian_eigenthings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_paths(model_name: str, method_name: str, prune_amt: float) -> str:
    if model_name == "lenet":
        if method_name == "causal":
            path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_causalpruner_5_10_1_0.001_{prune_amt}/model.trained.ckpt"
        elif method_name == "mag":
            path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        elif method_name == "noprune":
            path = "/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_noprune/model.trained.ckpt"

    elif model_name == "resnet20":
        causal_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_causalpruner_1_10_1_0.001_{prune_amt}/model.trained.ckpt"
        mag_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        base_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_noprune/model.trained.ckpt"
        if method_name == "causal":
            path = causal_path
        elif method_name == "mag":
            path = mag_path
        elif method_name == "noprune":
            path = base_path

    elif model_name == "resnet18":
        causal_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_causalpruner_20_10_10_0.001_0.9/model.trained.ckpt"
        mag_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_magpruner_0.9/model.trained.ckpt"
        base_path = "/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_noprune/model.trained.ckpt"
        if method_name == "causal":
            path = causal_path
        elif method_name == "mag":
            path = mag_path
        elif method_name == "noprune":
            path = base_path

    return path


def load_weights(model, path):
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


def get_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def compute_eigenvals_resnet20_cifar10():
    model_name = "resnet20"
    list_method_name = ["causal", "mag", "noprune"]
    list_prune_amt = [0.9, 0.7, 0.5, 0]
    for method_name, prune_amt in itertools.product(list_method_name, list_prune_amt):
        # Skip noprune for non-zero prune amounts
        if method_name == "noprune" and prune_amt != 0:
            continue
        if prune_amt == 0 and method_name != "noprune":
            continue
        print(f"Computing eigenvalues for {model_name} with {method_name} and {prune_amt}")
        path = get_paths(model_name, method_name, prune_amt)
        model = get_model(model_name, "cifar10", checkpoint_dir=None)
        model = load_weights(model, path)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet20", "./data")
        testset = tmp[1]
        testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=8)

        # Compute the hessian
        criterion = nn.CrossEntropyLoss()
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            model,
            testloader,
            criterion,
            40,
            mode="lanczos",
            max_possible_gpu_samples=1024,
            full_dataset=True,
            use_gpu=True,
        )
        fname = f"./results/eigvals/eigenvals_{model_name}_{method_name}_{prune_amt}.txt"
        np.savetxt(fname, eigenvals)
        print(f"Eigenvalues saved to {fname}")


def compute_eigenvals_resnet18_cifar10():
    model_name = "resnet18"
    list_method_name = ["causal", "mag", "noprune"]
    list_prune_amt = [0.9, 0]
    for method_name, prune_amt in itertools.product(list_method_name, list_prune_amt):
        # Skip noprune for non-zero prune amounts
        if method_name == "noprune" and prune_amt != 0:
            continue
        if prune_amt == 0 and method_name != "noprune":
            continue
        print(f"Computing eigenvalues for {model_name} with {method_name} and {prune_amt}")
        path = get_paths(model_name, method_name, prune_amt)
        model = get_model(model_name, "cifar10", checkpoint_dir=None)
        model = load_weights(model, path)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet18", "./data")
        testset = tmp[1]
        testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=8)

        # Compute the hessian
        criterion = nn.CrossEntropyLoss()
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            model,
            testloader,
            criterion,
            40,
            mode="lanczos",
            max_possible_gpu_samples=1024,
            full_dataset=True,
            use_gpu=True,
        )
        fname = f"./results/eigvals/eigenvals_{model_name}_{method_name}_{prune_amt}.txt"
        np.savetxt(fname, eigenvals)
        print(f"Eigenvalues saved to {fname}")


if __name__ == "__main__":
    compute_eigenvals_resnet18_cifar10()
