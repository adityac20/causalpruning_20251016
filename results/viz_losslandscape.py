import sys
import os
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import copy
from typing import Callable, Union
import itertools
from tqdm import tqdm

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from tests.models import get_model
from tests.datasets import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_random_direction(state_dict: dict, seed: int = 42) -> dict:
    # Set random seed for reproducibility
    rng_torch = torch.Generator(device=device).manual_seed(seed)
    rng_np = np.random.default_rng(seed)

    new_state_dict = copy.deepcopy(state_dict)

    for name, param in state_dict.items():
        if param.dim() <= 1:
            continue
        else:
            param_shape = param.shape
            random_weights = torch.randn_like(param)
            nn.init.normal_(random_weights, generator=rng_torch)
            for d, w in zip(random_weights, param):
                d.mul_(w.norm() / (d.norm() + 1e-10))
            new_state_dict[name].copy_(random_weights)

    return new_state_dict


def interpolate_models(
    base_model: nn.Module,
    base_state_dict: dict,
    x_state_dict: dict,
    y_state_dict: dict,
    alpha: float,
    beta: float,
) -> nn.Module:

    interpolated_state_dict = copy.deepcopy(base_state_dict)

    # Interpolate parameters
    for name in interpolated_state_dict.keys():
        if base_state_dict[name].dim() <= 1:
            continue

        interpolated_param = (
            base_state_dict[name] + alpha * x_state_dict[name] + beta * y_state_dict[name]
        )
        interpolated_state_dict[name].copy_(interpolated_param)

    base_model.load_state_dict(interpolated_state_dict)
    return base_model


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
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


def create_loss_landscape_grid(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    resolution: int = 25,
    seed: int = 42,
) -> tuple:
    print("Generating random direction vectors with filter-wise normalization...")

    # Generate two random direction vectors using filter-wise normalization
    base_state_dict = copy.deepcopy(model.state_dict())
    x_direction = init_random_direction(base_state_dict, seed=seed)
    y_direction = init_random_direction(base_state_dict, seed=2 * seed)

    # Create coordinate grids
    alphas = np.linspace(x_min, x_max, resolution)
    betas = np.linspace(y_min, y_max, resolution)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    # Initialize loss grid
    loss_grid = np.zeros((resolution, resolution))

    print(f"Evaluating loss at {resolution}x{resolution} = {resolution**2} points...")

    # Evaluate loss at each grid point
    pbar = tqdm(total=resolution**2)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            pbar.update(1)
            # Create interpolated model
            interpolated_model = interpolate_models(
                model, base_state_dict, x_direction, y_direction, alpha, beta
            )

            # Evaluate loss
            loss = evaluate_loss(interpolated_model, test_loader, loss_fn)
            loss_grid[j, i] = loss  # Note: j, i for proper matrix indexing

    pbar.close()

    return alpha_grid, beta_grid, loss_grid


def plot_loss_landscape(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    loss_grid: np.ndarray,
    title: str = "Loss Landscape",
    save_path: str = None,
):
    """
    Plot the loss landscape using matplotlib.

    Args:
        alpha_grid: Alpha coordinate grid
        beta_grid: Beta coordinate grid
        loss_grid: Loss values at each grid point
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Create subplots
    fig = plt.figure(figsize=(15, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(alpha_grid, beta_grid, loss_grid, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_xlabel("Alpha (X Direction)")
    ax1.set_ylabel("Beta (Y Direction)")
    ax1.set_title(f"{title} - Contour Plot")
    ax1.grid(True, alpha=0.3)

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection="3d")
    surface = ax2.plot_surface(alpha_grid, beta_grid, loss_grid, cmap="viridis", alpha=0.8)
    ax2.set_xlabel("Alpha (X Direction)")
    ax2.set_ylabel("Beta (Y Direction)")
    ax2.set_zlabel("Loss")
    ax2.set_title(f"{title} - 3D Surface")

    plt.colorbar(surface, ax=ax2, shrink=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# Example usage function
def save_model_loss_landscape(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    path: str,
):
    """
    Complete pipeline to visualize loss landscape of a model.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        loss_fn: Loss function (defaults to CrossEntropyLoss)
        resolution: Grid resolution
        range_scale: Scale for the coordinate range
        device: Device to run on
        save_path: Path to save the plot
    """

    print("Starting loss landscape visualization...")

    loss_fn = nn.CrossEntropyLoss()
    resolution = 21
    range_scale = 1.0
    save_path = None

    # Create loss landscape grid
    alpha_grid, beta_grid, loss_grid = create_loss_landscape_grid(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        x_min=-range_scale,
        x_max=range_scale,
        y_min=-range_scale,
        y_max=range_scale,
        resolution=resolution,
    )

    # Save alpha and beta grids
    np.save(f"{path}/alpha_grid.npy", alpha_grid)
    np.save(f"{path}/beta_grid.npy", beta_grid)
    np.save(f"{path}/loss_grid.npy", loss_grid)

    # # Plot the results
    # plot_loss_landscape(
    #     alpha_grid, beta_grid, loss_grid, title="Neural Network Loss Landscape", save_path=save_path
    # )

    return alpha_grid, beta_grid, loss_grid


def get_paths(model_name: str, method_name: str, prune_amt: float) -> str:
    if model_name == "lenet":
        if method_name == "causal":
            path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_causalpruner_5_10_1_0.001_{prune_amt}/model.prune.final.ckpt"
        elif method_name == "mag":
            path = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_1/lenet_cifar10_magpruner_{prune_amt}/model.prune.final.ckpt"
        elif method_name == "noprune":
            path = "/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_noprune/model.trained.ckpt"

    elif model_name == "resnet20":
        if method_name == "causal":
            path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_causalpruner_10_10_1_0.001_0.9/model.trained.ckpt"
        elif method_name == "mag":
            path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_magpruner_0.9/model.trained.ckpt"
        elif method_name == "noprune":
            path = "/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_noprune/model.trained.ckpt"
    return path


def load_weights(model, path, flag_noprune: bool = False):
    if flag_noprune:
        model.load_state_dict(torch.load(path)["model"])
        return model

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


if __name__ == "__main__":
    model_name = "lenet"
    list_method_name = ["causal", "mag", "noprune"]
    list_prune_amt = [0.9, 0.7, 0.5, 0]
    # list_model_name = ["lenet", "resnet20"]
    for method_name, prune_amt in itertools.product(list_method_name, list_prune_amt):
        # Skip noprune for non-zero prune amounts
        if method_name == "noprune" and prune_amt != 0:
            continue
        if prune_amt == 0 and method_name != "noprune":
            continue
        print(f"Computing eigenvalues for {model_name} with {method_name} and {prune_amt}")

        path = get_paths(model_name, method_name, prune_amt)
        model = get_model(model_name, "cifar10", checkpoint_dir=None)
        model = load_weights(model, path, flag_noprune=method_name == "noprune")
        model.to(device)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", model_name, "./data")
        testset = tmp[1]
        testloader = DataLoader(testset, batch_size=1024, shuffle=True, num_workers=8)

        path_save = f"/home/think-server/Documents/cpn/results/loss_landscape2/{model_name}_{method_name}_{prune_amt}/"
        os.makedirs(path_save, exist_ok=True)
        save_model_loss_landscape(model, testloader, path_save)
