import sys
import os
import itertools
import pdb

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
    if flag_noprune:
        try:
            state_dict = torch.load(path)["model"]
        except:
            state_dict = torch.load(path)
        model.load_state_dict(state_dict)
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


def load_weights_v2(model, path, flag_fc_prune: bool = True):
    try:
        state_dict = torch.load(path)["model"]
    except:
        state_dict = torch.load(path)

    # Apply pruning structure to match the checkpoint
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.0)
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.0)

    # Load the checkpoint state dict (this includes weight_orig and weight_mask)
    model.load_state_dict(state_dict, strict=False)

    return model


def apply_pruning(model, flag_fc_prune: bool = True):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")
        if isinstance(module, nn.Linear) and flag_fc_prune:
            prune.remove(module, "weight")
    return model


def train_model(model, trainloader, testloader, num_epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=num_epochs,
        pct_start=0.3,
    )
    optimizer.zero_grad()
    best_acc = 0
    pbar_epoch = tqdm(total=num_epochs, leave=False)
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(total=len(trainloader), leave=False)
        loss_total = 0
        num_samples = 0
        for data in trainloader:
            optimizer.zero_grad()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(images)
            num_samples += len(images)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} loss: {loss_total / num_samples:.4f}")
        pbar.close()
        scheduler.step()
        acc = get_accuracy(model, testloader)
        if acc > best_acc:
            best_acc = acc
        pbar_epoch.update(1)
        pbar_epoch.set_description(
            f"Epoch {epoch} loss: {loss_total / num_samples}, acc: {acc}, best_acc: {best_acc}"
        )
    pbar_epoch.close()
    return model, best_acc


def train_model_imagenet(model, trainloader, testloader, num_epochs=1):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer.zero_grad()
    best_acc = 0
    pbar_epoch = tqdm(total=num_epochs, leave=False)
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(total=len(trainloader), leave=False)
        loss_total = 0
        num_samples = 0
        pbar_batch = tqdm(total=len(trainloader), leave=False)
        for data in trainloader:
            optimizer.zero_grad()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(images)
            num_samples += len(images)
            pbar_batch.update(1)
            pbar_batch.set_description(f"Epoch {epoch} loss: {loss_total / num_samples:.4f}")
        pbar_batch.close()
        acc = get_accuracy(model, testloader)
        if acc > best_acc:
            best_acc = acc
        pbar_epoch.update(1)
        pbar_epoch.set_description(
            f"Epoch {epoch} loss: {loss_total / num_samples}, acc: {acc}, best_acc: {best_acc}"
        )
    pbar_epoch.close()
    return model, best_acc


def results_vgg_tinyimagenet():
    with open("./results/exp01v4.txt", "w") as f:
        f.write("model,acc,sparsity,best_acc\n")

    for i in [20, 29]:
        base_path = f"/home/think-server/Documents/cpn/checkpoints_all_results/vgglike_trained_tinyimagenet_causalpruner_30_60_1_0.001_0.98/model.prune.{i}.ckpt"

        # Load the model
        model = get_model("vgglike", "tinyimagenet", checkpoint_dir=None)
        if not os.path.exists(base_path):
            print(f"Model not found at {base_path}")
            continue
        model = load_weights_v2(model, base_path).to(device)

        # Get the datasets
        tmp = get_dataset("tinyimagenet", "vgglike", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Train the model
        model, best_acc = train_model(model, trainloader, testloader, num_epochs=300)

        # apply pruning
        model = apply_pruning(model)

        # Get the scores
        model.eval()
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")

        with open("./results/exp01v4.txt", "a") as f:
            f.write(f"{i},{acc},{sparsity},{best_acc}\n")


def results_vgg_cifar10():
    with open("./results/exp01_vgg_cifar10_v2.txt", "w") as f:
        f.write("model,acc,sparsity,best_acc\n")

    for i in [20, 27]:
        base_path = f"/home/think-server/Documents/cpn/checkpoints_all_results/vgglike_trained_cifar10_causalpruner_30_60_10_0.001_0.98/model.prune.{i}.ckpt"

        # Load the model
        model = get_model("vgglike", "cifar10", checkpoint_dir=None)
        if not os.path.exists(base_path):
            print(f"Model not found at {base_path}")
            continue
        model = load_weights_v2(model, base_path).to(device)

        # Get the datasets
        tmp = get_dataset("cifar10", "vgglike", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Train the model
        model, best_acc = train_model(model, trainloader, testloader, num_epochs=300)

        # apply pruning
        model = apply_pruning(model)

        # Get the scores
        model.eval()
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")

        with open("./results/exp01_vgg_cifar10_v2.txt", "a") as f:
            f.write(f"{i},{acc},{sparsity},{best_acc}\n")


def results_mobilenet_imagenet():
    path = "/home/think-server/Documents/cpn/checkpoints_all_results2/mobilenet_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.prune.final.ckpt"
    path_save = f"/home/think-server/Documents/cpn/checkpoints_all_results2/mobilenet_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.trained2.ckpt"

    # Load the model
    model = get_model("mobilenet_untrained", "imagenet", checkpoint_dir=None)
    model = load_weights_v2(model, path).to(device)
    print("Loaded model...")

    with open("./results/exp01_mobilenet_imagenet_v3.txt", "w") as f:
        f.write("model,acc,sparsity,best_acc\n")

    # Get the datasets
    tmp = get_dataset("imagenet", "mobilenet_untrained", "./data")
    trainset = tmp[0]
    testset = tmp[1]
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1024, shuffle=True, num_workers=8
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)

    # Train the model
    model, best_acc = train_model_imagenet(model, trainloader, testloader, num_epochs=5)

    # apply pruning
    model = apply_pruning(model)

    torch.save(model.state_dict(), path_save)

    # Get the scores
    model.eval()
    acc = get_accuracy(model, testloader)
    best_acc = acc
    sparsity = compute_sparsity(model)
    print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")


def results_resnet50_imagenet():
    path = "/home/think-server/Documents/cpn/checkpoints_all_results/resnet50_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.prune.final.ckpt"
    path_save = f"/home/think-server/Documents/cpn/checkpoints_all_results/resnet50_trained_imagenet_causalpruner_1_1_1_0.001_0.8/model.trained2.ckpt"

    # Load the model
    model = get_model("resnet50_untrained", "imagenet", checkpoint_dir=None)
    model = load_weights_v2(model, path).to(device)
    print("Loaded model...")

    # Get the datasets
    tmp = get_dataset("imagenet", "resnet50_untrained", "./data")
    trainset = tmp[0]
    testset = tmp[1]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=8)

    # Train the model
    model, best_acc = train_model_imagenet(model, trainloader, testloader, num_epochs=30)

    # apply pruning
    model = apply_pruning(model)

    torch.save(model.state_dict(), path_save)

    # Get the scores
    model.eval()
    acc = get_accuracy(model, testloader)
    best_acc = acc
    sparsity = compute_sparsity(model)
    print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")


def results_resnet20_cifar10():
    with open("./results/exp01_resnet20_cifar10_v1.txt", "w") as f:
        f.write("model,acc,sparsity,best_acc\n")

    for i in np.arange(40, -1, -3):
        base_path = f"/home/think-server/Documents/cpn/checkpoints_all_results/resnet20_trained_cifar10_causalpruner_40_60_10_0.001_0.98/model.prune.{i}.ckpt"

        # Load the model
        model = get_model("resnet20", "cifar10", checkpoint_dir=None)
        if not os.path.exists(base_path):
            print(f"Model not found at {base_path}")
            continue
        model = load_weights_v2(model, base_path).to(device)

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet20", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Train the model
        model, best_acc = train_model(model, trainloader, testloader, num_epochs=10)

        # apply pruning
        model = apply_pruning(model)

        # Get the scores
        model.eval()
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")

        with open("./results/exp01_resnet20_cifar10_v1.txt", "a") as f:
            f.write(f"{i},{acc},{sparsity},{best_acc}\n")


def train_resnet20_cifar10():
    with open("./results/exp01_resnet20_cifar10_v2.txt", "w") as f:
        f.write("model,acc,sparsity,best_acc\n")

    causal_path = f"/home/think-server/Documents/cpn/checkpoints_all_results2/resnet20_cifar10_causalpruner_1_1_1_0.001_0.9/model.trained.ckpt"
    mag_path = f"/home/think-server/Documents/cpn/checkpoints_all_results/resnet20_trained_cifar10_magpruner_0.9/model.trained.ckpt"
    base_path = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_1/resnet20_cifar10_noprune/model.trained.ckpt"

    # Load the model
    model = get_model("resnet20", "cifar10", checkpoint_dir=None)
    model = load_weights_v2(model, causal_path).to(device)

    # Get the datasets
    tmp = get_dataset("cifar10", "resnet20", "./data")
    trainset = tmp[0]
    testset = tmp[1]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    best_acc = 0
    # Train the model
    model, best_acc = train_model(model, trainloader, testloader, num_epochs=300)

    # apply pruning
    model = apply_pruning(model)

    causal_path_save = f"/home/think-server/Documents/cpn/checkpoints_all_results2/resnet20_cifar10_causalpruner_1_1_1_0.001_0.9/model.trained.ckpt"
    torch.save(model.state_dict(), causal_path_save)

    # Get the scores
    model.eval()
    acc = get_accuracy(model, testloader)
    sparsity = compute_sparsity(model)
    print(f"Sparsity: {sparsity}, Acc: {acc}, best acc: {best_acc}")


if __name__ == "__main__":
    train_resnet20_cifar10()

    # Sparsity: 0.7979252290030028, Acc: 0.56822, best acc: 0.56822
