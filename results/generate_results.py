import sys
import os
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tests.models import get_model
from tests.datasets import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def compute_sparsity(model):
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total


def load_pruned_weights(model, path):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.0)

    model.load_state_dict(torch.load(path)["model"])

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if prune.is_pruned(module):
                prune.remove(module, "weight")

    return model


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


def ablation_scores_resnet20_cifar10():
    with open("./results/scores_resnet20cifar10_ablation.csv", "w") as f:
        f.write("N_PRE,N_ITER,N_EPOCH_PRUNE,PRUNE_AMT,Accuracy,Sparsity\n")

    list_N_PRE = [0, 10, 20, 30]
    list_N_ITER = [1, 5, 10, 20]
    list_N_EPOCH_PRUNE = [5, 10, 20]
    list_PRUNE_AMT = [0.9, 0.98]

    for N_PRE, N_ITER, N_EPOCH_PRUNE, PRUNE_AMT in itertools.product(
        list_N_PRE, list_N_ITER, list_N_EPOCH_PRUNE, list_PRUNE_AMT
    ):
        # fpath = f"/home/think-server/Documents/cpn/checkpoints_ablation/prune_amount_{PRUNE_AMT}/N_PRE_{N_PRE}/resnet20_cifar10_causalpruner_{N_ITER}_0_{N_EPOCH_PRUNE}_0_{PRUNE_AMT}/model.trained.ckpt"
        fpath = f"/home/think-server/Documents/cpn/checkpoints_ablation/prune_amount_{PRUNE_AMT}/N_PRE_{N_PRE}/resnet20_cifar10_causalpruner_{N_ITER}_10_{N_EPOCH_PRUNE}_0.001_{PRUNE_AMT}/model.trained.ckpt"

        # Load the model
        model = get_model("resnet20", "cifar10", checkpoint_dir=None)
        if not os.path.exists(fpath):
            print(f"Model not found at {fpath}")
            continue

        model = load_pruned_weights(model, fpath).to(device)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet20", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Get the scores
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        with open("./results/scores_resnet20cifar10_ablation.csv", "a") as f:
            f.write(f"{N_PRE},{N_ITER},{N_EPOCH_PRUNE},{PRUNE_AMT},{acc},{sparsity}\n")


def phase_shift_lenetcifar10():
    with open("./results/scores_lenetcifar10.csv", "w") as f:
        f.write("rep,algorithm,prune_amt,Accuracy,Sparsity\n")
    list_PRUNE_AMT = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_PRUNE_AMT += [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
    list_PRUNE_AMT += [0.99, 0.995, 0.999]
    list_REP = range(1, 7)
    list_PRUNER = ["causalpruner", "magpruner"]

    for rep, pruner, prune_amt in itertools.product(list_REP, list_PRUNER, list_PRUNE_AMT):
        if pruner == "causalpruner":
            # fpath = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_NITER10/lenetcifar10_{rep}/lenet_cifar10_causalpruner_5_10_1_0.001_{prune_amt}/model.trained.ckpt"
            fpath = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10/lenetcifar10_{rep}/lenet_cifar10_causalpruner_1_10_1_0.001_{prune_amt}/model.prune.final.ckpt"
        elif pruner == "magpruner":
            # fpath = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10_v3/lenetcifar10_{rep}/lenet_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
            fpath = f"/home/think-server/Documents/cpn/checkpoints/lenet_cifar10/lenetcifar10_{rep}/lenet_cifar10_magpruner_{prune_amt}/model.prune.final.ckpt"
        # Load the model
        model = get_model("lenet", "cifar10", checkpoint_dir=None)
        if not os.path.exists(fpath):
            print(f"Model not found at {fpath}")
            continue

        model = load_pruned_weights(model, fpath).to(device)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "lenet", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Get the scores
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        with open("./results/scores_lenetcifar10.csv", "a") as f:
            f.write(f"{rep},{pruner},{prune_amt},{acc},{sparsity}\n")


def scores_resnet20_cifar10():
    fpath_write = "./results/scores_resnet20cifar10.csv"
    with open(fpath_write, "w") as f:
        f.write("rep,algorithm,N_ITER,prune_amt,Accuracy,Sparsity\n")
    list_PRUNE_AMT = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    list_PRUNER = ["causalpruner", "magpruner"]
    list_REP = range(1, 3)
    list_N_ITER = [1, 10]

    for rep, pruner, N_ITER, prune_amt in itertools.product(
        list_REP, list_PRUNER, list_N_ITER, list_PRUNE_AMT
    ):
        if N_ITER == 1 and pruner == "magpruner":
            continue  # MAG pruner with 1 iteration is not computed

        if pruner == "causalpruner":
            fpath = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_{rep}/resnet20_cifar10_causalpruner_{N_ITER}_10_1_0.001_{prune_amt}/model.trained.ckpt"
        elif pruner == "magpruner":
            fpath = f"/home/think-server/Documents/cpn/checkpoints/resnet20_cifar10/resnet20_cifar10_{rep}/resnet20_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        # Load the model
        model = get_model("resnet20", "cifar10", checkpoint_dir=None)
        if not os.path.exists(fpath):
            print(f"Model not found at {fpath}")
            continue

        model = load_pruned_weights(model, fpath).to(device)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet20", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Get the scores
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        with open(fpath_write, "a") as f:
            f.write(f"{rep},{pruner},{N_ITER},{prune_amt},{acc},{sparsity}\n")


def scores_resnet18_cifar10():
    with open("./results/scores_resnet18cifar10.csv", "w") as f:
        f.write("algorithm,prune_amt,Accuracy,Sparsity\n")
    list_PRUNE_AMT = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    list_PRUNER = ["causalpruner", "magpruner"]

    for pruner, prune_amt in itertools.product(list_PRUNER, list_PRUNE_AMT):

        if pruner == "causalpruner":
            fpath = f"/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_causalpruner_10_0_10_0_{prune_amt}/model.trained.ckpt"
        elif pruner == "magpruner":
            fpath = f"/home/think-server/Documents/cpn/checkpoints/resnet18_cifar10_magpruner_{prune_amt}/model.trained.ckpt"
        # Load the model
        model = get_model("resnet18", "cifar10", checkpoint_dir=None)
        if not os.path.exists(fpath):
            print(f"Model not found at {fpath}")
            continue

        model = load_pruned_weights(model, fpath).to(device)
        model.eval()

        # Get the datasets
        tmp = get_dataset("cifar10", "resnet18", "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        # Get the scores
        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        with open("./results/scores_resnet18cifar10.csv", "a") as f:
            f.write(f"{pruner},{prune_amt},{acc},{sparsity}\n")


def scores_list():
    base_path = "/home/think-server/Documents/cpn/checkpoints_all_results/"
    lenet_mnist = "lenet_mnist_causalpruner_30_10_1_0.001_0.993/model.trained.ckpt"
    vgg_cifar10 = "vgglike_cifar10_causalpruner_30_10_1_0.001_0.98/model.trained.ckpt"
    vgg_tinyimagenet = "vgglike_tinyimagenet_causalpruner_30_10_1_0.001_0.98/model.trained.ckpt"

    dict_models = {
        ("lenet", "mnist"): os.path.join(base_path, lenet_mnist),
        ("vgglike", "cifar10"): os.path.join(base_path, vgg_cifar10),
        ("vgglike", "tinyimagenet"): os.path.join(base_path, vgg_tinyimagenet),
    }

    with open("./results/scores_list.csv", "w") as f:
        f.write("model_name,dataset_name,acc,sparsity\n")

    for (model_name, dataset_name), path in dict_models.items():
        model = get_model(model_name, dataset_name, checkpoint_dir=None)
        model = load_pruned_weights(model, path).to(device)
        model.eval()

        tmp = get_dataset(dataset_name, model_name, "./data")
        trainset = tmp[0]
        testset = tmp[1]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        acc = get_accuracy(model, testloader)
        sparsity = compute_sparsity(model)
        with open("./results/scores_list.csv", "a") as f:
            f.write(f"{model_name},{dataset_name},{acc},{sparsity}\n")


if __name__ == "__main__":
    # phase_shift_lenetcifar10()
    # scores_resnet20_cifar10()
    ablation_scores_resnet20_cifar10()
    # scores_list()
