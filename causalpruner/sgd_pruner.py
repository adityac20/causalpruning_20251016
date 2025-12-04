import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
from functools import partial
import gc
import glob
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from causalpruner.average import AverageMeter
from causalpruner.base import Pruner, PrunerConfig
from causalpruner.causal_weights_trainer import (
    CausalWeightsTrainerConfig,
    get_causal_weights_trainer,
    CausalWeightsTrainerTorch, #CHANGE: Import for schedule calculation
)

_ZSTATS_PATTERN = "zstats.pth"


@dataclass
class ZStats:
    num_params: int
    mean: torch.Tensor
    std: torch.Tensor
    global_mean: torch.Tensor
    global_std: torch.Tensor


class ParamDataset(Dataset):
    def __init__(
        self,
        weights_base_dir: str,
        loss_base_dir: str,
        train_lr: float,
    ):
        self.weights_base_dir = weights_base_dir
        self.loss_base_dir = loss_base_dir
        file_pattern = "ckpt.*"
        self.num_items = min(
            len(glob.glob(file_pattern, root_dir=self.weights_base_dir)),
            len(glob.glob(file_pattern, root_dir=self.loss_base_dir)),
        )
        self.lr_scaling_factor = train_lr * train_lr
        self.loss_zstats = self._load_zstats(self.loss_base_dir)

    @torch.no_grad()
    def _load_zstats(self, base_dir: str) -> ZStats:
        zstats_dict = torch.load(os.path.join(base_dir, _ZSTATS_PATTERN))
        zstats = ZStats(
            num_params=zstats_dict["num_params"],
            global_mean=zstats_dict["global_mean"],
            global_std=zstats_dict["global_std"],
            mean=zstats_dict["mean"],
            std=zstats_dict["std"],
        )
        return zstats

    @torch.no_grad()
    def __len__(self) -> int:
        return self.num_items

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_weights = self.get_delta_param(self.weights_base_dir, idx)
        delta_weights /= self.lr_scaling_factor
        delta_loss = self.get_delta_param(self.loss_base_dir, idx)
        delta_loss /= self.loss_zstats.mean
        return delta_weights, delta_loss

    @torch.no_grad()
    def get_delta_param(
        self, dir: str, idx: int, zstats: Optional[ZStats] = None
    ) -> torch.Tensor:
        file_path = os.path.join(dir, f"ckpt.{idx}")
        val = torch.load(file_path)
        val = torch.nan_to_num(val, nan=0, posinf=0, neginf=0)
        return val


class ZStatsComputer:
    def __init__(self):
        self.sum_x = None
        self.sum_x_squared = None
        self.num_items = 0
        self.num_params = 0
        self.mean_ = None
        self.std_ = None
        self.global_sum_x = 0.0
        self.global_sum_x_squared = 0.0
        self.global_num_items = 0
        self.global_mean_ = None
        self.global_std_ = None

    @torch.no_grad()
    def to(self, device: torch.device) -> "ZStatsComputer":
        def move_tensor(tensor):
            if tensor is not None and tensor.device != device:
                return tensor.to(device)
            else:
                return tensor

        self.sum_x = move_tensor(self.sum_x)
        self.sum_x_squared = move_tensor(self.sum_x_squared)
        self.mean_ = move_tensor(self.mean_)
        self.std_ = move_tensor(self.std_)
        return self

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        if self.sum_x is None:
            self.num_params = torch.numel(x)
            self.sum_x = torch.zeros_like(x)
            self.sum_x_squared = torch.zeros_like(x)
        x_squared = torch.square(x)
        self.sum_x += x
        self.sum_x_squared += x_squared
        self.num_items += 1
        self.global_sum_x += torch.sum(x)
        self.global_sum_x_squared += torch.sum(x_squared)
        self.global_num_items += torch.count_nonzero(x)

    @property
    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        if self.mean_ is None:
            self.mean_ = self.sum_x / self.num_items
        return self.mean_

    @property
    @torch.no_grad()
    def global_mean(self) -> torch.Tensor:
        if self.global_mean_ is None:
            self.global_mean_ = self.global_sum_x / self.global_num_items
        return self.global_mean_

    @property
    @torch.no_grad()
    def std(self) -> torch.Tensor:
        if self.std_ is None:
            variance = (self.sum_x_squared / self.num_items) - torch.square(self.mean)
            std_dev = torch.sqrt(variance)
            self.std_ = std_dev
        return self.std_

    @property
    @torch.no_grad()
    def global_std(self) -> torch.Tensor:
        if self.global_std_ is None:
            variance = (
                self.global_sum_x_squared / self.global_num_items
            ) - torch.square(self.global_mean)
            std_dev = torch.sqrt(variance)
            self.global_std_ = std_dev
        return self.global_std_


class DeltaComputer:
    def __init__(
        self,
        transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = torch.nn.Identity(),
    ):
        self.first_tensor = None
        self.second_tensor = None
        self.transform = transform
        self.zstats_computer = ZStatsComputer()

    @torch.no_grad()
    def add_first(self, weight: torch.Tensor):
        self.first_tensor = weight

    @torch.no_grad()
    def add_second(self, weight: torch.Tensor):
        self.second_tensor = weight

    @torch.no_grad()
    def get_delta(self) -> Optional[torch.Tensor]:
        if self.first_tensor is None or self.second_tensor is None:
            return None
        delta = self.second_tensor - self.first_tensor
        result = self.transform(delta)
        self.zstats_computer.add(result)
        return result


@dataclass
class SGDPrunerConfig(PrunerConfig):
    prune_dataloader: DataLoader
    prune_optimizer_lr: float
    num_prune_iterations: int
    num_prune_epochs: int
    threaded_checkpoint_writer: bool
    delete_checkpoint_dir_after_training: bool
    trainer_config: CausalWeightsTrainerConfig
    num_batches_in_epoch: int = -1
    loss_fn: Callable = partial(F.cross_entropy, label_smoothing=0.1)
    return_masks: bool = False


class SGDPruner(Pruner):
    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)

        self.prune_dataloader = self.fabric.setup_dataloaders(config.prune_dataloader)

        self.loss_checkpoint_dir = os.path.join(self.checkpoint_dir, "loss")
        self.weights_checkpoint_dir = os.path.join(self.checkpoint_dir, "weights")

        if self.fabric.is_global_zero:
            if config.start_clean and os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.loss_checkpoint_dir, exist_ok=True)
            os.makedirs(self.weights_checkpoint_dir, exist_ok=True)
            self.threaded_checkpoint_writer = config.threaded_checkpoint_writer
            if self.threaded_checkpoint_writer:
                self.checkpointer = ThreadPoolExecutor()
                self.checkpoint_futures = []
        self.fabric.barrier()

        self.num_params = 0
        self.params_to_dims = dict()
        for param in self.params:
            self.params_to_dims[param] = np.prod(
                self.modules_dict[param].weight.size(), dtype=int
            )
            self.num_params += self.params_to_dims[param]
        self.trainer_config = config.trainer_config

        self.prune_optimizer = optim.SGD(
            config.model.parameters(), lr=config.prune_optimizer_lr
        )
        self.prune_optimizer_lr = config.prune_optimizer_lr
        self.prune_optimizer = self.fabric.setup_optimizers(self.prune_optimizer)

        self.verbose = config.verbose

    def run_prune_iteration(self) -> None:
        super().run_prune_iteration()

        #CHANGE: Create a dynamic pruning schedule for this iteration
        self._create_prune_schedule()
        #ENDCHANGE

        self.start_iteration()
        config = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.model = config.model.to(device)
        num_batches_in_epoch = config.num_batches_in_epoch
        prune_pbar = tqdm(
            range(config.num_prune_epochs),
            leave=False,
            desc="Pruning",
            dynamic_ncols=True,
        )
        for epoch in prune_pbar:
            config.model.train()
            loss_avg = AverageMeter(self.fabric)
            epoch_pbar = tqdm(
                self.prune_dataloader,
                leave=False,
                desc=f"Prune epoch: {epoch}",
                dynamic_ncols=True,
            )
            batch_counter = 0
            for inputs, labels in epoch_pbar:
                self.prune_optimizer.zero_grad(set_to_none=True)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.provide_loss_before_step(loss)
                self.fabric.backward(loss)
                loss_avg.update(loss)
                self.prune_optimizer.step()
                with torch.no_grad():
                    outputs = config.model(inputs)
                    loss = config.loss_fn(outputs, labels)
                    self.provide_loss_after_step(loss)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
                batch_counter += 1
            epoch_pbar.close()
            loss = loss_avg.mean()
            iter_str = f"{self.iteration}/{config.num_prune_iterations}"
            epoch_str = f"{epoch + 1}/{config.num_prune_epochs}"
            prune_pbar.set_description(
                f"Prune: Iteration {iter_str}; "
                + f"Epoch: {epoch_str}; "
                + f"Loss/Train: {loss:.4f}"
            )
        prune_pbar.close()
        print("Computing masks")
        self.compute_masks()
        self.reset_weights()
        self.reset_params()

    #CHANGE: Add method to create a pruning schedule per iteration
# In causalpruner/sgd_pruner.py

    def _create_prune_schedule(self):
        dummy_trainer = CausalWeightsTrainerTorch(
            self.trainer_config,
            num_params=1,
            initial_mask=torch.ones(1),
            prune_iteration=self.iteration + 1,
            num_prune_iterations=self.config.num_prune_iterations,
            verbose=False,
        )
        prune_amount_this_iteration_global = dummy_trainer.prune_amount_this_iteration

        num_layers = len(self.params)
        min_scale, max_scale = 0.5, 1.5
        self.layer_prune_amounts = {}
        tqdm.write("Layer pruning amounts for this iteration:")
        for i, param in enumerate(self.params):
            if num_layers > 1:
                scale = min_scale + (max_scale - min_scale) * (i / (num_layers - 1))
            else:
                scale = 1.0
            
            amount = prune_amount_this_iteration_global * scale
            
            #CHANGE: Clamp the pruning amount to a valid range [0.0, 0.999]
            # An amount >= 1.0 is invalid and will cause a crash.
            amount = max(0.0, min(amount, 0.95))
            #ENDCHANGE
            
            self.layer_prune_amounts[param] = amount
            tqdm.write(f"- {param}: {amount:.4f}")


    @torch.no_grad()
    def start_iteration(self):
        if self.fabric.is_global_zero:
            iteration_name = f"{self.iteration}"
            self.loss_dir = os.path.join(self.loss_checkpoint_dir, iteration_name)
            os.makedirs(self.loss_dir, exist_ok=True)
            self.weights_dir = os.path.join(
                self.weights_checkpoint_dir, iteration_name
            )
            os.makedirs(self.weights_dir, exist_ok=True)
            
            #CHANGE: Create per-layer delta computers and directories
            self.delta_weights_computers = {
                param: DeltaComputer(transform=torch.square) for param in self.params
            }
            self.layer_weights_dirs = {}
            for param in self.params:
                layer_dir = os.path.join(self.weights_dir, param)
                os.makedirs(layer_dir, exist_ok=True)
                self.layer_weights_dirs[param] = layer_dir
            #ENDCHANGE
            
            self.delta_loss_computer = DeltaComputer()
            self.checkpoint_futures = []
            self.init_model_state = copy.deepcopy(self.config.model.state_dict())
        self.fabric.barrier()


    @torch.no_grad()
    def provide_loss_before_step(self, loss: torch.tensor) -> None:
        if self.fabric.is_global_zero:
            torch.cuda.synchronize()
            self.delta_loss_computer.add_first(loss)
            #CHANGE: Capture weights per layer
            for param in self.params:
                weight = self.modules_dict[param].weight.detach().clone()
                self.delta_weights_computers[param].add_first(weight)
            #ENDCHANGE
            torch.cuda.synchronize()
        self.fabric.barrier()

    @torch.no_grad()
    def provide_loss_after_step(self, loss: torch.tensor) -> None:
        if self.fabric.is_global_zero:
            torch.cuda.synchronize()
            self.delta_loss_computer.add_second(loss)
            delta_loss = self.delta_loss_computer.get_delta().to("cpu")
            if delta_loss is not None:
                self.write_tensor(delta_loss, self._get_checkpoint_path(self.loss_dir))
            
            #CHANGE: Capture and write delta_weights per layer
            for param in self.params:
                weight = self.modules_dict[param].weight.detach().clone()
                self.delta_weights_computers[param].add_second(weight)
                delta_weights = self.delta_weights_computers[param].get_delta()
                if delta_weights is not None:
                    layer_dir = self.layer_weights_dirs[param]
                    self.write_tensor(
                        delta_weights.flatten().to("cpu"),
                        self._get_checkpoint_path(layer_dir),
                    )
            #ENDCHANGE
            torch.cuda.synchronize()
        self.fabric.barrier()
        self.counter += 1

    @torch.no_grad()
    def write_tensor(self, tensor: torch.Tensor, path: str):
        if not self.threaded_checkpoint_writer:
            torch.save(tensor, path)
            return
        while psutil.virtual_memory().percent >= 99.5:
            time.sleep(0.1)
        future = self.checkpointer.submit(torch.save, tensor, path)
        self.checkpoint_futures.append(future)

    def compute_masks(self):
        #CHANGE: Get masks from the new layer-wise training function and apply them
        new_masks = self.train_pruning_weights()
        self.config.model.load_state_dict(self.init_model_state)
        with torch.no_grad():
            for module_name, module in self.modules_dict.items():
                prune.custom_from_mask(module, "weight", new_masks[module_name])
        #ENDCHANGE
        torch.cuda.empty_cache()
        gc.collect()

    #CHANGE: Re-implement this function to loop through layers and train a separate trainer for each
    def train_pruning_weights(self) -> dict[str, torch.Tensor]:
        if self.threaded_checkpoint_writer and self.fabric.is_global_zero:
            concurrent.futures.wait(self.checkpoint_futures)
            del self.checkpoint_futures
            self.checkpoint_futures = []
            self._write_zscaling_params()
        self.fabric.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        new_masks = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for param in tqdm(self.params, desc="Analyzing Layers"):
            param_module = self.modules_dict[param]
            if not hasattr(param_module, "weight_mask"):
                initial_mask = torch.ones_like(param_module.weight)
            else:
                initial_mask = param_module.weight_mask.detach()

            layer_weights_dir = self.layer_weights_dirs[param]
            dataset = ParamDataset(
                layer_weights_dir,
                self.loss_dir,
                self.prune_optimizer_lr,
            )

            # Skip if no data was collected for this layer
            if len(dataset) == 0:
                tqdm.write(f"Skipping layer {param}, no data collected.")
                new_masks[param] = initial_mask.clone()
                continue
            
            # Use a local copy of the config to modify prune_amount
            local_trainer_config = copy.deepcopy(self.trainer_config)
            local_trainer_config.prune_amount = self.layer_prune_amounts[param]

            trainer = get_causal_weights_trainer(
                local_trainer_config,
                self.params_to_dims[param],
                initial_mask.flatten(),
                self.iteration + 1,
                self.config.num_prune_iterations,
                self.verbose,
            )

            batch_size = local_trainer_config.batch_size
            if batch_size < 0 or not trainer.supports_batch_training():
                batch_size = len(dataset)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=local_trainer_config.pin_memory,
                shuffle=True,
                num_workers=local_trainer_config.num_dataloader_workers,
                persistent_workers=local_trainer_config.num_dataloader_workers > 0,
            )

            trainer.fit(dataloader)
            
            mask_flat = trainer.get_non_zero_weights()
            new_masks[param] = mask_flat.to(device).reshape_as(param_module.weight)

            # Clean up layer-specific checkpoints
            if self.fabric.is_global_zero:
                self._delete_checkpoint_dir(layer_weights_dir)

        if self.fabric.is_global_zero:
            self._delete_checkpoint_dir(self.loss_dir)
        self.fabric.barrier()
        
        return new_masks
    #ENDCHANGE

    def _delete_checkpoint_dir(self, dirpath: str):
        if not self.config.delete_checkpoint_dir_after_training:
            return
        if os.path.exists(dirpath): #CHANGE: Add check for existence
            shutil.rmtree(dirpath)


    @torch.no_grad()
    def get_masks(self) -> dict[str, torch.Tensor]:
        masks = {}
        for param_name, module in self.modules_dict.items():
            # The prune utility attaches the mask as an attribute named 'weight_mask'
            if hasattr(module, "weight_mask"):
                # Make a detached copy to avoid holding onto computation graph history
                masks[param_name] = module.weight_mask.detach().clone()
            else:
                # If the layer has never been pruned, its mask is all ones.
                masks[param_name] = torch.ones_like(module.weight)
        return masks

    def _get_checkpoint_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, f"ckpt.{self.counter}")

    def _write_zscaling_params(self):
        self._write_zscaling_params_from_computer(
            self.delta_loss_computer, self.loss_dir
        )
        #CHANGE: Write z-stats for each layer
        for param, computer in self.delta_weights_computers.items():
            layer_dir = self.layer_weights_dirs[param]
            self._write_zscaling_params_from_computer(computer, layer_dir)
        #ENDCHANGE

    def _write_zscaling_params_from_computer(
        self, computer: DeltaComputer, dir_path: str
    ):
        dir_path = os.path.join(dir_path, _ZSTATS_PATTERN)
        zstats_computer = computer.zstats_computer.to("cpu")
        torch.save(
            {
                "num_params": zstats_computer.num_params,
                "mean": zstats_computer.mean,
                "std": zstats_computer.std,
                "global_mean": zstats_computer.global_mean,
                "global_std": zstats_computer.global_std,
            },
            dir_path,
        )