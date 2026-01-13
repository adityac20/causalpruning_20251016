from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from lightning.fabric import Fabric
import numpy as np
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from causalpruner import lrrt
from causalpruner.average import AverageMeter
from causalpruner.lasso_optimizer import LassoSGD

class SegmentedLinear(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.total_in = sum(model_dims)
        self.num_models = len(model_dims)
        self.model_dims = model_dims # Store for normalization later

        # One weight for every single input feature
        self.weight = nn.Parameter(torch.zeros(self.total_in)) # Changed to nn.Parameter for correct registration
        
        indices = []
        start = 0
        self.layer_slices = [] # Store slices for easy indexing
        for i, dim in enumerate(model_dims):
            indices.append(torch.full((dim,), i, dtype=torch.long))
            self.layer_slices.append(slice(start, start + dim))
            start += dim
            
        self.register_buffer("model_indices", torch.cat(indices))

    def forward(self, x):
        products = x * self.weight

        batch_size = x.shape[0]
        out = torch.zeros(batch_size, self.num_models, device=x.device)

        target_index = self.model_indices.expand(batch_size, -1)
        out.scatter_add_(1, target_index, products)
        return out


@dataclass
class CausalWeightsTrainerConfig:
    fabric: Fabric
    init_lr: float
    batch_size: int
    num_dataloader_workers: int
    pin_memory: bool
    l1_regularization_coeff: float
    prune_amount: float
    max_iter: int
    loss_tol: float
    num_iter_no_change: int
    backend: Literal["sklearn", "torch"] = "torch"


class CausalWeightsTrainer(ABC):
    def __init__(self, config: CausalWeightsTrainerConfig):
        self.config = config
        self.init_lr = config.init_lr
        self.l1_regularization_coeff = config.l1_regularization_coeff
        self.prune_amount = config.prune_amount
        self.max_iter = config.max_iter
        self.loss_tol = config.loss_tol
        self.num_iter_no_change = config.num_iter_no_change

    def supports_batch_training(self) -> bool:
        return True

    @abstractmethod
    def fit(self, dataloader: DataLoader, num_epochs: int = -1) -> int:
        raise NotImplementedError("Use the sklearn or pytorch version")

    @abstractmethod
    def get_non_zero_weights(self) -> torch.Tensor:
        raise NotImplementedError("Use the sklearn or pytorch version")


class CausalWeightsTrainerSklearn(CausalWeightsTrainer):
    def __init__(self, config: CausalWeightsTrainerConfig):
        super().__init__(config)
        self.trainer = SGDRegressor(
            loss="squared_error",
            penalty="l1",
            alpha=self.l1_regularization_coeff,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.loss_tol,
            n_iter_no_change=self.num_iter_no_change,
            shuffle=True,
        )

    def supports_batch_training(self) -> bool:
        return False

    def fit(self, dataloader: DataLoader, num_epochs: int = -1) -> int:
        X, Y = next(iter(dataloader))
        X = X.cpu().numpy()
        Y = np.ravel(Y.cpu().numpy())
        self.trainer.fit(X, Y)
        return self.trainer.n_iter_

    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = np.copy(self.trainer.coef_)
        mask = np.atleast_2d(mask)
        mask = np.all(mask == 0, axis=0)
        mask = np.where(mask, 0, 1)
        return torch.tensor(mask)


class CausalWeightsTrainerTorch(CausalWeightsTrainer):
    def __init__(
        self,
        config: CausalWeightsTrainerConfig,
        num_params: int,
        initial_mask: torch.Tensor,
        prune_iteration: int,
        num_prune_iterations: int,
        verbose: bool,
        model_dims: list[int] = None 
    ):
        super().__init__(config)
        self.fabric = config.fabric
        self.num_params = num_params
        self.verbose = verbose
        self.model_dims = model_dims # CHANGE: Store dimensions

        # CHANGE: Use SegmentedLinear instead of standard Linear
        if self.model_dims is None:
             raise ValueError("model_dims must be provided for SegmentedLinear")
             
        self.layer = SegmentedLinear(self.model_dims)
        
        # CHANGE: Initialize weights check with small random weights
        nn.init.constant_(self.layer.weight, 0.0) 

        if initial_mask.device != self.layer.weight.device:
            initial_mask = initial_mask.to(self.layer.weight.device)
        
        # CHANGE: We cannot use prune.custom_from_mask easily on the custom module immediately
        # We will manage masking manually or map it to the 'weight' parameter
        # For simplicity, we register the buffer manually to mimic pruning behavior
        self.layer.register_buffer('weight_mask', initial_mask.clone())
        
        # Apply initial mask
        with torch.no_grad():
            self.layer.weight.mul_(self.layer.weight_mask)

        alpha = self.l1_regularization_coeff / num_params
        self.optimizer = LassoSGD(
            self.layer.parameters(),
            lr=self.init_lr,
            alpha=alpha,
        )
        self.layer, self.optimizer = self.fabric.setup(self.layer, self.optimizer)
        self.prune_amount_this_iteration = self._compute_current_prune_amount(
            prune_iteration, num_prune_iterations, self.prune_amount
        )


    def fit(self, dataloader: DataLoader, num_epochs: int = -1) -> int:
        tqdm.write(f"Prune amount this iteration: {self.prune_amount_this_iteration}")

        dataloader = self.fabric.setup_dataloaders(dataloader)

        best_loss = np.inf
        iter_no_change = 0

        if num_epochs > 0:
            self.max_iter = num_epochs
            self.num_iter_no_change = num_epochs

        if self.max_iter < 1:
            return

        tqdm.write(f"Setting learning rate to {lrrt.get_optimizer_lr(self.optimizer)}")

        conv_iter = self.max_iter
        
        # CHANGE: Variables to store average layer outputs for importance calculation
        layer_importance_accumulator = torch.zeros(self.layer.num_models, device=self.fabric.device)
        total_samples = 0

        self.layer.train()
        for iter in trange(
            self.max_iter, leave=False, desc="Prune weight fitting", dynamic_ncols=True
        ):
            loss_avg = AverageMeter(self.fabric)
            
            # Reset accumulator every epoch if we only want the last epoch's stats, 
            # but usually accumulating over the final epoch is best.
            # For simplicity, we will calculate importance in a separate pass or just use the weights magnitude.
            if iter == self.max_iter - 1:
                layer_importance_accumulator.zero_()
                total_samples = 0

            for X, Y in tqdm(dataloader, leave=False, dynamic_ncols=True):
                self.optimizer.zero_grad(set_to_none=True)
                
                # CHANGE: Layer-wise Normalization (Min-Max per layer)
                # Note: Doing this inside the loop is expensive. 
                # Ideally, this is done in the Dataset, but we do it here for access to slices.
                X_norm = torch.empty_like(X)
                for slc in self.layer.layer_slices:
                    layer_data = X[:, slc]
                    # Avoid division by zero with epsilon
                    min_val = layer_data.min(dim=1, keepdim=True)[0]
                    max_val = layer_data.max(dim=1, keepdim=True)[0]
                    numerator = layer_data - min_val
                    denominator = max_val - min_val + 1e-8
                    X_norm[:, slc] = numerator / denominator
                
                # Forward pass returns [batch, num_models]
                outputs = self.layer(X_norm) 
                
                # Accumulate importance (magnitude of contribution)
                if iter == self.max_iter - 1:
                    with torch.no_grad():
                        layer_importance_accumulator += outputs.abs().sum(dim=0)
                        total_samples += X.shape[0]

                # CHANGE: Sum layer outputs to get predicted total delta_L
                preds = outputs.sum(dim=1)
                
                Y = Y.view(preds.size())
                loss = F.mse_loss(preds, Y, reduction="mean")
                self.fabric.backward(loss)
                self.optimizer.step()
                
                # Enforce mask during training
                with torch.no_grad():
                    self.layer.weight.mul_(self.layer.weight_mask)
                    
                loss_avg.update(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_model_state = deepcopy(self.layer.state_dict())
                
        self.layer.load_state_dict(best_model_state)
        
        # CHANGE: Dynamic Pruning Schedule Calculation
        # 1. Calculate Average Importance per layer
        avg_importance = layer_importance_accumulator / total_samples
        # Handle case where importance is 0 to avoid div by zero
        avg_importance = avg_importance + 1e-6 
        
        # 2. Inverse Proportionality: Higher importance -> Lower prune %
        # Formula: Prune_fraction_i is proportional to 1 / importance_i
        inverse_imp = 1.0 / avg_importance
        total_inverse = inverse_imp.sum()
        
        # 3. Calculate how many weights to prune globally
        # current_mask is 1 for Keep, 0 for Prune
        current_active = self.layer.weight_mask.sum().item()
        
        # We need to prune 'prune_amount_this_iteration' percent of the *remaining* weights
        # Or is prune_amount_this_iteration the target sparsity? 
        # Looking at _compute_target_prune_amount, it seems to be cumulative sparsity.
        # Let's assume prune_amount_this_iteration is the Fraction to remove in THIS step relative to total parameters.
        # Logic from original code: `prune.l1_unstructured(..., amount=self.prune_amount_this_iteration)`
        # The `amount` in pytorch prune can be a float (fraction of total parameters).
        
        total_weights_to_prune = int(self.num_params * self.prune_amount_this_iteration)
        
        # 4. Distribute these "cuts" among layers
        # raw_cuts[i] = ( (1/imp_i) / sum(1/imp) ) * total_to_prune
        raw_cuts = (inverse_imp / total_inverse) * total_weights_to_prune
        cuts_per_layer = raw_cuts.long() # Floor to integer
        
        # 5. WATER-FILLING ALGORITHM (Fix for overflow)
        # Check if any layer is being asked to prune more than it has active weights
        # We need to know how many active weights are currently in each layer
        active_per_layer = torch.zeros(self.layer.num_models, device=self.fabric.device)
        for i, slc in enumerate(self.layer.layer_slices):
            active_per_layer[i] = self.layer.weight_mask[slc].sum()

        diff = cuts_per_layer - active_per_layer
        overflow = torch.clamp(diff, min=0).sum() # Sum of cuts that exceed capacity
        
        # Loop until no overflow
        while overflow > 0:
            print("Using overflow Algorithm")
            cuts_per_layer = torch.min(cuts_per_layer, active_per_layer.long())
            
            valid_mask = cuts_per_layer < active_per_layer.long()
            if valid_mask.sum() == 0: break

            valid_inverse = inverse_imp * valid_mask.float()
            valid_total_inverse = valid_inverse.sum()
            

            additional_cuts = (valid_inverse / valid_total_inverse) * overflow
            cuts_per_layer += additional_cuts.long()
            
            diff = cuts_per_layer - active_per_layer
            new_overflow = torch.clamp(diff, min=0).sum()
            
            if new_overflow >= overflow: 
                break 
            overflow = new_overflow

        # CHANGE: Apply pruning per layer
        final_mask = self.layer.weight_mask.clone()
        weights_cpu = self.layer.weight.detach().abs()
        
        for i, slc in enumerate(self.layer.layer_slices):
            n_prune = int(cuts_per_layer[i].item())
            if n_prune <= 0: continue
            
            layer_weights = weights_cpu[slc]
            layer_mask = final_mask[slc]
            
            # Find the smallest n_prune weights that are currently active (mask=1)
            # Set their mask to 0.
            # We filter for active weights only
            active_indices = torch.nonzero(layer_mask).squeeze()
            if active_indices.numel() == 0: continue
            
            if active_indices.numel() <= n_prune:
                # Prune everything
                layer_mask[:] = 0
            else:
                active_values = layer_weights[active_indices]
                # Find threshold
                kth_val = torch.kthvalue(active_values, n_prune).values
                # Prune weights <= threshold
                # Be careful not to un-prune previously pruned weights
                new_zeros = (layer_weights <= kth_val) & (layer_mask == 1)
                layer_mask[new_zeros] = 0
                
            final_mask[slc] = layer_mask

        self.layer.weight_mask = final_mask
        # Update weights to zero out pruned ones
        self.layer.weight.data.mul_(self.layer.weight_mask)
        
        return conv_iter

    # ... get_non_zero_weights logic remains roughly the same, 
    # but ensure it returns self.layer.weight_mask
    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        return torch.flatten(self.layer.weight_mask)

    def _compute_current_prune_amount(
        self,
        prune_iteration: int,
        num_prune_iterations: int,
        total_prune_amount: float,
    ) -> float:
        target_prune_amount_this_iteration = self._compute_target_prune_amount(
            prune_iteration, num_prune_iterations, total_prune_amount
        )
        N = 1.0 - target_prune_amount_this_iteration
        target_prune_amount_last_iteration = self._compute_target_prune_amount(
            prune_iteration - 1, num_prune_iterations, total_prune_amount
        )
        M = 1.0 - target_prune_amount_last_iteration
        return (M - N) / M

    def _compute_target_prune_amount(
        self, prune_iteration: int, num_prune_iterations: int, total_prune_amount: float
    ) -> float:
        return total_prune_amount * (
            1 - (1 - prune_iteration / num_prune_iterations) ** 3
        )

    def supports_batch_training(self) -> bool:
        return True



def get_causal_weights_trainer(
    config: CausalWeightsTrainerConfig, *args
) -> CausalWeightsTrainer:
    if config.backend == "sklearn":
        return CausalWeightsTrainerSklearn(config)
    elif config.backend == "torch":
        return CausalWeightsTrainerTorch(config, *args)
    raise NotImplementedError("Unsupported backed for CausalWeightsTrainer")