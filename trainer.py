import torch
from dataclasses import dataclass, field

import training

from typing import Callable, Sequence
# CIP pool runs Python 3.9, TypeAlias in typing for >= 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

Tensor: TypeAlias = torch.Tensor
DataLoader: TypeAlias = torch.utils.data.DataLoader


@dataclass(slots=True, eq=False)
class Trainer():
    dataloaders: list[DataLoader]
    model: torch.nn.Module
    optimiser: torch.optim.Optimizer
    residualfn: Callable[[Tensor, Tensor], Tensor]
    lossfn_data: Callable[[Tensor, Tensor], Tensor]
    lossfn_residual: Callable[[Tensor, Tensor], Tensor]

    # Optional attributes
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = field(default=None)
    grad_norm_limit: float = field(default=None)

    # "private" attributes
    zeros_: Tensor = field(default=None, init=False)

    def train_epoch_(self, data_dl: DataLoader, residual_dl: DataLoader) -> dict[str, float]:
        # Returns mean losses
        mean_losses = {key: [] for key in ("data", "residual", "total")}
        for batch_data, batch_residual in training.cycle_shorter_iterators([data_dl, residual_dl]):

            x_data, y_data = batch_data
            x_residual, y_residual = batch_residual
            # Loading y_residual from batch instead of torch.zeros for uniformity

            residual = self.residualfn(self.model(x_residual), x_residual)
            if self.zeros_ is None:
                self.zeros_ = torch.zeros_like(residual)

            self.optimiser.zero_grad()

            mean_losses["data"].append(loss_data := self.lossfn_data(self.model(x_data), y_data))
            mean_losses["residual"].append(loss_residual := self.lossfn_residual(residual, self.zeros_))
            mean_losses["total"].append(loss_total := loss_data + loss_residual)

            loss_total.backward()

            if self.grad_norm_limit is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_norm_limit)

            self.optimiser.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Calculate mean of all saved losses and return as a dict
        return {key: (sum(_list) / len(_list)).detach().item() for
                key, _list in mean_losses.items()}


@dataclass(slots=True, eq=False)
class Trainer_lists():
    dataloaders: list[DataLoader]
    model: torch.nn.Module
    optimiser: torch.optim.Optimizer
    callables: list[Callable[[Tensor, Tensor, Tensor], Tensor]]
    # Optional attributes
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = field(default=None)
    grad_norm_limit: float = field(default=None)

    def __post_init__(self) -> None:
        assert len(self.dataloaders) == len(self.callables), \
            "Each DataLoader must have a corresponding callable or loss function"

    def train_epoch(self) -> list[float]:
        losses_mean = [float() for i in range(len(self.callables) + 1)]

        for superbatch in training.cycle_shorter_iterators(self.dataloaders):
            # superbatch: [(x, y), (x, y), ...], each (x, y) corresponds to a loss_fn
            losses = [f(x, self.model(x), y) for
                      f, (x, y) in zip(self.callables, superbatch)]
            losses.append(loss_total := sum(losses))

            self.optimiser.zero_grad()

            loss_total.backward()

            if self.grad_norm_limit is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_norm_limit)

            self.optimiser.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            losses_mean = [losses_mean[i] + losses[i].detach().item() for
                           i in range(len(losses))]

        return [value / len(losses_mean) for value in losses_mean]
