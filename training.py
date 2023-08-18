import numpy as np
import pandas as pd
import torch

from typing import Union, Callable, Sequence, TypeVar
from collections.abc import Iterator, Generator
# CIP pool runs Python 3.9, TypeAlias in typing for >= 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

Tensor: TypeAlias = torch.Tensor
T = TypeVar("T")


def train_step_data(x, y, model, loss_fn) -> Tensor:
    return loss_fn(model(x), y)


def train_step_residual(x, model, residual_fn, loss_fn) -> Tensor:
    return loss_fn(residual_fn(model(x), x), torch.zeros_like(x))


def chain_callables(base_arg: Tensor,
                    model: Callable[[Tensor], Tensor],
                    second_args: Sequence[Tensor],
                    callables: Sequence[Callable[[Tensor, Tensor], Tensor]]) -> list[Tensor]:

    results = [model(base_arg)]
    for i in range(len(callables)):
        results.append(callables[i](results[-1], second_args[i]))
    return results


def cycle_shorter_iterators(iterator_list: list[Iterator[T]]) -> Generator[list[T], None, None]:
    # Combine multiple iterators of different lengths
    # Returned iterator lasts until the longest iterator in the input list lasts
    # All other (i.e. shorter) iterators in the input will be cycled
    # Intended to combine multiple torch Dataloaders of different lengths
    # Inspired by itertools.zip_longest()
    assert (n_active := len(iterator_list)) > 0, "Input list is empty?"
    iterators = [iter(it) for it in iterator_list]
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                n_active -= 1
                if not n_active:
                    return
                iterators[i] = iter(iterator_list[i])  # Cycle expired iterator
                value = next(iterators[i])
            values.append(value)
        yield values


class SampledDataset():
    """
    Create dataset from (x, y) data
    For use with torch dataloader
    """
    def __init__(self, x: Tensor, y: Tensor) -> None:
        assert x.size() == y.size()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (self.x[idx], self.y[idx])


class PINN_Dataset():
    """
    Reads data (for data loss) from .csv files, intended for use
    with torch.utils.data.Dataloader.

    Returns data as (input array, output array), where inputs and outputs are
    wrt the PINN model.
    """
    def __init__(self,
                 filename: str,
                 input_cols: Union[list[str], list[int]],
                 output_cols: Union[list[str], list[int]]):
        """
        filename: .csv file containing data
        input_cols, output_cols: column names or column indices of input and
        output data in the .csv file

        input_cols, output_cols must be lists to guarantee numpy.ndarray is returned
        """
        data = self._read_data(filename)
        # Split data into inputs and outputs
        self._inputs, self._outputs = self._split_inputs_outputs(data, input_cols, output_cols)

    def _read_data(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename)

    def _split_inputs_outputs(self,
                              data: pd.DataFrame,
                              input_cols: Union[list[str], list[int]],
                              output_cols: Union[list[str], list[int]]
                              ) -> tuple[np.ndarray, np.ndarray]:

        # try-catch block to access columns in .csv by either names or indices
        try:
            # data.loc inputs string labels (column names)
            inputs, outputs = [data.loc[:, labels].to_numpy() for
                               labels in (input_cols, output_cols)]
        except KeyError:
            # data.iloc expects int indices
            inputs, outputs = [data.iloc[:, labels].to_numpy() for
                               labels in (input_cols, output_cols)]

        assert len(inputs) == len(outputs)
        return (inputs, outputs)

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return (self._inputs[idx], self._outputs[idx])
# end class PINN_Dataset


class WeightedScalarLoss():
    """
    Callable class to calculate a generic scalar loss, multiplied by a weight.
    """
    def __init__(self,
                 loss_fn: Callable[[Tensor, Tensor], float],
                 weight: float = 1.0):
        self.loss_fn = loss_fn  # Function/callable class pointer
        self.weight = weight

    def __call__(self,
                 prediction: Tensor,
                 target: Tensor) -> float:
        return (self.weight * self.loss_fn(prediction, target))
# end class WeightedScalarLoss


class UniformRandomSampler():
    """
    Callable class, returns tensors with uniform-random entries, in the shape (n_points, n_dims).
    Each column corresponds to a space or time dimension (x, y, z, t) in the governing equations.

    n_dims inferred from number of 'extents' arguments passed to constructor.

    Each sequence in 'extents' defines the sampling interval in the corresponding coordinate,
    i.e. the corresponding column of the returned tensor.
    """
    def __init__(self,
                 n_points: int,
                 extents: Sequence[Sequence[float]],
                 *,
                 requires_grad=True):
        """
        'extents' must be a sequence of sequences,
            eg. [(0.0, 1.0), (0.0, 0.0)] for e1: [0,1], e2: [0,0]
        """
        self.n_points = n_points
        self.extents = extents
        self.requires_grad = requires_grad
        # Lambdas used to generate tensors
        self.generate = lambda _range: (torch.Tensor(n_points, 1).uniform_(*_range).requires_grad_(requires_grad))

    def __call__(self) -> Tensor:
        return torch.hstack([self.generate(coord) for coord in self.extents])
