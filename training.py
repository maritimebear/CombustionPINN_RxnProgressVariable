import numpy as np
import pandas as pd
import torch

from typing import Union, Callable, TypeAlias
Tensor: TypeAlias = torch.tensor


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
                 loss_fn: Callable[[Tensor], float],
                 weight: float = 1.0):
        self.loss_fn = loss_fn  # Function/callable class pointer
        self.weight = weight

    def __call__(self,
                 prediction: Tensor,
                 target: Tensor) -> float:
        return (self.weight * self.loss_fn(prediction, target))
# end class WeightedScalarLoss
