import pandas as pd
import torch
import numpy as np  # Required for possible float-int comparisons, return types
from typing import Union, Optional


class PINN_Dataset(torch.utils.data.Dataset):
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


class Interior_Partial_Dataset(PINN_Dataset):
    """
    Reads data (for data loss) from .csv files, intended for use
    with torch.utils.data.Dataloader.

    Excludes boundary points from read data if required.

    Can use only partial data from the .csv, reading only specified rows of
    data to reduce the size of the dataset.

    Returns data as (input array, output array), where inputs and outputs are
    wrt the PINN model.
    """
    def __init__(self,
                 filename: str,
                 input_cols: Union[list[str], list[int]],
                 output_cols: Union[list[str], list[int]],
                 exclude_bounds: Optional[list[tuple[float, float]]] = None,
                 sampling_idxs: Optional[list[int]] = None):
        """
        filename: .csv file containing data

        input_cols, output_cols: column names or column indices of input and
        output data in the .csv file

        exclude_bounds: List of tuples containing domain bounds to exclude from dataset,
        in order to sample only interior points

        sampling_idxs: indices of rows to use from the whole dataset

        input_cols, output_cols must be lists to guarantee numpy.ndarray is returned
        """
        data = super()._read_data(filename)
        # Exclude boundary points if bounds are specified
        if exclude_bounds is not None:
            data = self._exclude_boundaries(data, exclude_bounds)
        # Reduce dataset if specified
        if sampling_idxs is not None:
            data = data.iloc[sampling_idxs, :]
        # Split data into inputs and outputs
        self._inputs, self._outputs = super()._split_inputs_outputs(data, input_cols, output_cols)

    def _exclude_boundaries(self,
                            data: pd.DataFrame,
                            exclude_bounds: list[tuple[float, float]]) -> pd.DataFrame:
        for axis_idx, axis in enumerate(exclude_bounds):
            for bound in axis:
                data.drop(data[np.isclose(data.iloc[:, axis_idx], bound)].index,
                          inplace=True)
        return data
