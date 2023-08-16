from dataclasses import dataclass, field


@dataclass
class DictLogger():
    # Class to log data to .csv files
    # Log updated from dict[str, list[float]] holding data
    # Intended to log loss and residual history

    data: dict[str, list[float]]
    filename: str  # path/to/logfile.csv
    index_column: str  # Label of index column, eg. Iteration, or Epoch
    idx: int = field(default=0, init=False)  # Tracks last-written index of data

    def __post_init__(self) -> None:
        # Create .csv file after __init__()
        assert self._get_len() == 0, "Check that all lists in the data dictionary are empty when initialising logger"
        # First line of .csv, names of columns
        header = ",".join(f"{label}" for label in [self.index_column, *self.data.keys()])
        with open(self.filename, "w") as file:  # Overwrite existing file
            file.write(f"{header}\n")

    def _get_len(self) -> int:
        # Get length of lists in self.data,
        # checks that all lists to be logged are of the same length
        n = None
        for _list in self.data.values():
            if n is None:
                n = len(_list)
            else:
                assert len(_list) == n, "Lengths of lists to be logged are not equal"
        return n

    def update(self) -> None:
        # Update .csv file with new data
        with open(self.filename, "a") as file:
            for i in range(self.idx, self._get_len()):
                # Write new data line-by-line, incrementing self.idx after each line
                line = [self.idx, *[_list[self.idx] for _list in self.data.values()]]
                file.write(",".join(f"{value}" for value in line) + "\n")
                self.idx += 1
