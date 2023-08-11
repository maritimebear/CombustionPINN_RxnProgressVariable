import matplotlib.pyplot as plt
import plotters


class Plot_and_Log_Scalar():
    # Log scalar results per epoch or iteration to .csv file and plot them
    # Overwrites existing .csv file, each instance of this class creates a matplotlib figure
    # Intended for logging and plotting loss curves, so plots semilogy by default - change manually if needed
    # Does not call plt.show()
    def __init__(self,
                 filename: str,
                 scalars_dict: dict[str, list[float]],
                 plot_xlabel: str,
                 plot_ylabel: str,
                 plot_title: str,
                 figsize=(8, 8)) -> None:
        # filename: name of .csv
        # scalars_dict: map of labels of each scalar to list containing that scalar
        self.filename = "log_" + filename + ".csv"
        self.scalars_dict = scalars_dict
        assert self._get_len() == 0, "Check that all lists in scalars_dict are empty when initialising logger"
        # Plot labels and title
        self.xlabel = plot_xlabel
        self.ylabel = plot_ylabel
        self.title = plot_title
        self.figsize = figsize
        # Plotting function not passed as a constructor parameter, monkey-patch if needed
        self.plot_function = plotters.semilogy_plot

        self.idx = 0  # Tracks last-written index of data

        # Create .csv file
        header = ",".join(f"{label}" for label in scalars_dict.keys())  # First line of .csv, names of scalar series
        with open(self.filename, "w") as file:  # Overwrite existing file
            file.write(header + "\n")

        # Set up plotting
        # No way to update existing plots using matplotlib?
        # with plt.ioff():
            # self.figure = plt.figure(figsize=figsize)
            # self.axes = self.figure.add_subplot(1, 1, 1)

    def _get_len(self) -> int:
        # Get length of lists in self.scalar_dict
        n = None
        for _list in self.scalars_dict.values():
            if n is None:
                n = len(_list)
            else:
                assert len(_list) == n, "Lengths of lists to be logged are not equal"
        return n

    def update_log(self) -> None:
        with open(self.filename, "a") as file:  # Append to .csv created in __init__()
            for i in range(self.idx, self._get_len()):
                # Write scalars line-by-line, incrementing self.idx after each line
                line = [_list[self.idx] for _list in self.scalars_dict.values()]
                file.write(",".join(f"{value}" for value in line) + "\n")
                self.idx += 1

    def update_plot(self) -> None:
        # Creates a new figure upon each call
        with plt.ioff():
            self.figure = plt.figure(figsize=self.figsize)
            self.axes = self.figure.add_subplot(1, 1, 1)
            for _label, _list in self.scalars_dict.items():
                self.axes = self.plot_function(self.axes, _list, label=_label,
                                               ylabel=self.ylabel, xlabel=self.xlabel, title=self.title)
