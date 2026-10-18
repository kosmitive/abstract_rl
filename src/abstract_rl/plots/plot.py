import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class Plot:
    """This class can be used to give a class access to subplots managed
    by itself."""

    def __init__(self, title: str, grid_shape: tuple[int, int] | None) -> None:
        """Constructs a new LayoutManagedPlot. Creates subplots according to grid_shapes."""

        # common settings
        self.title = title
        if grid_shape is None:
            self.fig = plt.figure()
            self.axes = plt.gca()
        else:
            self.fig, self.axes = plt.subplots(*grid_shape)

        self.fig.suptitle(title)

        # build up grid

    def get_axes(self) -> Axes:
        """Get all axes"""
        return self.axes

    def show(self) -> None:
        """This method stops the interactive mode if activated
        and shows the plot."""
        plt.show()

    def save(self, filename: str) -> None:
        self.fig.savefig(filename)
