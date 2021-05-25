import matplotlib.pyplot as plt


class Plot:

    """This class can be used to give a class access to subplots managed
    by itself."""

    def __init__(self, title, grid_shape):
        """Constructs a new LayoutManagedPlot. Creates subplots according to grid_shapes.
        """

        # common settings
        self.title = title
        if grid_shape is None:
            self.fig = plt.figure()
            self.axes = plt.gca()
        else:
            self.fig, self.axes = plt.subplots(*grid_shape)

        self.fig.suptitle(title)

        # build up grid

    def get_axes(self):
        """Get all axes"""
        return self.axes

    def show(self):
        """This method stops the interactive mode if activated
        and shows the plot."""
        plt.show()

    def save(self, filename):
        self.fig.savefig(filename)