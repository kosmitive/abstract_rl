import numpy as np

from abstract_rl.src.plots.plot import Plot


class TrajectoryPlot(Plot):
    """A plot capable of displaying a graph along with the value function
    and the actions used in each state."""

    def __init__(self, num_trajectories=3):
        super().__init__('Best Trajectories', [num_trajectories, 1])
        self.num_trajectories = num_trajectories

    def update(self, trajectory_list):

        assert len(trajectory_list) == self.num_trajectories
        for i in range(self.num_trajectories):
            traj = trajectory_list[i]
            ar_timesteps = np.arange(len(traj))
            axes = super().get_axes()[i] if self.num_trajectories > 1 else super().get_axes()
            axes.cla()
            axes.plot(ar_timesteps, traj.actions, color="#215ab7")

            # labels
            axes.set_xlabel("t")
            axes.set_ylabel("a")

    def save(self, filename):
        self.fig.savefig(filename)
