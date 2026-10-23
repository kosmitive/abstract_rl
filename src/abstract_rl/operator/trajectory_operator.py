import mlx.core as mx
import numpy as np

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


class TrajectoryOperator:
    """
    A simple evaluation operator interface.
    """

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        raise NotImplementedError

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator. The annotations, which are
        computed lazily as MLX arrays, are evaluated at once and stored as numpy arrays.
        :param trajectories: trajectories to transform.
        """

        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()

        for trajectory in trajectories:
            self.transform(trajectory)
        materialize(trajectories)


def materialize(trajectories: list[Trajectory]) -> None:
    """Evaluate all MLX annotations of the trajectories in one go and replace them by numpy arrays."""
    lazy = [(t, k, v) for t in trajectories for k, v in t._annotations.items() if isinstance(v, mx.array)]
    mx.eval([v for _, _, v in lazy])
    for t, k, v in lazy:
        t[k] = np.array(v)
