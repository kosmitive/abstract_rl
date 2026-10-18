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
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectories: trajectories to transform.
        """

        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()

        for trajectory in trajectories:
            self.transform(trajectory)
