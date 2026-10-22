import jax

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


class TrajectoryOperator:
    """
    A simple evaluation operator interface.
    """

    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory (one lane) with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectory: trajectory to transform.
        :return: The transformed trajectory.
        """

        raise NotImplementedError

    def transform_all(self, state: ModelState, trajectories: TrajectoryCollection) -> TrajectoryCollection:
        """
        Transform all lanes of a collection with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectories: trajectories to transform.
        """

        return trajectories.with_trajectories(
            jax.vmap(lambda trajectory: self.transform(state, trajectory))(trajectories.trajectories())
        )
