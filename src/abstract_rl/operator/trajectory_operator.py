from typing import Any

import jax

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


class TrajectoryOperator:
    """
    A simple evaluation operator interface.
    """

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory (one lane) with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectory: trajectory to transform.
        :param key: Random key of stochastic operators.
        :return: The transformed trajectory.
        """

        raise NotImplementedError

    def transform_all(
        self, state: ModelState, trajectories: TrajectoryCollection, key: Any = None
    ) -> TrajectoryCollection:
        """
        Transform all lanes of a collection with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectories: trajectories to transform.
        :param key: Random key of stochastic operators, split into one key per lane.
        """

        lanes = trajectories.trajectories()
        if key is None:
            out = jax.vmap(lambda trajectory: self.transform(state, trajectory))(lanes)
        else:
            keys = jax.random.split(key, len(trajectories))
            out = jax.vmap(lambda trajectory, k: self.transform(state, trajectory, k))(lanes, keys)
        return trajectories.with_trajectories(out)
