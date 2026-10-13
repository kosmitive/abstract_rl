"""Optional running-mean filters on states and rewards."""

import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class FilterOperator(TrajectoryOperator):
    """Applies the environment's running filters, once per trajectory."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.env = mc["env"]
        self.filter_states = conf["states"]
        self.filter_rewards = conf["rewards"]

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """Filter every trajectory, then refresh the on-device copy when one is attached."""
        if isinstance(trajectories, TrajectoryCollection):
            for trajectory in trajectories.trajectories():
                self.transform(trajectory)
            trajectories._sync_device()
            return
        super().transform_all(trajectories)

    def transform(self, trajectory: Trajectory) -> None:
        if self.filter_states and "unfiltered_states" not in trajectory:
            trajectory["unfiltered_states"] = trajectory._states
            trajectory._states = np.asarray(self.env.state_filter(trajectory["unfiltered_states"], False))
        if self.filter_rewards and "unfiltered_rewards" not in trajectory:
            trajectory["unfiltered_rewards"] = trajectory._rewards
            trajectory._rewards = np.asarray(self.env.reward_filter(trajectory["unfiltered_rewards"], False))
