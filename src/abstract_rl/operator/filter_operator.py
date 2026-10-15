"""Optional running-mean filter on rewards.

State normalisation is ``obs_norm``. It is applied when the policy acts and again, with the same
frozen statistics, when the policy and value networks read observations. Stored trajectories keep
the unnormalised observation, so this operator no longer rewrites states.
"""

import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class FilterOperator(TrajectoryOperator):
    """Applies the environment's running reward filter, once per trajectory."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.env = mc["env"]
        self.filter_rewards = conf["rewards"]

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """Filter every trajectory, then refresh the on-device copy when rewards changed.

        With the reward filter off the merge that just ran already uploaded the episodes. A second
        upload would compile a new scatter each time the stored length changes.
        """
        if isinstance(trajectories, TrajectoryCollection):
            for trajectory in trajectories.trajectories():
                self.transform(trajectory)
            if self.filter_rewards:
                trajectories._sync_device()
            return
        super().transform_all(trajectories)

    def transform(self, trajectory: Trajectory) -> None:
        if self.filter_rewards and "unfiltered_rewards" not in trajectory:
            trajectory["unfiltered_rewards"] = trajectory._rewards
            trajectory._rewards = np.asarray(self.env.reward_filter(trajectory["unfiltered_rewards"], False))
