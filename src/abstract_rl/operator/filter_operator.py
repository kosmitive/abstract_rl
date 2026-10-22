from typing import Any

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class FilterOperator(TrajectoryOperator):
    """
    Represents a simple filter operator using the integrated filter from the environment.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new filter operator
        :param mc: The model configuration to use.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.env = mc["env"]

        # get discount
        self.filter_states = conf["states"]
        self.filter_rewards = conf["rewards"]

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state (gives the running filters).
        :param trajectory: trajectory to transform.
        """
        if self.filter_states and "unfiltered_states" not in trajectory:
            state_filter = state.get("state_filter")
            trajectory = trajectory.set("unfiltered_states", trajectory.states)
            trajectory = trajectory.replace(
                states=state_filter(trajectory.states), next_states=state_filter(trajectory.next_states)
            )

        if self.filter_rewards and "unfiltered_rewards" not in trajectory:
            reward_filter = state.get("reward_filter")
            trajectory = trajectory.set("unfiltered_rewards", trajectory.rewards)
            trajectory = trajectory.replace(rewards=reward_filter(trajectory.rewards))
        return trajectory
