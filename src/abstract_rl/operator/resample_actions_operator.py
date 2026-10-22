from typing import Any

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class ResampleOperator(TrajectoryOperator):
    """
    Samples for each state additional actionsa and equips
    """

    def __repr__(self) -> str:
        return "resample"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new retrace operator.
        :param mc: The model configuration with env and so on.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.add_act = conf["add_acts"]
        self.mc = mc

        env = mc["env"]
        self.discount = env.discount()

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the operator.
        """
        policy = state.get("policy", True)
        q_net = state.get("q_network", True)

        # sample actions for sample based
        t_states = trajectory.states
        f_stats = policy.forward(t_states)
        sampled_actions = policy.sample_actions(key, suff_stats=f_stats, num_actions=self.add_act)
        ll = policy.log_prob(sampled_actions, suff_stats=f_stats)
        q_vals = q_net.q_val_multi(t_states, sampled_actions)
        return trajectory.set("add_act", sampled_actions).set("add_q", q_vals).set("add_pi", ll)
