from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.mixed import to_mx
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
        self.policy = mc.get("policy", True)
        self.q_net = mc.get("q_network", True)

        env = mc["env"]
        self.discount = env.discount()

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the operator. The annotations stay lazy MLX arrays until
        transform_all evaluates them.
        """

        # sample actions for sample based
        t_states = to_mx(trajectory.states)
        f_stats = self.policy.forward(t_states)
        sampled_actions = self.policy.sample_actions(suff_stats=f_stats, num_actions=self.add_act)
        ll = self.policy.log_prob(sampled_actions, suff_stats=f_stats)
        q_vals = self.q_net.q_val_multi(t_states, sampled_actions)
        trajectory["add_act"] = sampled_actions
        trajectory["add_q"] = q_vals
        trajectory["add_pi"] = ll
