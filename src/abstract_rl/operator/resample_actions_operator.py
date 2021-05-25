import torch

from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator


class ResampleOperator(TrajectoryOperator):
    """
    Samples for each state additional actionsa and equips
    """

    def __repr__(self):
        return "resample"

    def __init__(self, mc):
        """
        Initializes a new retrace operator.
        :param mc: The model configuration with env and so on.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get('conf')
        self.conf = conf
        self.add_act = conf['add_acts']
        self.mc = mc
        self.policy = mc.get('policy', True)
        self.q_net = mc.get('q_network', True)

        env = mc['env']
        self.discount = env.discount()

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the operator.
        """

        # sample actions for sample based
        t_states = torch.Tensor(trajectory.states)
        f_stats = self.policy.forward(t_states)
        sampled_actions = self.policy.sample_actions(suff_stats=f_stats, num_actions=self.add_act)
        ll = self.policy.log_prob(sampled_actions, suff_stats=f_stats)
        q_vals = self.q_net.q_val(t_states.repeat([self.add_act, 1]), sampled_actions.view([-1, 1]))
        q_vals = q_vals.view(sampled_actions.size())
        act_rew = sampled_actions.detach().numpy()
        q_vals = q_vals.detach().numpy()
        ll = ll.detach().numpy()
        trajectory['add_act'] = act_rew
        trajectory['add_q'] = q_vals
        trajectory['add_pi'] = ll

