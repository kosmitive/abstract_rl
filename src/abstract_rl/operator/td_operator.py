import numpy as np
import torch
from torch.distributions import transforms

from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator


class TemporalDifferenceOperator(TrajectoryOperator):
    """
    Represents a generalized advantage estimation operator based upon https://arxiv.org/abs/1506.02438.
    """

    def __repr__(self):
        return "td"

    def __init__(self, mc):
        """
        Initializes a new generalized advantage estimator.
        :param mc: The model configuration to use.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get('conf')
        self.conf = conf
        self.v_net = mc.get('v_network', True)

        # get discount
        self.env = mc['env']

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # deactivate grad
        with torch.no_grad():

            # obtain rewards
            discount = self.env.discount()
            t_states = torch.Tensor(trajectory.states)
            t_rewards = torch.Tensor(trajectory.rewards)
            dones = torch.Tensor(trajectory.dones)
            v = self.v_net.forward(t_states)
            mask = 1 - dones

            tl = len(trajectory)
            tv = torch.zeros(tl+1)
            td_errors = torch.zeros_like(dones)
            prev_value = 0

            for t in reversed(range(tl)):
                tv[t] = t_rewards[t] + discount * mask[t] * tv[t+1]
                td_errors[t] = t_rewards[t] + discount * prev_value * mask[t] - v[t]
                prev_value = v.data[t, 0]

            trajectory['tv'] = tv[:-1]
            trajectory['td'] = td_errors
