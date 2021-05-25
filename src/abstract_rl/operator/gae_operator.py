import numpy as np
import torch
from torch.distributions import transforms

from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator


class GeneralizedAdvantagesEstimationOperator(TrajectoryOperator):
    """
    Represents a generalized advantage estimation operator based upon https://arxiv.org/abs/1506.02438.
    """

    def __repr__(self):
        return "gae"

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
        self.gae_lambda = torch.Tensor([self.conf['lambda']])

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # deactivate grad
        with torch.no_grad():

            tl = len(trajectory)

            # obtain rewards
            discount = self.env.discount()

            # use v approximation to get advantages
            td_errors = torch.Tensor(trajectory['td'])
            dones = torch.Tensor(trajectory.dones)
            mask = 1 - dones

            # reserve space for advantages
            a = torch.zeros(tl + 1)

            # get combined ratio
            gae_gamma_lambda = discount * self.gae_lambda

            for t in reversed(range(tl)):
                a[t] = td_errors[t] + mask[t] * gae_gamma_lambda * a[t + 1]

            trajectory['a'] = a[:-1]
