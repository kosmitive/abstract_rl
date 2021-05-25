import math

import numpy as np

from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.src.policy.non_parametric.non_parametric_variational_policy_builder import NonParametricVariationalPolicyBuilder


class VariationalRankingOperator(TrajectoryOperator):
    """
    Represents a variational reweighting operator see https://arxiv.org/abs/1806.06920.
    """

    def __repr__(self):
        return "reweighing"

    def __init__(self, mc):
        """
        Initializes a new variational reweighting operator.
        :param mc: The model configuration to use.
        """
        self.conf = mc['conf']
        self.mc = mc
        self.policy = mc.get('policy', True)
        self.num_add_actions = self.conf.get_root('res_add_acts')
        self.b = NonParametricVariationalPolicyBuilder(self.mc)

    def hook(self):
        self.b.calc_eta()

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # call the super class
        q = self.b.build(trajectory)
        _, N = q.samples.shape
        max_q_samples = np.max(q.samples)

        q_norm = np.log(np.sum(np.exp(q.samples - max_q_samples), 1)) + max_q_samples * math.log(N)
        q.samples -= np.expand_dims(q_norm, 1)

        # add a annotation to the trajectory
        trajectory['grad_ranking'] = q.samples
