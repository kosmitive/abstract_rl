import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.policy.non_parametric.non_parametric_variational_policy_builder import (
    NonParametricVariationalPolicyBuilder,
)


class VariationalReweightingOperator(TrajectoryOperator):
    """
    Represents a variational reweighting operator see https://arxiv.org/abs/1806.06920.
    """

    def __repr__(self) -> str:
        return "reweighing"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new variational reweighting operator.
        :param mc: The model configuration to use.
        """
        self.conf = mc["conf"]
        self.mc = mc
        self.policy = mc.get("policy", True)
        self.num_add_actions = self.conf.get_root("res_add_acts")
        self.b = NonParametricVariationalPolicyBuilder(self.mc)

    def hook(self) -> None:
        self.b.calc_eta()

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # call the super class
        q = self.b.build(trajectory)
        # normalize per state, e.g. log q(a|s) = Q(s,a) / eta - log sum_a' exp(Q(s,a') / eta)
        max_q_samples = np.max(q.samples, axis=1, keepdims=True)
        q_norm = np.log(np.sum(np.exp(q.samples - max_q_samples), axis=1, keepdims=True)) + max_q_samples
        q.samples = q.samples - q_norm

        # add a annotation to the trajectory
        trajectory["grad_ranking"] = q.samples
