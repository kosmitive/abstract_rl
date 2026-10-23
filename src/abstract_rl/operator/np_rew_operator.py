from typing import Any

import jax.numpy as jnp

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.policy.non_parametric.non_parametric_variational_policy_builder import (
    NonParametricVariationalPolicyBuilder,
)


class VariationalRankingOperator(TrajectoryOperator):
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
        self.num_add_actions = self.conf.get_root("res_add_acts")
        self.b = NonParametricVariationalPolicyBuilder(self.mc)

    def hook(self, state: ModelState, tc: TrajectoryCollection) -> ModelState:
        return self.b.calc_eta(state, tc)

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # call the super class
        q = self.b.build(state, trajectory)
        # normalize per state, e.g. log q(a|s) = Q(s,a) / eta - log sum_a' exp(Q(s,a') / eta)
        max_q_samples = jnp.max(q.samples, axis=1, keepdims=True)
        q_norm = jnp.log(jnp.sum(jnp.exp(q.samples - max_q_samples), axis=1, keepdims=True)) + max_q_samples
        q.samples = q.samples - q_norm

        # add a annotation to the trajectory
        return trajectory.set("grad_ranking", q.samples)
