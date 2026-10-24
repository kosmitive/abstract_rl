from typing import Any

import mlx.core as mx

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch


class TRPOAlgorithm(PolicyOptimizationAlgorithm, TrustRegionGradient):
    """
    Trust region policy optimization (https://arxiv.org/abs/1502.05477). On-policy: the advantages come from
    generalized advantage estimation on freshly collected trajectories, and the surrogate objective is maximized with
    a natural gradient step under a KL trust region.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        PolicyOptimizationAlgorithm.__init__(self, mc)
        conf = mc.get("conf")

        # trpo vars
        trpo_eps = conf["tr_eps"]
        trpo_cg_max_k = conf["cg_max_k"]
        trpo_cg_residual_tol = conf["cg_res_tol"]
        trpo_dvp_damping = conf["dvp_damping"]
        self.policy = self.mc.get("policy", False)

        TrustRegionGradient.__init__(
            self, self.policy, trpo_eps, trpo_cg_max_k, trpo_cg_residual_tol, trpo_dvp_damping, "max"
        )

    def tr_inputs(self) -> list[Any]:
        """The KL is measured against the target policy, its parameters are an input of the compiled step."""
        return [self.mc.get("policy", True).state]

    def tr_distance(self, batch: Batch) -> mx.array:
        """Pass back KL divergence for training divergence.

        :param batch: The batch to use.
        :return: The KL divergence.
        """
        return self.policy.kl_divergence(batch, est="mean", mode="m-projection")

    def tr_obj(self, batch: Batch) -> mx.array:
        """Pass back the likelihood ratio gradient.

        :param batch: The batch to use.
        :return: The likelihood ratio gradient
        """
        return self.policy.likelihood_ratio(batch)
