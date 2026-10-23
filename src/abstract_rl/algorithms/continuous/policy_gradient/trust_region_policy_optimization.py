from typing import Any

from jaxtyping import Array, Float

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState


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

        TrustRegionGradient.__init__(
            self, "policy", trpo_eps, trpo_cg_max_k, trpo_cg_residual_tol, trpo_dvp_damping, "max"
        )

    def tr_distance(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """Pass back KL divergence for training divergence.

        :param state: The traced state holding the (candidate) policy and its target.
        :param batch: The batch to use.
        :return: The KL divergence.
        """
        return state.get("policy").kl_divergence(state, batch, est="mean", mode="m-projection")

    def tr_obj(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """Pass back the likelihood ratio gradient.

        :param state: The traced state holding the (candidate) policy.
        :param batch: The batch to use.
        :return: The likelihood ratio gradient
        """
        return state.get("policy").likelihood_ratio(state, batch)
