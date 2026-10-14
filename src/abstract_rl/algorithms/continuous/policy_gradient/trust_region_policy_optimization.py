"""Trust region policy optimisation with GAE advantages."""

from jaxtyping import Array, Float

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch


class TRPOAlgorithm(PolicyOptimizationAlgorithm, TrustRegionGradient):
    """Trust region policy optimisation.

    Advantages come from GAE. The value function is fit separately. See https://arxiv.org/abs/1502.05477.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        PolicyOptimizationAlgorithm.__init__(self, mc)
        conf = mc.get("conf")
        self.policy = self.mc.get("policy", False)
        TrustRegionGradient.__init__(
            self,
            self.policy,
            float(conf["tr_eps"]),
            int(conf["cg_max_k"]),
            float(conf["cg_res_tol"]),
            float(conf["dvp_damping"]),
            "max",
        )

    def tr_distance(self, batch: Batch) -> Float[Array, ""]:
        """Mean KL of the target policy against the current one (m-projection).

        Args:
            batch: States the divergence is averaged over.

        Returns:
            Scalar mean KL.
        """
        return self.policy.kl_divergence(batch, est="mean", mode="m-projection")

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Negative standardised likelihood-ratio surrogate.

        Args:
            batch: On-policy batch with advantages in ``a``.

        Returns:
            Scalar surrogate. Lower is a better step.
        """
        return self.policy.likelihood_ratio(batch)
