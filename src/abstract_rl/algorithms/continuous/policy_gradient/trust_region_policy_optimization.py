"""Trust region policy optimisation with GAE advantages."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.obs_norm import normalize_obs
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, kl_divergence, likelihood_ratio


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
        states = normalize_obs(self.policy.observation_norm(), jnp.asarray(batch.states, dtype=jnp.float32))
        target = self.policy.mc.get("policy", True).params
        return self.tr_divergence(self.policy.params, target, states)

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Negative standardised likelihood-ratio surrogate.

        Args:
            batch: On-policy batch with advantages in ``a``.

        Returns:
            Scalar surrogate. Lower is a better step.
        """
        states = normalize_obs(self.policy.observation_norm(), jnp.asarray(batch.states, dtype=jnp.float32))
        return self.tr_objective(
            self.policy.params,
            states,
            jnp.asarray(batch.actions, dtype=jnp.float32),
            jnp.asarray(batch.log_likelihoods, dtype=jnp.float32),
            jnp.asarray(batch["a"], dtype=jnp.float32),
        )

    @staticmethod
    def tr_objective(
        params: GaussianParams,
        states: Float[Array, "batch obs"],
        actions: Float[Array, "batch act"],
        old_log_prob: Float[Array, " batch"],
        advantages: Float[Array, " batch"],
    ) -> Float[Array, ""]:
        """Surrogate used by the natural step and by ``tr_obj``."""
        return likelihood_ratio(params, states, actions, old_log_prob, advantages)

    @staticmethod
    def tr_divergence(
        params: GaussianParams,
        target: GaussianParams,
        states: Float[Array, "batch obs"],
    ) -> Float[Array, ""]:
        """Mean m-projection KL used by the natural step and by ``tr_distance``."""
        return kl_divergence(params, target, states, mode_name="m-projection", est="mean")
