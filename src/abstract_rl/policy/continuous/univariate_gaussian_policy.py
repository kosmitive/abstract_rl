"""Univariate Gaussian policy. Action samples have shape ``[batch, k]``."""

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.gaussian_math import SIGMA_MIN, SQUASH_BOUND, SQUASH_U_MAX


class UnivariateGaussianPolicy(ContinuousPolicy):
    """Gaussian policy for a single action dimension."""

    def __init__(self, mc: ModelConfiguration) -> None:
        if mc.env.action_dim != 1:
            raise ValueError("UnivariateGaussianPolicy requires a one-dimensional action space")
        super().__init__(mc, flat_actions=True)

    def get_suff_stats_dim(self) -> int:
        """Mean and standard deviation."""
        return 2


# Re-exported for callers that historically imported the squash limits from this module.
__all__ = ["SIGMA_MIN", "SQUASH_BOUND", "SQUASH_U_MAX", "UnivariateGaussianPolicy"]
