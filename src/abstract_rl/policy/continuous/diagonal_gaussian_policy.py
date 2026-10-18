"""Diagonal Gaussian policy for action spaces of any dimension."""

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.gaussian_math import head_activations, sigma_is_softplus

__all__ = ["DiagonalGaussianPolicy", "head_activations", "sigma_is_softplus"]


class DiagonalGaussianPolicy(ContinuousPolicy):
    """Gaussian policy with a diagonal covariance.

    Samples and extra actions are ``[batch, k, act]``. One action per state may also be passed as
    ``[batch, act]``. Log probabilities are ``[batch, k]``.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        super().__init__(mc, flat_actions=False)
