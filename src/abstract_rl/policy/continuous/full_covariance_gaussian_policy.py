"""State-dependent full-covariance Gaussian policy."""

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy

__all__ = ["FullCovarianceGaussianPolicy"]


class FullCovarianceGaussianPolicy(ContinuousPolicy):
    """Gaussian policy with a state-dependent covariance ``L(s) L(s)^T``.

    The network outputs ``μ(s)`` and the lower-triangular Cholesky factor ``L(s)``, packed in
    row-major order. The diagonal of ``L`` is ``softplus(raw) + policy_min_std``. Off-diagonal
    entries stay linear. Samples are ``u = μ + L z`` with ``z ~ N(0, I)``. Log probabilities
    and the KL are evaluated with triangular solves.

    Squashing is the same componentwise map as the diagonal policy,
    ``a = center + scale * tanh(u)``. The log density subtracts
    ``sum log(scale (1 - tanh(u)^2))``. The KL stays the KL of the Gaussian on ``u``, because
    that map is a bijection shared by the two policies. Both the squashed and the unsquashed
    policy are valid.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        super().__init__(mc, flat_actions=False)
        if self.cov_type != "full":
            raise ValueError("FullCovarianceGaussianPolicy requires cov_type 'full'")

    def get_suff_stats_dim(self) -> int:
        """Mean plus the packed lower-triangular Cholesky factor."""
        action_dim = self.mc.env.action_dim
        return action_dim + action_dim * (action_dim + 1) // 2
