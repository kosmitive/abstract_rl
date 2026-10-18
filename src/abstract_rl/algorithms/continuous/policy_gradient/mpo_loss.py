"""E-step weights and the decoupled M-step KL of maximum a posteriori policy optimization.

The temperature dual and the non-parametric weights follow Abdolmaleki et al., 2018. The mean and covariance
constraints are the decoupled Gaussians used by Acme: online mean with the target covariance, and target mean with
the online covariance. Both KLs are ``KL(target || mixed)``, i.e. ``KL(pi_old || pi)``.
"""

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

ETA_FLOOR = 1e-8
ALPHA_FLOOR = 0.0


def project_positive(raw: torch.Tensor, floor: float) -> torch.Tensor:
    """Map an unconstrained parameter through a softplus and keep it at or above ``floor``.

    Args:
        raw: Unconstrained dual parameter.
        floor: Smallest allowed positive value. Zero leaves the softplus unchanged.

    Returns:
        The projected multiplier, with the same shape as ``raw``.
    """

    return torch.clamp(F.softplus(raw), min=floor)


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus`` for a strictly positive tensor.

    Args:
        value: Positive tensor the softplus should reproduce.

    Returns:
        The unconstrained parameter ``log(exp(value) - 1)``.
    """

    return torch.log(torch.expm1(value.clamp(min=1e-4)))


def normalized_weights(q_values: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Per-state softmax of ``Q / eta`` via log-sum-exp. The result is detached.

    Args:
        q_values: Critic values of shape ``[batch, num_actions]``.
        eta: Positive temperature.

    Returns:
        Weights of shape ``[batch, num_actions]`` that sum to one over the action axis.
    """

    tempered = q_values.detach() / eta
    log_weights = tempered - torch.logsumexp(tempered, dim=-1, keepdim=True)
    return log_weights.exp().detach()


def dual_shapes(action_dim: int, cov_type: str, per_dim: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shapes of the mean and covariance multipliers.

    Args:
        action_dim: Dimension of the action.
        cov_type: ``single``, ``fixed``, ``state`` or ``full``.
        per_dim: Whether each diagonal dimension has its own multiplier.

    Returns:
        ``(mean_shape, covariance_shape)``.
    """

    if not per_dim:
        return (1,), (1,)
    match cov_type:
        case "full":
            # a full covariance does not split per coordinate; the mean quadratic form does
            return (action_dim,), (1,)
        case _:
            return (action_dim,), (action_dim,)


def _reduce_kl(kl: torch.Tensor, per_dim: bool) -> torch.Tensor:
    """Average a ``[batch, ...]`` KL over the batch. ``per_dim`` keeps the action axis."""

    if per_dim:
        return kl.mean(dim=0)
    flat = kl.reshape(kl.shape[0], -1).sum(dim=-1)
    return flat.mean()


def decoupled_kl_diagonal(
    online: torch.Tensor, target: torch.Tensor, per_dim: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoupled diagonal-Gaussian KL, averaged over the batch.

    The statistic rows are ``[mean, std]``. The mean term uses the online mean with the target standard deviation,
    and the covariance term uses the target mean with the online standard deviation.

    Args:
        online: Online sufficient statistics of shape ``[batch, 2 * action_dim]``.
        target: Target sufficient statistics, same shape, detached from the target network.
        per_dim: Keep one KL per action dimension instead of summing them.

    Returns:
        Mean KL and covariance KL. Each is a scalar, or a vector of length ``action_dim`` when ``per_dim`` is set.
    """

    action_dim = online.shape[-1] // 2
    mu_online, sigma_online = online[:, :action_dim], online[:, action_dim:]
    mu_target, sigma_target = target[:, :action_dim], target[:, action_dim:]
    # KL(pi_old || pi): the first argument of kl_divergence is the target policy
    kl_mean = kl_divergence(Normal(mu_target, sigma_target), Normal(mu_online, sigma_target))
    kl_std = kl_divergence(Normal(mu_target, sigma_target), Normal(mu_target, sigma_online))
    return _reduce_kl(kl_mean, per_dim), _reduce_kl(kl_std, per_dim)


def _mean_kl_per_dimension(mu_target: torch.Tensor, mu_online: torch.Tensor, scale_tril: torch.Tensor) -> torch.Tensor:
    """Per-coordinate pieces of ``0.5 (mu_t - mu) @ inv(cov) @ (mu_t - mu)`` with ``cov = L @ L.T``.

    Args:
        mu_target: Target mean of shape ``[batch, action_dim]``.
        mu_online: Online mean, same shape.
        scale_tril: Lower-triangular factor of the covariance shared by both arguments, shape ``[batch, A, A]``.

    Returns:
        A tensor of shape ``[batch, action_dim]`` that sums to the mean KL over the action axis.
    """

    diff = mu_target - mu_online
    solved = torch.linalg.solve_triangular(scale_tril, diff.unsqueeze(-1), upper=False)
    precision_diff = torch.linalg.solve_triangular(scale_tril.transpose(-1, -2), solved, upper=True).squeeze(-1)
    return 0.5 * diff * precision_diff


def decoupled_kl_full(
    mu_online: torch.Tensor,
    factor_online: torch.Tensor,
    mu_target: torch.Tensor,
    factor_target: torch.Tensor,
    per_dim: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoupled full-covariance Gaussian KL, averaged over the batch.

    Args:
        mu_online: Online mean, shape ``[batch, action_dim]``.
        factor_online: Online Cholesky factor, shape ``[batch, action_dim, action_dim]``.
        mu_target: Target mean.
        factor_target: Target Cholesky factor.
        per_dim: Split the mean KL into per-coordinate contributions. The covariance KL stays a scalar.

    Returns:
        Mean KL and covariance KL.
    """

    target = MultivariateNormal(mu_target, scale_tril=factor_target)
    mixed_mean = MultivariateNormal(mu_online, scale_tril=factor_target)
    mixed_cov = MultivariateNormal(mu_target, scale_tril=factor_online)
    if per_dim:
        kl_mean = _mean_kl_per_dimension(mu_target, mu_online, factor_target)
        kl_cov = kl_divergence(target, mixed_cov)
        return _reduce_kl(kl_mean, True), _reduce_kl(kl_cov, False)
    return _reduce_kl(kl_divergence(target, mixed_mean), False), _reduce_kl(kl_divergence(target, mixed_cov), False)


def weighted_log_likelihood(log_prob: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted maximum likelihood ``E_s[sum_j w_j log pi(a_j | s)]``.

    Args:
        log_prob: Log density of the sampled actions, shape ``[batch, num_actions]``.
        weights: Detached per-state weights, same shape.

    Returns:
        The mean over states of the weighted sum.
    """

    return torch.sum(weights * log_prob, dim=1).mean()


def lagrangian_penalty(kl: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``sum alpha * KL`` with the multiplier stopped, so the policy step does not train it.

    Args:
        kl: KL averaged over the batch, a scalar or a vector with one entry per constraint.
        alpha: Positive multiplier of the same shape.

    Returns:
        The scalar penalty added to the policy loss.
    """

    return torch.sum(alpha.detach() * kl)


def dual_descent_loss(kl: torch.Tensor, alpha: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Dual ``sum alpha * (epsilon - stopgrad(KL))``. Descent implements ``alpha <- alpha - beta (epsilon - KL)``.

    Args:
        kl: KL averaged over the batch.
        alpha: Positive multiplier that receives the gradient.
        epsilon: Constraint value.

    Returns:
        The scalar dual loss.
    """

    return torch.sum(alpha * (epsilon - kl.detach()))
