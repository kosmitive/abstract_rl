"""Gaussian densities shared by the univariate, diagonal and full-covariance policies.

``head_activations`` and ``sigma_is_softplus`` describe the policy head. The benchmark configs sometimes put a
softplus on the sigma outputs inside the network; the policy must not apply a second one.

A full covariance is a state-dependent Cholesky factor ``L(s)``. The network emits the mean and the
lower triangle; the diagonal of ``L`` is ``softplus(raw) + sigma_floor``. ``sigma_floor`` is
``policy_min_std`` (default 0, optionally annealed to 0). Softplus is already positive, so the
historical additive ``1e-6`` is not applied. Samples are ``u = μ + L z``.

Samples from the squashed policy are ``a = center + scale * tanh(u)`` with ``u`` drawn from that
Gaussian. The squash is componentwise, including for a full covariance. The KL of a squashed policy
is the KL of those underlying Gaussians, because the squash is a bijection shared by both distributions.
"""

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


def head_activations(act_fn: Any, num_actions: int) -> tuple[list[Any], list[Any]]:
    """Resolve activations of a policy head with outputs ``[mu_1..mu_A, sigma_1..sigma_A]``.

    The last entry of ``act_fn`` is either one activation name for all outputs, a pair ``[mu_act, sigma_act]``
    which is expanded to one entry per output, or a list with one name per output.

    Args:
        act_fn: Per-layer activation names from the run config.
        num_actions: Action dimension.

    Returns:
        The activation list for the network and the activation names of the sigma outputs.
    """
    act_fn = list(act_fn)
    last = act_fn[-1]
    if isinstance(last, list | tuple):
        last = list(last)
        if len(last) == 2 and num_actions > 1:
            last = [last[0]] * num_actions + [last[1]] * num_actions
        elif len(last) not in (2, 2 * num_actions):
            raise ValueError(f"the activations of the last layer need 2 or {2 * num_actions} entries, got {len(last)}")
        sigma_acts = last[len(last) // 2 :]
        if len(act_fn) > 1:
            act_fn[-1] = last
        else:
            raise ValueError("per output activations require one entry per layer")
    else:
        sigma_acts = [last]
    return act_fn, sigma_acts


def sigma_is_softplus(sigma_acts: Sequence[Any]) -> bool:
    """Whether the network already applies a softplus to every sigma output."""
    flags = {isinstance(a, str) and a.lower() == "softplus" for a in sigma_acts}
    if len(flags) != 1:
        raise ValueError("either all or none of the sigma outputs may use a softplus activation")
    return flags.pop()


# Historical additive floor on a softplus standard deviation. Diagonals now add ``policy_min_std``
# instead (default 0). The constant stays so existing imports keep resolving.
SIGMA_MIN = 1e-6
SQUASH_U_MAX = 5.0
SQUASH_BOUND = math.tanh(SQUASH_U_MAX)
_LOG_2PI = math.log(2.0 * math.pi)
_ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)


class GaussianParams(eqx.Module):
    """Trainable Gaussian policy.

    Sufficient statistics are ``[mu_1..mu_A, sigma_1..sigma_A]``. ``log_sigma`` is set only for a
    state-independent standard deviation (``single`` is trained, ``fixed`` is a static tuple).
    """

    network: MultilayerPerceptron
    log_sigma: Float[Array, " act"] | None
    cov_type: str = eqx.field(static=True)
    squash: bool = eqx.field(static=True)
    sigma_is_softplus: bool = eqx.field(static=True)
    action_center: tuple[float, ...] = eqx.field(static=True)
    action_scale: tuple[float, ...] = eqx.field(static=True)
    fixed_log_sigma: tuple[float, ...] | None = eqx.field(static=True)
    # Schedule value, not a weight. ``flat_params.weight_spec`` keeps it out of Adam, TRPO and L2.
    sigma_floor: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.zeros((), dtype=jnp.float32))

    def _scale(self) -> Float[Array, " act"]:
        return jnp.asarray(self.action_scale, dtype=jnp.float32)

    def _center(self) -> Float[Array, " act"]:
        return jnp.asarray(self.action_center, dtype=jnp.float32)


def scheduled_sigma_floor(min_std: float, anneal_steps: int, env_steps: int) -> float:
    """Standard-deviation floor at ``env_steps``.

    ``anneal_steps <= 0`` keeps ``min_std`` for the whole run. Otherwise the floor starts at
    ``min_std`` and falls linearly to zero once ``env_steps`` reaches ``anneal_steps``.

    Args:
        min_std: Floor at the start of training.
        anneal_steps: Environment steps over which the floor decays. Zero keeps it constant.
        env_steps: Environment steps collected so far.

    Returns:
        The floor to add on a state-dependent diagonal, or to enforce on a state-independent std.
    """
    if anneal_steps <= 0:
        return float(min_std)
    done = min(max(int(env_steps), 0), int(anneal_steps)) / float(anneal_steps)
    return float(min_std) * (1.0 - done)


def softplus_inverse(value: float) -> float:
    """Inverse of softplus, matching the identity region of ``jax.nn.softplus`` above 20.

    Args:
        value: Positive softplus output.

    Returns:
        Bias whose softplus equals ``value``.
    """
    if value > 20.0:
        return float(value)
    return float(math.log(math.expm1(value)))


def positive_sigma(
    raw: Float[Array, "... act"],
    already_softplus: bool,
    floor: Float[Array, ""],
) -> Float[Array, "... act"]:
    """Standard deviation ``softplus(raw) + floor``.

    When the network already applied a softplus, ``raw`` is used in its place so the activation
    is not applied twice. ``floor`` is ``policy_min_std`` and is not a trained weight.

    Args:
        raw: Network output for one standard deviation, or for a Cholesky diagonal.
        already_softplus: Whether ``raw`` already passed through a softplus.
        floor: Amount added on the diagonal. Zero leaves a plain softplus.

    Returns:
        Positive widths of the same shape as ``raw``.
    """
    shift = jax.lax.stop_gradient(floor)
    if already_softplus:
        return raw + shift
    return jax.nn.softplus(raw) + shift


def _state_independent_sigma(
    log_sigma: Float[Array, " act"],
    floor: Float[Array, ""],
    like: Float[Array, "batch act"],
) -> Float[Array, "batch act"]:
    """``max(exp(log_sigma), floor)`` broadcast over the batch. A floor of zero leaves ``exp`` unchanged."""
    shift = jax.lax.stop_gradient(floor)
    return jnp.maximum(jnp.exp(log_sigma), shift)[None, :] + jnp.zeros_like(like)


def forward(params: GaussianParams, states: Float[Array, "batch obs"]) -> Float[Array, "batch stats"]:
    """Sufficient statistics ``[mu, sigma]`` for every state.

    Args:
        params: Policy parameters.
        states: States, shape ``[batch, obs]``.

    Returns:
        Sufficient statistics. Diagonal policies store ``[mu, sigma]``. A full covariance stores
        the mean and the packed Cholesky factor.
    """
    match params.cov_type:
        case "state":
            out = params.network.forward(states)
            act = out.shape[-1] // 2
            mu, raw = out[:, :act], out[:, act:]
            sigma = positive_sigma(raw, params.sigma_is_softplus, params.sigma_floor)
        case "single":
            mu = params.network.forward(states)
            assert params.log_sigma is not None
            sigma = _state_independent_sigma(params.log_sigma, params.sigma_floor, mu)
        case "fixed":
            mu = params.network.forward(states)
            assert params.fixed_log_sigma is not None
            log_sigma = jnp.asarray(params.fixed_log_sigma, dtype=jnp.float32)
            sigma = _state_independent_sigma(log_sigma, params.sigma_floor, mu)
        case "full":
            out = params.network.forward(states)
            action_dim = len(params.action_center)
            mu = out[:, :action_dim]
            packed = with_positive_diagonal(
                out[:, action_dim:], action_dim, params.sigma_is_softplus, params.sigma_floor
            )
            return jnp.concatenate([mu, packed], axis=-1)
        case _:
            raise NotImplementedError(f"unknown covariance type {params.cov_type}")
    return jnp.concatenate([mu, sigma], axis=-1)


def _split(stats: Float[Array, "batch stats"]) -> tuple[Float[Array, "batch act"], Float[Array, "batch act"]]:
    act = stats.shape[-1] // 2
    return stats[:, :act], stats[:, act:]


def _mu_sigma(
    params: GaussianParams,
    states: Float[Array, "batch obs"] | None,
    suff_stats: Float[Array, "batch stats"] | None,
) -> tuple[Float[Array, "batch act"], Float[Array, "batch act"]]:
    match (states is None, suff_stats is None):
        case (False, True):
            assert states is not None
            return _split(forward(params, states))
        case (True, False):
            assert suff_stats is not None
            return _split(suff_stats)
        case _:
            raise ValueError("pass either states or suff_stats")


def _stored_stats(
    params: GaussianParams,
    states: Float[Array, "batch obs"] | None,
    suff_stats: Float[Array, "batch stats"] | None,
) -> Float[Array, "batch stats"]:
    match (states is None, suff_stats is None):
        case (False, True):
            assert states is not None
            return forward(params, states)
        case (True, False):
            assert suff_stats is not None
            return suff_stats
        case _:
            raise ValueError("pass either states or suff_stats")


def _full_mu_factor(
    params: GaussianParams,
    states: Float[Array, "batch obs"] | None,
    suff_stats: Float[Array, "batch stats"] | None,
) -> tuple[Float[Array, "batch act"], Float[Array, "batch act act"]]:
    stats = _stored_stats(params, states, suff_stats)
    action_dim = len(params.action_center)
    return stats[:, :action_dim], unpack_lower(stats[:, action_dim:], action_dim)


def _full_mu(
    params: GaussianParams,
    states: Float[Array, "batch obs"] | None,
    suff_stats: Float[Array, "batch stats"] | None,
) -> Float[Array, "batch act"]:
    mu, _factor = _full_mu_factor(params, states, suff_stats)
    return mu


def _full_log_prob(
    params: GaussianParams,
    actions: Float[Array, "..."],
    states: Float[Array, "batch obs"] | None,
    suff_stats: Float[Array, "batch stats"] | None,
) -> Float[Array, "batch k"]:
    """Log density of a full-covariance Gaussian, with the same squash as the diagonal policy."""
    mu, factor = _full_mu_factor(params, states, suff_stats)
    acted = as_action_batch(actions, mu.shape[-1])
    if not params.squash:
        return cholesky_log_prob(acted, mu, factor)
    y = jnp.clip((acted - params._center()) / params._scale(), -SQUASH_BOUND, SQUASH_BOUND)
    u = jnp.arctanh(y)
    log_jac = jnp.log(params._scale() * (1.0 - jnp.square(y))).sum(axis=-1)
    return cholesky_log_prob(u, mu, factor) - log_jac


def _gaussian_log_prob(
    x: Float[Array, "..."], mu: Float[Array, "..."], sigma: Float[Array, "..."]
) -> Float[Array, "..."]:
    return -0.5 * ((x - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * _LOG_2PI


def as_action_batch(actions: Float[Array, "..."], act_dim: int) -> Float[Array, "batch k act"]:
    """Normalise actions to ``[batch, k, act]``.

    A vector ``[batch, act]`` is one action per state. A matrix whose last axis is not the action
    dimension is treated as ``[batch, k]`` for a one-dimensional action.
    """
    if actions.ndim == 3:
        return actions
    if actions.ndim == 1:
        actions = actions[:, None]
    if actions.shape[-1] == act_dim:
        return actions[:, None, :]
    if act_dim == 1:
        return actions[:, :, None]
    raise ValueError(f"cannot interpret actions of shape {actions.shape} with action dim {act_dim}")


def log_prob(
    params: GaussianParams,
    actions: Float[Array, "..."],
    states: Float[Array, "batch obs"] | None = None,
    suff_stats: Float[Array, "batch stats"] | None = None,
) -> Float[Array, "batch k"]:
    """Log density of actions under the (optionally squashed) Gaussian.

    Args:
        params: Policy parameters.
        actions: ``[batch, k, act]`` or ``[batch, act]``.
        states: States, if ``suff_stats`` is not given.
        suff_stats: Precomputed sufficient statistics.

    Returns:
        Log probabilities, shape ``[batch, k]``.
    """
    if params.cov_type == "full":
        return _full_log_prob(params, actions, states, suff_stats)
    mu, sigma = _mu_sigma(params, states, suff_stats)
    act = as_action_batch(actions, mu.shape[-1])
    mu_k, sigma_k = mu[:, None, :], sigma[:, None, :]
    if not params.squash:
        return _gaussian_log_prob(act, mu_k, sigma_k).sum(axis=-1)
    y = jnp.clip((act - params._center()) / params._scale(), -SQUASH_BOUND, SQUASH_BOUND)
    u = jnp.arctanh(y)
    log_jac = jnp.log(params._scale() * (1.0 - jnp.square(y)))
    return (_gaussian_log_prob(u, mu_k, sigma_k) - log_jac).sum(axis=-1)


def mode(
    params: GaussianParams,
    states: Float[Array, "batch obs"] | None = None,
    suff_stats: Float[Array, "batch stats"] | None = None,
) -> Float[Array, "batch act"]:
    """Most likely action of each state."""
    if params.cov_type == "full":
        mu = _full_mu(params, states, suff_stats)
    else:
        mu, _ = _mu_sigma(params, states, suff_stats)
    if params.squash:
        return params._center() + params._scale() * jnp.tanh(mu)
    return mu


def sample(
    params: GaussianParams,
    key: PRNGKeyArray,
    num_actions: int,
    states: Float[Array, "batch obs"] | None = None,
    suff_stats: Float[Array, "batch stats"] | None = None,
) -> Float[Array, "batch k act"]:
    """Draw ``num_actions`` actions for every state.

    Args:
        params: Policy parameters.
        key: Random key.
        num_actions: Samples per state.
        states: States, if ``suff_stats`` is not given.
        suff_stats: Precomputed sufficient statistics.

    Returns:
        Actions, shape ``[batch, k, act]``.
    """
    if params.cov_type == "full":
        mu, factor = _full_mu_factor(params, states, suff_stats)
        noise = jax.random.normal(key, (mu.shape[0], num_actions, mu.shape[-1]), dtype=jnp.float32)
        u = mu[:, None, :] + jnp.einsum("bij,bkj->bki", factor, noise)
    else:
        mu, sigma = _mu_sigma(params, states, suff_stats)
        noise = jax.random.normal(key, (mu.shape[0], num_actions, mu.shape[-1]), dtype=jnp.float32)
        u = mu[:, None, :] + sigma[:, None, :] * noise
    if not params.squash:
        return u
    u = jnp.clip(u, -SQUASH_U_MAX, SQUASH_U_MAX)
    return params._center() + params._scale() * jnp.tanh(u)


def tril_diag_index(action_dim: int) -> np.ndarray:
    """Row-major index of each diagonal entry inside a packed lower triangle."""
    index = np.arange(action_dim, dtype=np.int32)
    return index * (index + 3) // 2


def pack_lower(factor: Float[Array, "batch act act"]) -> Float[Array, "batch tril"]:
    """Pack the lower triangle, including the diagonal, in row-major order."""
    rows, cols = np.tril_indices(factor.shape[-1])
    return factor[:, rows, cols]


def unpack_lower(packed: Float[Array, "batch tril"], action_dim: int) -> Float[Array, "batch act act"]:
    """Inverse of ``pack_lower``. Entries above the diagonal are zero."""
    rows, cols = np.tril_indices(action_dim)
    factor = jnp.zeros((packed.shape[0], action_dim, action_dim), dtype=packed.dtype)
    return factor.at[:, rows, cols].set(packed)


def with_positive_diagonal(
    packed: Float[Array, "batch tril"],
    action_dim: int,
    already_softplus: bool,
    floor: Float[Array, ""],
) -> Float[Array, "batch tril"]:
    """Softplus the diagonal of a packed Cholesky factor and add ``floor``. Off-diagonal entries stay raw."""
    diagonal = positive_sigma(packed[:, tril_diag_index(action_dim)], already_softplus, floor)
    return packed.at[:, tril_diag_index(action_dim)].set(diagonal)


def cholesky_log_prob(
    values: Float[Array, "batch k act"],
    mu: Float[Array, "batch act"],
    factor: Float[Array, "batch act act"],
) -> Float[Array, "batch k"]:
    """Log density of ``N(mu, L L^T)`` at ``values``.

    ``L`` is lower triangular. The density is
    ``-1/2 ||L^{-1}(x - mu)||^2 - sum log diag(L) - (k/2) log(2 pi)``.
    """
    batch, count, action_dim = values.shape
    residual = jnp.reshape(values - mu[:, None, :], (batch * count, action_dim))
    tiled = jnp.broadcast_to(factor[:, None, :, :], (batch, count, action_dim, action_dim))
    solved = jax.scipy.linalg.solve_triangular(
        jnp.reshape(tiled, (batch * count, action_dim, action_dim)),
        residual[..., None],
        lower=True,
    )[..., 0]
    solved = jnp.reshape(solved, (batch, count, action_dim))
    log_diag = jnp.sum(jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * jnp.sum(jnp.square(solved), axis=-1) - log_diag[:, None] - 0.5 * action_dim * _LOG_2PI


def kl_mean_covariance(
    mu: Float[Array, "batch act"],
    sigma: Float[Array, "batch act"],
    other_mu: Float[Array, "batch act"],
    other_sigma: Float[Array, "batch act"],
) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
    """``KL_mu`` and ``KL_Sigma`` of ``KL(N(mu, sigma) || N(other))``.

    ``KL_mu`` fixes both covariances at the target. ``KL_Sigma`` fixes both means at the target.
    The two terms sum to ``kl_normal``.
    """
    inv_var = 1.0 / jnp.square(other_sigma)
    kl_mu = 0.5 * jnp.sum(jnp.square(mu - other_mu) * inv_var, axis=-1)
    ratio = jnp.square(sigma) * inv_var
    kl_sigma = 0.5 * jnp.sum(ratio - 1.0 - jnp.log(ratio), axis=-1)
    return kl_mu, kl_sigma


def kl_mean_covariance_cholesky(
    mu: Float[Array, "batch act"],
    factor: Float[Array, "batch act act"],
    other_mu: Float[Array, "batch act"],
    other_factor: Float[Array, "batch act act"],
) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
    """``KL_mu`` and ``KL_Sigma`` of ``KL(N(mu, L L^T) || N(other_mu, L_other L_other^T))``.

    ``KL_mu = 1/2 ||L_other^{-1} (mu - other_mu)||^2`` uses the target covariance.
    ``KL_Sigma = 1/2 (||L_other^{-1} L||_F^2 - k + log det target - log det current)`` uses the target mean.
    """
    delta = jax.scipy.linalg.solve_triangular(other_factor, (mu - other_mu)[..., None], lower=True)[..., 0]
    quad = jnp.sum(jnp.square(delta), axis=-1)
    mapped = jax.scipy.linalg.solve_triangular(other_factor, factor, lower=True)
    trace = jnp.sum(jnp.square(mapped), axis=(-2, -1))
    logdet = 2.0 * (
        jnp.sum(jnp.log(jnp.diagonal(other_factor, axis1=-2, axis2=-1)), axis=-1)
        - jnp.sum(jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), axis=-1)
    )
    return 0.5 * quad, 0.5 * (trace - mu.shape[-1] + logdet)


def kl_parts(
    current: Float[Array, "batch stats"],
    target: Float[Array, "batch stats"],
    cov_type: str,
    action_dim: int,
) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
    """Per-state mean and covariance KL of ``KL(current || target)``.

    Diagonal statistics are ``[mu, sigma]``. Full-covariance statistics are the mean followed by
    the packed Cholesky factor.
    """
    if cov_type == "full":
        return kl_mean_covariance_cholesky(
            current[:, :action_dim],
            unpack_lower(current[:, action_dim:], action_dim),
            target[:, :action_dim],
            unpack_lower(target[:, action_dim:], action_dim),
        )
    return kl_mean_covariance(
        current[:, :action_dim],
        current[:, action_dim:],
        target[:, :action_dim],
        target[:, action_dim:],
    )


def kl_normal(
    mu: Float[Array, "batch act"],
    sigma: Float[Array, "batch act"],
    other_mu: Float[Array, "batch act"],
    other_sigma: Float[Array, "batch act"],
) -> Float[Array, " batch"]:
    """KL(N(mu, sigma) || N(other_mu, other_sigma)), summed over action dimensions."""
    var_ratio = (jnp.square(sigma) + jnp.square(mu - other_mu)) / (2.0 * jnp.square(other_sigma))
    kl = jnp.log(other_sigma / sigma) + var_ratio - 0.5
    return kl.sum(axis=-1)


def kl_divergence(
    params: GaussianParams,
    target: GaussianParams,
    states: Float[Array, "batch obs"],
    suff_stats: Float[Array, "batch stats"] | None = None,
    mode_name: str = "i-projection",
    est: str = "sum",
) -> Float[Array, ""]:
    """Analytic KL between this Gaussian and the target policy, reduced over the batch.

    Args:
        params: Current policy. Ignored when ``suff_stats`` is passed.
        target: Target (old) policy.
        states: States the KL is averaged over.
        suff_stats: Optional sufficient statistics of the current policy.
        mode_name: ``i-projection`` is KL(current || target), ``m-projection`` the reverse.
        est: ``sum`` or ``mean`` over states.
    """
    current = forward(params, states) if suff_stats is None else suff_stats
    old = jax.lax.stop_gradient(forward(target, states))
    if params.cov_type == "full":
        action_dim = len(params.action_center)
        match mode_name:
            case "i-projection":
                kl_mu, kl_sigma = kl_parts(current, old, "full", action_dim)
            case "m-projection":
                kl_mu, kl_sigma = kl_parts(old, current, "full", action_dim)
            case _:
                raise NotImplementedError(mode_name)
        per_state = kl_mu + kl_sigma
        match est:
            case "sum":
                return per_state.sum()
            case "mean":
                return per_state.mean()
            case _:
                raise NotImplementedError(est)
    mu, sigma = _split(current)
    t_mu, t_sigma = _split(old)
    match mode_name:
        case "i-projection":
            per_state = kl_normal(mu, sigma, t_mu, t_sigma)
        case "m-projection":
            per_state = kl_normal(t_mu, t_sigma, mu, sigma)
        case _:
            raise NotImplementedError(mode_name)
    match est:
        case "sum":
            return per_state.sum()
        case "mean":
            return per_state.mean()
        case _:
            raise NotImplementedError(est)


def entropy_from_sigma(sigma: Float[Array, "batch act"]) -> Float[Array, ""]:
    """Closed-form entropy of a diagonal Gaussian, mean over the batch."""
    return (_ENTROPY_CONST + jnp.log(sigma)).sum(axis=-1).mean()


def entropy_from_cholesky(factor: Float[Array, "batch act act"]) -> Float[Array, ""]:
    """Closed-form entropy of ``N(mu, L L^T)``, mean over the batch."""
    log_diag = jnp.sum(jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), axis=-1)
    return (_ENTROPY_CONST * factor.shape[-1] + log_diag).mean()


def likelihood_ratio(
    params: GaussianParams,
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch act"],
    old_log_prob: Float[Array, " batch"],
    advantages: Float[Array, " batch"],
) -> Float[Array, ""]:
    """Negative TRPO surrogate (standardised advantages), for minimisation.

    Args:
        params: Policy being optimised.
        states: States of the on-policy batch.
        actions: Actions that were executed, shape ``[batch, act]``.
        old_log_prob: Log density of those actions under the behaviour policy.
        advantages: GAE advantages.

    Returns:
        Scalar surrogate. Lower is a better policy step.
    """
    stats = forward(params, states)
    new_log_prob = log_prob(params, actions, suff_stats=stats).reshape(-1)
    adv = advantages.reshape(-1)
    # torch.std is the unbiased estimate
    denom = jnp.std(adv, ddof=1) if adv.shape[0] > 1 else jnp.std(adv)
    standardised = (adv - jnp.mean(adv)) / (denom + 1e-8)
    ratio = jnp.exp(new_log_prob - old_log_prob.reshape(-1))
    return -jnp.mean(ratio * standardised)


def act_stochastic(
    params: GaussianParams, states: Float[Array, "batch obs"], key: PRNGKeyArray
) -> tuple[Float[Array, "batch act"], Float[Array, " batch"]]:
    """Sample one action per state and its log density. Jitted by the caller."""
    stats = forward(params, states)
    actions = sample(params, key, 1, suff_stats=stats)
    log_density = log_prob(params, actions, suff_stats=stats)
    return actions[:, 0, :], log_density[:, 0]


def act_deterministic(
    params: GaussianParams, states: Float[Array, "batch obs"]
) -> tuple[Float[Array, "batch act"], Float[Array, " batch"]]:
    """Mode of the policy and its log density."""
    stats = forward(params, states)
    actions = mode(params, suff_stats=stats)
    log_density = log_prob(params, actions, suff_stats=stats)
    return actions, log_density[:, 0]
