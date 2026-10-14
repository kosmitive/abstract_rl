"""Diagonal Gaussian densities shared by the univariate and multi-dimensional policies.

``head_activations`` and ``sigma_is_softplus`` describe the policy head. The benchmark configs sometimes put a
softplus on the sigma outputs inside the network; the policy must not apply a second one.


Samples from the squashed policy are ``a = center + scale * tanh(u)`` with ``u ~ N(mu, sigma)``.
The KL of a squashed policy is the KL of those underlying Gaussians, because the squash is a bijection
shared by both distributions.
"""

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
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

    def _scale(self) -> Float[Array, " act"]:
        return jnp.asarray(self.action_scale, dtype=jnp.float32)

    def _center(self) -> Float[Array, " act"]:
        return jnp.asarray(self.action_center, dtype=jnp.float32)


def positive_sigma(raw: Float[Array, "... act"], already_softplus: bool) -> Float[Array, "... act"]:
    """Make a standard deviation strictly positive without applying softplus twice."""
    if already_softplus:
        return jnp.maximum(raw, SIGMA_MIN)
    return jax.nn.softplus(raw) + SIGMA_MIN


def forward(params: GaussianParams, states: Float[Array, "batch obs"]) -> Float[Array, "batch stats"]:
    """Sufficient statistics ``[mu, sigma]`` for every state.

    Args:
        params: Policy parameters.
        states: States, shape ``[batch, obs]``.

    Returns:
        Sufficient statistics, shape ``[batch, 2 * act]``.
    """
    match params.cov_type:
        case "state":
            out = params.network.forward(states)
            act = out.shape[-1] // 2
            mu, raw = out[:, :act], out[:, act:]
            sigma = positive_sigma(raw, params.sigma_is_softplus)
        case "single":
            mu = params.network.forward(states)
            assert params.log_sigma is not None
            sigma = jnp.exp(params.log_sigma)[None, :] + jnp.zeros_like(mu)
        case "fixed":
            mu = params.network.forward(states)
            assert params.fixed_log_sigma is not None
            sigma = jnp.exp(jnp.asarray(params.fixed_log_sigma, dtype=jnp.float32))[None, :] + jnp.zeros_like(mu)
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
    mu, sigma = _mu_sigma(params, states, suff_stats)
    noise = jax.random.normal(key, (mu.shape[0], num_actions, mu.shape[-1]), dtype=jnp.float32)
    u = mu[:, None, :] + sigma[:, None, :] * noise
    if not params.squash:
        return u
    u = jnp.clip(u, -SQUASH_U_MAX, SQUASH_U_MAX)
    return params._center() + params._scale() * jnp.tanh(u)


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
