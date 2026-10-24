"""One Gaussian policy forward for JAX and MLX.

The formulas are the ones in ``UnivariateGaussianPolicy`` and
``DiagonalGaussianPolicy``: layer norm only after the first linear, the custom
ELU, softplus via ``logaddexp``, and the tanh squash those classes use.
"""

from __future__ import annotations

from typing import Any

from abstract_rl.interop.namespace import LOG_SQRT_2PI, ArrayNS
from abstract_rl.interop.spec import GaussianSpec
from abstract_rl.policy.continuous.univariate_gaussian_policy import SQUASH_BOUND, SQUASH_U_MAX

_SQUASH_BOUND = SQUASH_BOUND
_SQUASH_U_MAX = SQUASH_U_MAX


def _activate(ns: ArrayNS, name: str, x: Any) -> Any:
    """Apply one activation name."""
    match name.lower():
        case "tanh":
            return ns.tanh(x)
        case "elu":
            return ns.elu(x)
        case "relu":
            return ns.relu(x)
        case "sigmoid":
            return ns.sigmoid(x)
        case "softplus":
            return ns.softplus(x)
        case "lrelu":
            return ns.leaky_relu(x)
        case "none" | "identity":
            return x
        case _:
            raise NotImplementedError(f"unknown activation {name}")


def _linear(ns: ArrayNS, theta: Any, layer: Any, x: Any) -> Any:
    """``x @ W.T + b`` with ``W`` stored row-major ``[out, in]``."""
    width = layer.n_weight
    weight = ns.reshape(theta[layer.weight_offset : layer.weight_offset + width], (layer.out_dim, layer.in_dim))
    bias = theta[layer.bias_offset : layer.bias_offset + layer.n_bias]
    y = x @ ns.transpose(weight) + bias
    if layer.layer_norm:
        y = ns.layer_norm(y)
    if len(layer.activation) == 1:
        return _activate(ns, layer.activation[0], y)
    parts = [_activate(ns, layer.activation[i], y[:, i : i + 1]) for i in range(layer.out_dim)]
    return ns.concatenate(parts, 1)


def network(ns: ArrayNS, spec: GaussianSpec, theta: Any, states: Any) -> Any:
    """The MLP, ``states`` shaped ``[D, obs]``."""
    x = states
    for layer in spec.layers:
        x = _linear(ns, theta, layer, x)
    return x


def suff_stats(ns: ArrayNS, spec: GaussianSpec, theta: Any, states: Any, fixed_log_sigma: Any = None) -> Any:
    """Sufficient statistics ``[mu, sigma]``, sigma strictly positive.

    Args:
        ns: Which framework evaluates the formulas.
        spec: The policy layout.
        theta: The flat trainable vector.
        states: A batch ``[D, obs]``.
        fixed_log_sigma: Shape ``(A,)``, required when ``cov_type`` is ``fixed``.

    Returns:
        A matrix ``[D, 2]`` (univariate) or ``[D, 2A]`` (diagonal).
    """
    out = network(ns, spec, theta, states)
    a_dim = spec.action_dim if spec.kind == "diagonal" else 1
    match spec.cov_type:
        case "state":
            mu, sigma = out[:, :a_dim], out[:, a_dim:]
            sigma = ns.maximum(sigma, spec.sigma_min) if spec.sigma_is_softplus else ns.softplus(sigma) + spec.sigma_min
            return ns.concatenate([mu, sigma], 1)
        case "single":
            assert spec.log_sigma_offset is not None
            log_sigma = theta[spec.log_sigma_offset : spec.log_sigma_offset + a_dim]
        case "fixed":
            if fixed_log_sigma is None:
                raise ValueError("fixed covariance needs fixed_log_sigma")
            log_sigma = fixed_log_sigma
        case _:
            raise NotImplementedError(spec.cov_type)
    sigma = ns.exp(log_sigma)
    sigma = ns.broadcast_to(ns.reshape(sigma, (1, a_dim)), (states.shape[0], a_dim))
    mu = out[:, :a_dim]
    return ns.concatenate([mu, sigma], 1)


def _split(spec: GaussianSpec, stats: Any) -> tuple[Any, Any]:
    a_dim = spec.action_dim if spec.kind == "diagonal" else 1
    return stats[:, :a_dim], stats[:, a_dim:]


def normal_log_prob(ns: ArrayNS, x: Any, mu: Any, sigma: Any) -> Any:
    """Elementwise log density of ``N(mu, sigma^2)``."""
    return -((x - mu) ** 2) / (2 * sigma**2) - ns.log(sigma) - LOG_SQRT_2PI


def normal_kl(ns: ArrayNS, p_mu: Any, p_sigma: Any, q_mu: Any, q_sigma: Any) -> Any:
    """Elementwise ``KL(N(p) || N(q))``."""
    var_ratio = (p_sigma / q_sigma) ** 2
    t1 = ((p_mu - q_mu) / q_sigma) ** 2
    return 0.5 * (var_ratio + t1 - 1 - ns.log(var_ratio))


def _squash_u(ns: ArrayNS, spec: GaussianSpec, actions: Any) -> tuple[Any, Any]:
    """Inverse squash and the log absolute Jacobian diagonal, per action component."""
    if spec.kind == "univariate":
        scale = spec.action_scale[0]
        y = ns.clip(actions / scale, -_SQUASH_BOUND, _SQUASH_BOUND)
        log_jac = ns.log(scale * (1 - y**2))
        return ns.arctanh(y), log_jac
    center = ns.const(spec.action_center)
    scale = ns.const(spec.action_scale)
    y = ns.clip((actions - center) / scale, -_SQUASH_BOUND, _SQUASH_BOUND)
    return ns.arctanh(y), ns.log(scale * (1 - y**2))


def log_prob(
    ns: ArrayNS, spec: GaussianSpec, theta: Any, actions: Any, states: Any, fixed_log_sigma: Any = None
) -> Any:
    """Log probability of ``actions`` under the policy at ``states``.

    Univariate actions are ``[D, K]`` and the result is ``[D, K]``. Diagonal
    actions are ``[D, K, A]`` or ``[D, A]`` and the result is ``[D, K]``.
    """
    stats = suff_stats(ns, spec, theta, states, fixed_log_sigma)
    return log_prob_from_stats(ns, spec, stats, actions)


def log_prob_from_stats(ns: ArrayNS, spec: GaussianSpec, stats: Any, actions: Any) -> Any:
    """Log probability given sufficient statistics already computed."""
    mu, sigma = _split(spec, stats)
    if spec.kind == "univariate":
        if spec.squash:
            u, log_jac = _squash_u(ns, spec, actions)
            return normal_log_prob(ns, u, mu, sigma) - log_jac
        return normal_log_prob(ns, actions, mu, sigma)
    if actions.ndim == 2:
        actions = actions[:, None, :]
    mu_b, sigma_b = mu[:, None, :], sigma[:, None, :]
    if spec.squash:
        u, log_jac = _squash_u(ns, spec, actions)
        return ns.sum(normal_log_prob(ns, u, mu_b, sigma_b) - log_jac, axis=-1)
    return ns.sum(normal_log_prob(ns, actions, mu_b, sigma_b), axis=-1)


def mode_from_stats(ns: ArrayNS, spec: GaussianSpec, stats: Any) -> Any:
    """Most likely action. Univariate ``[D, 1]``, diagonal ``[D, A]``."""
    mu, _sigma = _split(spec, stats)
    if not spec.squash:
        return mu
    if spec.kind == "univariate":
        return spec.action_scale[0] * ns.tanh(mu)
    center = ns.const(spec.action_center)
    scale = ns.const(spec.action_scale)
    return center + scale * ns.tanh(mu)


def sample_from_stats(ns: ArrayNS, spec: GaussianSpec, stats: Any, noise: Any) -> Any:
    """Reparameterised sample. ``noise`` is ``[D, K]`` or ``[D, K, A]``."""
    mu, sigma = _split(spec, stats)
    if spec.kind == "univariate":
        u = mu + sigma * noise
        if spec.squash:
            u = spec.action_scale[0] * ns.tanh(ns.clip(u, -_SQUASH_U_MAX, _SQUASH_U_MAX))
        return u
    u = mu[:, None, :] + sigma[:, None, :] * noise
    if spec.squash:
        center = ns.const(spec.action_center)
        scale = ns.const(spec.action_scale)
        u = center + scale * ns.tanh(ns.clip(u, -_SQUASH_U_MAX, _SQUASH_U_MAX))
    return u


def act_one(
    ns: ArrayNS,
    spec: GaussianSpec,
    theta: Any,
    obs: Any,
    noise: Any | None,
    deterministic: bool,
    fixed_log_sigma: Any = None,
) -> tuple[Any, Any]:
    """One observation to one action and its log probability.

    Args:
        obs: Shape ``(obs,)``.
        noise: Univariate ``(1, 1)`` or diagonal ``(1, 1, A)``. Ignored when ``deterministic``.
        deterministic: Use the mode instead of a sample.

    Returns:
        The action shaped ``(A,)`` and a scalar log probability.
    """
    stats = suff_stats(ns, spec, theta, obs[None, :], fixed_log_sigma)
    # ``deterministic`` is a Python bool, so this branch is static under jit.
    if deterministic or noise is None:
        drawn = mode_from_stats(ns, spec, stats)
    else:
        drawn = sample_from_stats(ns, spec, stats, noise)
    logp = log_prob_from_stats(ns, spec, stats, drawn)
    return drawn.reshape((spec.action_dim,)), logp.reshape(())


def entropy_from_stats(ns: ArrayNS, spec: GaussianSpec, stats: Any) -> Any:
    """Mean Gaussian entropy. Squashed policies have no closed form."""
    if spec.squash:
        raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
    _mu, sigma = _split(spec, stats)
    per = 0.5 + LOG_SQRT_2PI + ns.log(sigma)
    if spec.kind == "diagonal":
        per = ns.sum(per, axis=1)
    return ns.mean(per)
