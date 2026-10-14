"""NumPy rollout for the Gaussian policy.

``ContinuousPolicy.act`` runs once per environment step. Entering a jitted program of that size
costs more than a two-layer network, so the rollout evaluates the same squash, standard deviation
and log Jacobian as ``gaussian_math`` in NumPy. Parameter updates keep using the JAX functions.
Rollout noise comes from a NumPy generator seeded once from the run's JAX key, so a step does not
dispatch a key split.
"""

from dataclasses import dataclass

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from abstract_rl.policy.continuous.gaussian_math import (
    _LOG_2PI,
    SIGMA_MIN,
    SQUASH_BOUND,
    SQUASH_U_MAX,
    GaussianParams,
)


@dataclass(frozen=True)
class NumpyPolicy:
    """Host copy of the weights a rollout needs. Rebuilt when the parameter object changes."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activations: tuple[tuple[str, ...], ...]
    layer_norm: bool
    cov_type: str
    squash: bool
    sigma_softplus: bool
    center: np.ndarray
    scale: np.ndarray
    log_sigma: np.ndarray | None


def extract_numpy_policy(params: GaussianParams) -> NumpyPolicy:
    """Copy the policy weights to NumPy arrays."""
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for layer in params.network.layers:
        weights.append(np.asarray(layer.weight, dtype=np.float32))
        if layer.bias is None:
            biases.append(np.zeros((layer.weight.shape[0],), dtype=np.float32))
        else:
            biases.append(np.asarray(layer.bias, dtype=np.float32))
    log_sigma: np.ndarray | None
    match params.cov_type:
        case "single":
            assert params.log_sigma is not None
            log_sigma = np.asarray(params.log_sigma, dtype=np.float32)
        case "fixed":
            assert params.fixed_log_sigma is not None
            log_sigma = np.asarray(params.fixed_log_sigma, dtype=np.float32)
        case "state":
            log_sigma = None
        case _:
            raise NotImplementedError(params.cov_type)
    return NumpyPolicy(
        weights=tuple(weights),
        biases=tuple(biases),
        activations=params.network.activations,
        layer_norm=bool(params.network.layer_norm),
        cov_type=params.cov_type,
        squash=bool(params.squash),
        sigma_softplus=bool(params.sigma_is_softplus),
        center=np.asarray(params.action_center, dtype=np.float32),
        scale=np.asarray(params.action_scale, dtype=np.float32),
        log_sigma=log_sigma,
    )


def seed_generator(key: PRNGKeyArray) -> np.random.Generator:
    """Seed a rollout generator from one JAX key."""
    bits = np.asarray(jax.random.bits(key)).reshape(-1)
    return np.random.default_rng(int(bits[0]))


def act_numpy(spec: NumpyPolicy, states: np.ndarray, noise: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """One action per row and its log density.

    Args:
        spec: Host copy of the policy.
        states: Observations, shape ``[batch, obs]``, float32.
        noise: Standard normal noise, shape ``[batch, act]``. ``None`` selects the mode.

    Returns:
        Actions ``[batch, act]`` and log densities ``[batch]``, both float32.
    """
    mu, sigma = _mu_sigma(spec, states)
    if noise is None:
        actions = spec.center + spec.scale * np.tanh(mu) if spec.squash else mu
    else:
        raw = mu + sigma * noise
        if spec.squash:
            raw = np.clip(raw, -SQUASH_U_MAX, SQUASH_U_MAX)
            actions = spec.center + spec.scale * np.tanh(raw)
        else:
            actions = raw
    actions = np.asarray(actions, dtype=np.float32)
    return actions, _log_prob(spec, actions, mu, sigma)


def _softplus(values: np.ndarray) -> np.ndarray:
    """``jax.nn.softplus`` for float32: identity past the overflow threshold, else ``log1p(exp)``."""
    return np.where(values > 20.0, values, np.log1p(np.exp(np.minimum(values, 20.0)))).astype(np.float32)


def _activate_one(name: str, values: np.ndarray) -> np.ndarray:
    match name.lower():
        case "none":
            return values
        case "softplus":
            return _softplus(values)
        case "elu":
            return np.where(values > 0, values, np.expm1(values)).astype(np.float32)
        case "prelu":
            return np.where(values >= 0, values, 0.25 * values).astype(np.float32)
        case "relu":
            return np.maximum(values, 0).astype(np.float32)
        case "lrelu":
            return np.where(values >= 0, values, 0.01 * values).astype(np.float32)
        case "tanh":
            return np.tanh(values).astype(np.float32)
        case "sigmoid":
            return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
        case _:
            raise NotImplementedError(name)


def _activate(names: tuple[str, ...], values: np.ndarray) -> np.ndarray:
    if len(names) == 1:
        return _activate_one(names[0], values)
    columns = [_activate_one(names[index], values[:, index : index + 1]) for index in range(values.shape[-1])]
    return np.concatenate(columns, axis=-1)


def _mlp(spec: NumpyPolicy, states: np.ndarray) -> np.ndarray:
    hidden = states @ spec.weights[0].T + spec.biases[0]
    if spec.layer_norm:
        centered = hidden - hidden.mean(axis=-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        hidden = centered / np.sqrt(variance + np.float32(1e-5))
    hidden = _activate(spec.activations[0], hidden)
    for index in range(1, len(spec.weights)):
        hidden = hidden @ spec.weights[index].T + spec.biases[index]
        hidden = _activate(spec.activations[index], hidden)
    return hidden.astype(np.float32)


def _positive_sigma(raw: np.ndarray, already_softplus: bool) -> np.ndarray:
    if already_softplus:
        return np.maximum(raw, np.float32(SIGMA_MIN))
    return _softplus(raw) + np.float32(SIGMA_MIN)


def _mu_sigma(spec: NumpyPolicy, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    match spec.cov_type:
        case "state":
            out = _mlp(spec, states)
            act = out.shape[-1] // 2
            mu = out[:, :act]
            sigma = _positive_sigma(out[:, act:], spec.sigma_softplus)
        case "single" | "fixed":
            mu = _mlp(spec, states)
            assert spec.log_sigma is not None
            sigma = np.exp(spec.log_sigma).astype(np.float32) + np.zeros_like(mu)
        case _:
            raise NotImplementedError(spec.cov_type)
    return mu.astype(np.float32), sigma.astype(np.float32)


def _log_prob(spec: NumpyPolicy, actions: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    acted = actions[:, None, :]
    mu_k = mu[:, None, :]
    sigma_k = sigma[:, None, :]
    if not spec.squash:
        density = -0.5 * ((acted - mu_k) / sigma_k) ** 2 - np.log(sigma_k) - 0.5 * _LOG_2PI
        return density.sum(axis=-1)[:, 0].astype(np.float32)
    y = np.clip((acted - spec.center) / spec.scale, -SQUASH_BOUND, SQUASH_BOUND)
    u = np.arctanh(y)
    log_jac = np.log(spec.scale * (1.0 - np.square(y)))
    density = -0.5 * ((u - mu_k) / sigma_k) ** 2 - np.log(sigma_k) - 0.5 * _LOG_2PI
    return (density - log_jac).sum(axis=-1)[:, 0].astype(np.float32)
