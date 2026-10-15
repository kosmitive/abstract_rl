"""Dual of the non-parametric MPO E-step."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)


def temperature_dual(q_values: Float[Array, "batch k"], eta: Float[Array, ""], eps: float) -> Float[Array, ""]:
    """Dual ``g(η) = η ε + η mean_s log mean_a exp(Q(s, a) / η)``.

    Args:
        q_values: Q samples, shape ``[batch, k]``.
        eta: Temperature.
        eps: KL bound of the E-step.

    Returns:
        Scalar dual value.
    """
    max_q = jnp.max(q_values, axis=1, keepdims=True)
    log_mean_exp = jax.nn.logsumexp((q_values - max_q) / eta, axis=1) - jnp.log(q_values.shape[1])
    return eta * eps + jnp.mean(max_q) + eta * jnp.mean(log_mean_exp)


def positive_temperature(raw: Float[Array, ""]) -> Float[Array, ""]:
    """Temperature ``softplus(raw)``, clipped to a positive range."""
    return jnp.clip(jax.nn.softplus(raw), 1e-6, 1e6)


def temperature_raw(eta: float) -> Float[Array, ""]:
    """Raw parameter whose softplus equals ``eta``."""
    value = float(np.clip(eta, 1e-4, 1e5))
    return jnp.asarray(np.log(np.expm1(value)), dtype=jnp.float32)


def descend_temperature(
    q_values: Float[Array, "batch k"],
    raw: Float[Array, ""],
    opt_state: Any,
    tx: optax.GradientTransformation,
    eps: float,
) -> tuple[Float[Array, ""], Any, Float[Array, ""]]:
    """One Adam step of the dual in the softplus parameter of η.

    Args:
        q_values: Q samples of one minibatch, shape ``[batch, k]``.
        raw: Unconstrained parameter, ``η = softplus(raw)``.
        opt_state: Adam state for ``raw``.
        tx: Adam transform. Its learning rate is the temperature step size.
        eps: KL bound of the E-step.

    Returns:
        Updated raw parameter, updated Adam state, and the positive temperature.
    """

    def loss(raw_: Float[Array, ""]) -> Float[Array, ""]:
        return temperature_dual(q_values, positive_temperature(raw_), eps)

    _value, grad = jax.value_and_grad(loss)(raw)
    updates, opt_state = tx.update(grad, opt_state, raw)
    raw = eqx.apply_updates(raw, updates)
    return raw, opt_state, positive_temperature(raw)
