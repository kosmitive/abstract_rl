"""Dual of the non-parametric MPO E-step."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from abstract_rl.misc.cli_printer import CliPrinter

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


def masked_temperature_dual(
    q_values: Float[Array, "batch k"], size: Array, eta: Float[Array, ""], eps: float
) -> Float[Array, ""]:
    """``temperature_dual`` averaged over the first ``size`` rows of a fixed allocation.

    The tail is excluded from the mean, which is the same dual as passing ``q_values[:size]``
    without compiling a new shape every time the replay grows.
    """
    valid = (jnp.arange(q_values.shape[0]) < size).astype(q_values.dtype)
    absent = jnp.array(-1.0e30, dtype=q_values.dtype)
    present = valid[:, None] > 0
    max_q = jnp.max(jnp.where(present, q_values, absent), axis=1, keepdims=True)
    shifted = jnp.where(present, (q_values - max_q) / eta, absent)
    log_mean_exp = jax.nn.logsumexp(shifted, axis=1) - jnp.log(q_values.shape[1])
    count = jnp.maximum(jnp.sum(valid), 1.0)
    return eta * eps + jnp.sum(max_q[:, 0] * valid) / count + eta * jnp.sum(log_mean_exp * valid) / count


def _masked_stats(q_values: Float[Array, "batch k"], size: Array) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Mean and standard deviation of the stored prefix, for the temperature log line."""
    valid = (jnp.arange(q_values.shape[0]) < size).astype(q_values.dtype)
    count = jnp.maximum(jnp.sum(valid), 1.0) * q_values.shape[1]
    present = valid[:, None] > 0
    mean = jnp.sum(jnp.where(present, q_values, 0)) / count
    variance = jnp.sum(jnp.where(present, jnp.square(q_values - mean), 0)) / count
    return mean, jnp.sqrt(variance)


_masked_stats_jit = jax.jit(_masked_stats)


@jax.jit
def _newton_temperature(
    q_values: Float[Array, "batch k"],
    size: Array,
    eta0: Float[Array, ""],
    eps: float,
    max_steps: int,
) -> tuple[Float[Array, ""], Array, Float[Array, ""]]:
    """Newton steps on the scalar dual. The minimum is the same point L-BFGS-B was solving for.

    The dual is convex in the temperature, so a damped Newton step reaches that minimum without
    a host synchronisation on every iteration.
    """

    def loss(eta: Float[Array, ""]) -> Float[Array, ""]:
        return masked_temperature_dual(q_values, size, eta, eps)

    def cond(state: tuple[Float[Array, ""], Array, Array, Float[Array, ""]]) -> Array:
        _eta, step, done, _dual = state
        return (step < max_steps) & (~done)

    def body(
        state: tuple[Float[Array, ""], Array, Array, Float[Array, ""]],
    ) -> tuple[Float[Array, ""], Array, Array, Float[Array, ""]]:
        eta, step, _done, dual = state
        grad = jax.grad(loss)(eta)
        hess = jnp.maximum(jax.grad(jax.grad(loss))(eta), 1e-12)

        def shrink(
            carry: tuple[Float[Array, ""], Float[Array, ""], Array], alpha: Float[Array, ""]
        ) -> tuple[tuple[Float[Array, ""], Float[Array, ""], Array], None]:
            best, best_loss, found = carry
            trial = jnp.clip(eta - alpha * grad / hess, 1e-6, 1e6)
            value = loss(trial)
            take = (value <= dual) & (~found) & jnp.isfinite(value)
            best = jnp.where(take, trial, best)
            best_loss = jnp.where(take, value, best_loss)
            return (best, best_loss, found | take), None

        alphas = jnp.asarray([1.0, 0.5, 0.25, 0.125, 1e-2], dtype=jnp.float64)
        (nxt, nxt_loss, found), _unused = jax.lax.scan(shrink, (eta, dual, jnp.bool_(False)), alphas)
        done = (jnp.abs(grad) < 1e-6) | (~found)
        return jnp.where(found, nxt, eta), step + jnp.int32(1), done, jnp.where(found, nxt_loss, dual)

    eta, steps, _done, dual = jax.lax.while_loop(
        cond,
        body,
        (jnp.clip(eta0, 1e-6, 1e6), jnp.int32(0), jnp.bool_(False), loss(jnp.clip(eta0, 1e-6, 1e6))),
    )
    return eta, steps, dual


def find_optimal_temperature(
    q_values: Float[Array, "batch k"] | np.ndarray,
    eps: float,
    eta0: float = 1.0,
    lr: float | None = None,
    max_steps: int = 200,
    size: int | None = None,
) -> Float[Array, " 1"]:
    """Minimise the convex temperature dual.

    Args:
        q_values: Q samples, shape ``[batch, k]``.
        eps: KL bound.
        eta0: Initial temperature.
        lr: Unused. Kept so older call sites still pass a learning rate.
        max_steps: Iteration cap.
        size: How many leading rows are real samples. ``None`` uses every row. Passing the
            replay capacity with this limit keeps the compiled dual fixed while the buffer fills.

    Returns:
        Optimal temperature, shape ``[1]``, float32.
    """
    del lr
    cli = CliPrinter().instance
    q_values = jnp.asarray(q_values, dtype=jnp.float64)
    limit = jnp.int32(q_values.shape[0] if size is None else size)
    mean, std = _masked_stats_jit(q_values, limit)
    cli.empty().print("solve for q(a|s)").empty()
    with cli.indent():
        cli.print("max	\t\t∫ u(s) ∫ q(a|s) Q(s,a) dads")
        cli.print("st.	\t\t∫ u(s) KL(q(a|s), π(a|s)) ds < ε")
        cli.print("    \t\t∫∫ u(s) q(a|s) dads = 1")
    cli.empty().print("statistics of problem").empty()
    with cli.indent():
        cli.print(f"Q(s,a) \t~ {float(mean):.5f} +- [{float(std):.5f}]")
        cli.print(f"ε \t\t= {eps}")

    start = jnp.asarray(eta0, dtype=jnp.float64).reshape(())
    eta, iters, dual = _newton_temperature(q_values, limit, start, eps, int(max_steps))
    with cli.indent():
        cli.empty().print(f"eta* = {float(eta):.6f} (dual = {float(dual):.6f}, iterations = {int(iters)})")
    return jnp.asarray([eta], dtype=jnp.float32)


def positive_temperature(raw: Float[Array, ""]) -> Float[Array, ""]:
    """Temperature ``softplus(raw)``, clipped to the range of the Newton solve."""
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


@eqx.filter_jit
def fit_temperature_sgd(
    q_values: Float[Array, "batch k"],
    raw: Float[Array, ""],
    opt_state: Any,
    tx: optax.GradientTransformation,
    eps: float,
    steps: int,
) -> tuple[Float[Array, ""], Any, Float[Array, ""]]:
    """Adam on the dual of one fixed batch. The minimum matches the Newton solve.

    Args:
        q_values: Q samples, shape ``[batch, k]``.
        raw: Initial unconstrained temperature.
        opt_state: Adam state for ``raw``.
        tx: Adam transform.
        eps: KL bound of the E-step.
        steps: Number of updates. Static.

    Returns:
        Final raw parameter, Adam state, and the positive temperature.
    """

    def body(
        carry: tuple[Float[Array, ""], Any], _step: Array
    ) -> tuple[tuple[Float[Array, ""], Any], Float[Array, ""]]:
        raw_, state = carry
        raw_, state, eta = descend_temperature(q_values, raw_, state, tx, eps)
        return (raw_, state), eta

    (raw, opt_state), etas = jax.lax.scan(body, (raw, opt_state), jnp.arange(steps))
    return raw, opt_state, etas[-1]
