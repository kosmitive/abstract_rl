"""Dual of the non-parametric MPO E-step."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
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


_dual_and_grad = jax.jit(jax.value_and_grad(temperature_dual, argnums=1))


def find_optimal_temperature(
    q_values: Float[Array, "batch k"] | np.ndarray,
    eps: float,
    eta0: float = 1.0,
    lr: float | None = None,
    max_steps: int = 200,
) -> Float[Array, " 1"]:
    """Minimise the convex temperature dual with L-BFGS-B.

    Args:
        q_values: Q samples, shape ``[batch, k]``.
        eps: KL bound.
        eta0: Initial temperature.
        lr: Unused. Kept so older call sites still pass a learning rate.
        max_steps: Iteration cap.

    Returns:
        Optimal temperature, shape ``[1]``, float32.
    """
    del lr
    cli = CliPrinter().instance
    q_values = jnp.asarray(q_values, dtype=jnp.float64)
    cli.empty().print("solve for q(a|s)").empty()
    with cli.indent():
        cli.print("max	\t\t∫ u(s) ∫ q(a|s) Q(s,a) dads")
        cli.print("st.	\t\t∫ u(s) KL(q(a|s), π(a|s)) ds < ε")
        cli.print("    \t\t∫∫ u(s) q(a|s) dads = 1")
    cli.empty().print("statistics of problem").empty()
    with cli.indent():
        cli.print(f"Q(s,a) \t~ {float(jnp.mean(q_values)):.5f} +- [{float(jnp.std(q_values)):.5f}]")
        cli.print(f"ε \t\t= {eps}")

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _dual_and_grad(q_values, jnp.float64(x[0]), eps)
        return float(value), np.asarray([float(grad)])

    start = float(np.clip(np.asarray(eta0).reshape(-1)[0], 1e-6, None))
    result = scipy.optimize.minimize(
        objective,
        np.asarray([start]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-6, None)],
        options={"maxiter": max_steps},
    )
    eta = float(result.x[0])
    with cli.indent():
        cli.empty().print(f"eta* = {eta:.6f} (dual = {result.fun:.6f}, iterations = {result.nit})")
    return jnp.asarray([eta], dtype=jnp.float32)
