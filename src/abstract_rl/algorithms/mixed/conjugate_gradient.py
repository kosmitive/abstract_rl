"""Conjugate gradient for the Fisher-vector product inside TRPO."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from abstract_rl.misc.cli_printer import CliPrinter


def cg_solve(
    matrix_vector: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    vector: Float[Array, " n"],
    max_k: Array,
    tol: Float[Array, ""],
) -> Float[Array, " n"]:
    """Conjugate gradients as a ``while_loop``, so a jitted caller compiles it once.

    The iteration matches ``conjugate_gradient``: stop after the squared residual drops to
    ``tol``, and always take at least one step when ``max_k`` is positive.
    """

    def cond(state: tuple[Array, Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Array, Array]) -> Array:
        step, _solution, _residual, _direction, _residual_norm, stop = state
        return (step < max_k) & (~stop)

    def body(
        state: tuple[Array, Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Array, Array],
    ) -> tuple[Array, Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Array, Array]:
        step, solution, residual, direction, residual_norm, _stop = state
        product = matrix_vector(direction)
        alpha = residual_norm / jnp.dot(direction, product)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_norm = jnp.dot(residual, residual)
        beta = next_norm / residual_norm
        direction = beta * direction + residual
        return step + jnp.int32(1), solution, residual, direction, next_norm, next_norm <= tol

    _step, solution, _residual, _direction, _norm, _stop = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.int32(0),
            jnp.zeros_like(vector),
            vector,
            vector,
            jnp.dot(vector, vector),
            jnp.bool_(False),
        ),
    )
    return solution


def conjugate_gradient(
    matrix_vector: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    vector: Float[Array, " n"],
    max_k: int = 10,
    tol: float = 1e-10,
) -> Float[Array, " n"]:
    """Solve ``A x = b`` with conjugate gradients.

    Args:
        matrix_vector: Computes the matrix-vector product ``A p``.
        vector: Right hand side ``b``.
        max_k: Iteration cap.
        tol: Stop when the residual squared norm falls to this value.

    Returns:
        The approximate solution.
    """
    cli = CliPrinter().instance
    with cli.header("conjugate gradients started", False):
        return cg_solve(matrix_vector, vector, jnp.int32(max_k), jnp.asarray(tol, dtype=vector.dtype))
