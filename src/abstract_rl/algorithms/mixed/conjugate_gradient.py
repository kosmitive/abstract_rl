"""Conjugate gradient for the Fisher-vector product inside TRPO."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def cg_solve(
    matrix_vector: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    vector: Float[Array, " n"],
    max_k: Array,
    tol: Float[Array, ""],
) -> Float[Array, " n"]:
    """Conjugate gradients as a ``while_loop``, so a jitted caller compiles it once.

    Stop after the squared residual drops to ``tol``, and always take at least one step when
    ``max_k`` is positive.
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
