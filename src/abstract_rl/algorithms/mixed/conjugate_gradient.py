"""Conjugate gradient for the Fisher-vector product inside TRPO."""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from abstract_rl.misc.cli_printer import CliPrinter


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
    solution = jnp.zeros_like(vector)
    residual = vector
    direction = vector
    residual_norm = jnp.dot(residual, residual)
    cli = CliPrinter().instance
    k = 0
    with cli.header("conjugate gradients started", False):
        while k < max_k:
            product = matrix_vector(direction)
            alpha = residual_norm / jnp.dot(direction, product)
            solution = solution + alpha * direction
            residual = residual - alpha * product
            next_norm = jnp.dot(residual, residual)
            beta = next_norm / residual_norm
            direction = beta * direction + residual
            residual_norm = next_norm
            if float(residual_norm) <= tol:
                break
            k += 1
    return solution
