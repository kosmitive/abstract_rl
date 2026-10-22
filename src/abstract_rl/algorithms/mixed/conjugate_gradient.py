from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def conjugate_gradient(
    A: Callable[[Float[Array, " n"]], Float[Array, " n"]], b: Float[Array, " n"], max_k: int = 10, tol: float = 1e-10
) -> Float[Array, " n"]:
    """
    Implementation of conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py.
    :param A: A function f : R[n] -> R[n], lambda to evaluate matrix vector product.
    :param b: The result of the vector multiplication.
    :param max_k: The maximum number of iterations.
    :param damping: Damping for vector products.
    :return: The found solution for the equation,
    """

    # start with residual
    x = jnp.zeros_like(b)
    r = b
    p = b

    k = 0
    rdotr = jnp.dot(r, r)

    Carry = tuple[
        Int[Array, ""], Float[Array, " n"], Float[Array, " n"], Float[Array, " n"], Float[Array, ""], Bool[Array, ""]
    ]

    def cond(carry: Carry) -> Bool[Array, ""]:
        k, _, _, _, _, converged = carry
        return (k < max_k) & ~converged

    def body(carry: Carry) -> Carry:
        k, x, r, p, rdotr, _ = carry

        # remaining fields
        _Avp = A(p)
        alpha = rdotr / jnp.dot(p, _Avp)

        # update x and r
        x = x + alpha * p
        r = r - alpha * _Avp

        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / rdotr
        p = beta * p + r
        rdotr = new_rdotr

        # calculate next conjugate gradient, the torch loop breaks once rdotr <= tol
        return k + 1, x, r, p, rdotr, rdotr <= tol

    init: Carry = (jnp.asarray(k), x, r, p, rdotr, jnp.asarray(False))
    _, x, _, _, _, _ = jax.lax.while_loop(cond, body, init)
    return x
