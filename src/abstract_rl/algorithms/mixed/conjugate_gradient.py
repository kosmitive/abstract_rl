from collections.abc import Callable

import mlx.core as mx


def conjugate_gradient(A: Callable, b: mx.array, max_k: int = 10, tol: float = 1e-10) -> mx.array:
    """
    Implementation of conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py.

    MLX has no while loop, so the loop is unrolled to exactly ``max_k`` iterations and can be traced by
    ``mx.compile``. Once the squared residual dropped to ``tol`` the iterate is frozen, which returns the same solution
    as stopping there, but the remaining matrix vector products are still evaluated.
    :param A: A function f : R[n] -> R[n], lambda to evaluate matrix vector product.
    :param b: The result of the vector multiplication.
    :param max_k: The (fixed) number of iterations.
    :param tol: Tolerance of the squared residual.
    :return: The found solution for the equation,
    """

    # start with residual
    x = mx.zeros_like(b)
    r = b
    p = b
    rdotr = mx.inner(r, r)
    active = mx.array(True)

    for _ in range(max_k):
        # remaining fields
        _Avp = A(p)
        alpha = rdotr / mx.inner(p, _Avp)

        # update x and r, a converged iteration keeps its values
        new_x = x + alpha * p
        new_r = r - alpha * _Avp
        new_rdotr = mx.inner(new_r, new_r)
        beta = new_rdotr / rdotr
        new_p = beta * p + new_r

        x = mx.where(active, new_x, x)
        r = mx.where(active, new_r, r)
        p = mx.where(active, new_p, p)
        rdotr = mx.where(active, new_rdotr, rdotr)
        active = mx.logical_and(active, rdotr > tol)

    return x
