import torch

from abstract_rl.src.misc.cli_printer import CliPrinter


def to_col(x):

    # make column vector
    n = x.shape[0]
    if len(x.shape) == 1: x = x.view([-1, 1])
    assert x.shape[0] == n and x.shape[1] == 1
    return x


def conjugate_gradient(A, b, max_k=10, tol=1e-10):
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
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    k = 0
    rdotr = torch.dot(r, r)
    cli = CliPrinter().instance

    with cli.header("conjugate gradients started", False):

        while k < max_k:

            # remaining fields
            _Avp = A(p)
            alpha = rdotr / torch.dot(p, _Avp)

            # update x and r
            x += alpha * p
            r -= alpha * _Avp

            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = beta * p + r
            rdotr = new_rdotr
            if rdotr <= tol: break

            # calculate next conjugate gradient
            k += 1

    return x
