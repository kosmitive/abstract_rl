import torch

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient


def test_mult_conjugate_gradients():
    n = 3
    A = torch.eye(n)
    b = torch.ones(n)
    _test_conjugate_gradients(A, b, 20, 1e-3)


def _test_conjugate_gradients(A, b, max_k=20, damping=1e-5):
    def fnA(v):
        return A @ v

    x = conjugate_gradient(fnA, b, max_k, damping)
    diff = torch.norm(fnA(x) - b)
    assert diff < 1e-3
