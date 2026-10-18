import torch

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient


def _solve(a, b, max_k=20, tol=1e-10):
    x = conjugate_gradient(lambda v: a @ v, b, max_k, tol)
    return x, torch.norm(a @ x - b)


def test_identity():
    n = 3
    _, residual = _solve(torch.eye(n), torch.ones(n))
    assert residual < 1e-3


def test_random_spd_matrix():
    torch.manual_seed(0)
    n = 10
    m = torch.randn(n, n, dtype=torch.float64)
    a = m @ m.T + n * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    x, residual = _solve(a, b, max_k=50)
    assert residual < 1e-6
    torch.testing.assert_close(x, torch.linalg.solve(a, b))
