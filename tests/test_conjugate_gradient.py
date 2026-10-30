import mlx.core as mx
import numpy as np

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient


def _solve(a, b, max_k=20, tol=1e-10):
    x = conjugate_gradient(lambda v: a @ v, b, max_k, tol)
    return x, mx.linalg.norm(a @ x - b).item()


def test_identity():
    n = 3
    _, residual = _solve(mx.eye(n), mx.ones(n))
    assert residual < 1e-3


def test_random_spd_matrix():
    rng = np.random.default_rng(0)
    n = 10
    m = rng.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    b = rng.standard_normal(n)
    x, residual = _solve(mx.array(a, dtype=mx.float64), mx.array(b, dtype=mx.float64), max_k=50)
    assert residual < 1e-6
    np.testing.assert_allclose(np.array(x), np.linalg.solve(a, b), rtol=1e-6, atol=1e-8)


def test_converged_iterations_are_frozen():
    # the fixed length loop must not move the solution after the residual dropped below the tolerance
    a = mx.array(np.diag([1.0, 2.0, 3.0]), dtype=mx.float64)
    b = mx.array([1.0, 1.0, 1.0], dtype=mx.float64)
    x3, _ = _solve(a, b, max_k=3)
    x50, _ = _solve(a, b, max_k=50)
    np.testing.assert_allclose(np.array(x3), np.array(x50), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(x50), [1.0, 0.5, 1 / 3], atol=1e-10)


def test_compiled_matches_eager():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((6, 6)).astype(np.float32)
    a = mx.array(m @ m.T + 6 * np.eye(6, dtype=np.float32))
    b = mx.array(rng.standard_normal(6).astype(np.float32))
    eager = conjugate_gradient(lambda v: a @ v, b, 10, 1e-10)
    compiled = mx.compile(lambda a, b: conjugate_gradient(lambda v: a @ v, b, 10, 1e-10))(a, b)
    np.testing.assert_allclose(np.array(compiled), np.array(eager), rtol=1e-5, atol=1e-6)
