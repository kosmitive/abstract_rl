import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient


def _solve(a, b, max_k=20, tol=1e-10):
    x = jax.jit(lambda bb: conjugate_gradient(lambda v: a @ v, bb, max_k, tol))(b)
    return x, jnp.linalg.norm(a @ x - b)


def test_identity():
    n = 3
    _, residual = _solve(jnp.eye(n), jnp.ones(n))
    assert residual < 1e-3


def test_random_spd_matrix():
    m = jax.random.normal(jax.random.key(0), (10, 10))
    a = m @ m.T + 10 * jnp.eye(10)
    b = jax.random.normal(jax.random.key(1), (10,))
    x, residual = _solve(a, b, max_k=50)
    assert residual < 1e-3
    np.testing.assert_allclose(x, jnp.linalg.solve(a, b), rtol=1e-3, atol=1e-4)


def test_stops_at_tolerance():
    # the exact solution of a diagonal system is found in at most n iterations, extra iterations must not change it
    a = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    b = jnp.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(_solve(a, b, max_k=3, tol=1e-12)[0], _solve(a, b, max_k=50, tol=1e-12)[0], atol=1e-6)
