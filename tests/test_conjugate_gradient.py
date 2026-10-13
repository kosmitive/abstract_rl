import jax
import jax.numpy as jnp

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient

jax.config.update("jax_enable_x64", True)


def _solve(a, b, max_k=20, tol=1e-10):
    x = conjugate_gradient(lambda v: a @ v, b, max_k, tol)
    return x, jnp.linalg.norm(a @ x - b)


def test_identity():
    n = 3
    _solution, residual = _solve(jnp.eye(n), jnp.ones(n))
    assert float(residual) < 1e-3


def test_random_spd_matrix():
    n = 10
    m = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.float64)
    a = m @ m.T + n * jnp.eye(n, dtype=jnp.float64)
    b = jax.random.normal(jax.random.PRNGKey(1), (n,), dtype=jnp.float64)
    x, residual = _solve(a, b, max_k=50, tol=1e-24)
    assert float(residual) < 1e-8
    assert jnp.allclose(x, jnp.linalg.solve(a, b), atol=1e-6)
