import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature


def test_temperature_satisfies_kl_bound():
    q = jax.random.normal(jax.random.key(0), (256, 20)) * 10
    eps = 0.1
    eta = float(jax.jit(lambda qq: find_optimal_temperature(qq, eps))(q)[0])

    # the KL between the reweighted and the uniform (sampling) distribution matches the bound
    w = jax.nn.softmax(q / eta, axis=1)
    kl = float(jnp.mean(jnp.sum(w * jnp.log(w * q.shape[1] + 1e-12), axis=1)))
    assert eta > 0
    assert np.isclose(kl, eps, rtol=0.1, atol=0.01)


def test_masked_rows_are_ignored():
    q = jax.random.normal(jax.random.key(1), (64, 20)) * 5
    padded = jnp.concatenate([q, jnp.full((16, 20), 1e6)])
    mask = jnp.arange(80) < 64
    np.testing.assert_allclose(
        find_optimal_temperature(padded, 0.1, mask=mask), find_optimal_temperature(q, 0.1), rtol=1e-5
    )
