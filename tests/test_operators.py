import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    find_optimal_temperature,
    temperature_dual,
)
from abstract_rl.operator.np_rew_operator import e_step_log_weights


def test_temperature_satisfies_kl_bound():
    q = jax.random.normal(jax.random.PRNGKey(0), (256, 20)) * 10
    eps = 0.1
    eta = float(find_optimal_temperature(q, eps)[0])

    weights = jax.nn.softmax(q / eta, axis=1)
    kl = float(jnp.mean(jnp.sum(weights * jnp.log(weights * q.shape[1] + 1e-12), axis=1)))
    assert eta > 0
    assert np.isclose(kl, eps, rtol=0.1, atol=0.01)


def test_temperature_dual_of_a_flat_q_is_eta_times_epsilon():
    q = jnp.zeros((4, 3))
    value = float(temperature_dual(q, jnp.float64(2.0), 0.1))
    assert np.isclose(value, 0.2, atol=1e-6)


def test_e_step_weights_are_a_per_state_log_softmax():
    q = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    eta = 0.5
    weights = e_step_log_weights(q, eta)
    expected = jax.nn.log_softmax(q / eta, axis=-1)
    assert np.allclose(np.asarray(weights), np.asarray(expected), atol=1e-6)
    assert np.allclose(np.exp(np.asarray(weights[1])), np.full(3, 1.0 / 3.0), atol=1e-6)
    assert np.allclose(np.exp(np.asarray(weights)).sum(axis=1), 1.0, atol=1e-6)
