import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    descend_temperature,
    temperature_dual,
    temperature_raw,
)
from abstract_rl.operator.np_rew_operator import e_step_log_weights


def _fit_eta(q, eps, steps=4000, lr=0.05):
    """Adam on the dual. A wide Q needs a few thousand steps to sit on the KL bound."""
    raw = temperature_raw(1.0)
    tx = optax.adam(lr)

    @eqx.filter_jit
    def fit(raw_, state, n_steps: int):
        def body(carry, _step):
            current, opt_state = carry
            current, opt_state, eta = descend_temperature(q, current, opt_state, tx, eps)
            return (current, opt_state), eta

        (_carry, etas) = jax.lax.scan(body, (raw_, state), jnp.arange(n_steps))
        return etas[-1]

    return fit(raw, tx.init(raw), steps)


def test_temperature_satisfies_kl_bound():
    q = jax.random.normal(jax.random.PRNGKey(0), (256, 20), dtype=jnp.float32) * 10
    eps = 0.1
    eta = _fit_eta(q, eps)

    weights = jax.nn.softmax(q / eta, axis=1)
    kl = float(jnp.mean(jnp.sum(weights * jnp.log(weights * q.shape[1] + 1e-12), axis=1)))
    assert float(eta) > 0
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
