"""Unrolled critic Adam versus the previous one-step scan.

The index stream is unchanged: each step still splits the key and draws one minibatch with
``randint``. Unrolling changes how XLA fuses that body, so the comparison is float32 absolute
tolerance on the per-step losses and on the weights, not bitwise equality.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.value_functions.fc_q_networks import _critic_scan, q_forward

jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def _critic_scan_unroll1(
    data: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    diff: object,
    static: object,
    opt_state: object,
    tx: object,
    num_steps: int,
    batch_size: int,
) -> tuple[object, object, jax.Array, jax.Array]:
    """The critic scan before unrolling. Same loss, same split, same Adam."""
    states, actions, targets, key, size = data

    def loss_on(weights: object, idx: jax.Array) -> jax.Array:
        network = eqx.combine(weights, static)
        prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
        return jnp.mean(jnp.square(prediction - targets[idx]))

    def body(
        carry: tuple[object, object, jax.Array], step: jax.Array
    ) -> tuple[tuple[object, object, jax.Array], tuple[jax.Array, jax.Array]]:
        weights, state, key_ = carry
        key_, sample_key = jax.random.split(key_)
        idx = jax.random.randint(sample_key, (batch_size,), 0, size)
        pre, grads = eqx.filter_value_and_grad(lambda w: loss_on(w, idx))(weights)
        updates, state = tx.update(grads, state, weights)
        weights = eqx.apply_updates(weights, updates)
        post = jax.lax.cond(step % 10 == 0, lambda w: loss_on(w, idx), lambda _w: pre, weights)
        return (weights, state, key_), (pre, post)

    (diff, opt_state, _), (pre, post) = jax.lax.scan(body, (diff, opt_state, key), jnp.arange(num_steps))
    return diff, opt_state, pre, post


def _flat(tree: object) -> np.ndarray:
    pieces = [np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]
    return np.concatenate(pieces)


def test_unrolled_critic_matches_per_step() -> None:
    """Same key and data. 36 steps leaves a remainder under an unroll of 8."""
    key = jax.random.PRNGKey(11)
    key, k_net, k_s, k_a, k_t, k_fit = jax.random.split(key, 6)
    net = MultilayerPerceptron(k_net, [4, 256, 256, 1], ["elu", "elu", "none"], False, "xavier_gaussian")
    diff, static = eqx.partition(net, eqx.is_inexact_array)
    tx = optax.adam(1e-3)
    opt_state = tx.init(diff)
    capacity = 512
    states = jax.random.normal(k_s, (capacity, 3), dtype=jnp.float32)
    actions = jax.random.normal(k_a, (capacity, 1), dtype=jnp.float32)
    targets = jax.random.normal(k_t, (capacity,), dtype=jnp.float32)
    data = (states, actions, targets, k_fit, jnp.int32(400))
    steps, batch = 36, 64
    fresh = (
        jax.tree.map(jnp.array, diff),
        jax.tree.map(jnp.array, opt_state),
    )
    reference = _critic_scan_unroll1(data, fresh[0], static, fresh[1], tx, steps, batch)
    fresh = (
        jax.tree.map(jnp.array, diff),
        jax.tree.map(jnp.array, opt_state),
    )
    updated = _critic_scan(data, fresh[0], static, fresh[1], tx, steps, batch)
    params = float(np.max(np.abs(_flat(updated[0]) - _flat(reference[0]))))
    pre = float(np.max(np.abs(np.asarray(updated[2]) - np.asarray(reference[2]))))
    post = float(np.max(np.abs(np.asarray(updated[3]) - np.asarray(reference[3]))))
    assert params <= 1e-5, params
    assert pre <= 1e-6, pre
    assert post <= 1e-6, post
    assert np.asarray(updated[2]).shape == (steps,)
