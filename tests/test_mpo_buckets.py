"""Full-allocation MPO kernels versus the power-of-two prefix.

The E-step is not restricted to the M-step sample. These tests compare the prefix kernels with the
same full-allocation scan. Chunk ``i`` uses ``split(key, n_full)[i]`` in both, which is the key
the full scan already used for that chunk. Retrace splits its key once per leading chunk in the
same order. Resampling, target statistics and Retrace are bitwise on this state. Ranking is
the exception: batched ``log_softmax`` depends on the row count, so a shorter prefix differs by
one float32 ulp, and that ulp is what the M-step tolerance covers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    _mpo_scan,
    _target_stats_all,
)
from abstract_rl.misc import buckets
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.operator.np_rew_operator import _ranking_full
from abstract_rl.operator.resample_actions_operator import _CHUNK as RESAMPLE_CHUNK
from abstract_rl.operator.resample_actions_operator import resample_buffer
from abstract_rl.operator.retrace_operator import _CHUNK as RETRACE_CHUNK
from abstract_rl.operator.retrace_operator import _retrace_slots, _slot_count
from abstract_rl.policy.continuous.gaussian_math import GaussianParams
from abstract_rl.value_functions.fc_q_networks import _critic_scan, q_forward

_STATS_CHUNK = 1024


def test_active_width_is_a_power_of_two_until_the_allocation():
    capacity = 50_000
    seen: list[int] = []
    previous = -1
    for size in range(1, capacity + 1):
        width = buckets.active_width(size, capacity)
        assert size <= width <= capacity
        if width != previous:
            seen.append(width)
            previous = width
    assert seen[-1] == capacity
    for width in seen[:-1]:
        assert width > 0 and width & (width - 1) == 0
    # Filling by 1000 does not mint a new compile at every exact size.
    widths = {buckets.active_width(1000 * epoch, capacity) for epoch in range(1, 11)}
    assert len(widths) < 10


def _policy(key: jax.Array) -> GaussianParams:
    network = MultilayerPerceptron(key, [3, 8, 1], ["tanh", "none"], False, "xavier_gaussian")
    return GaussianParams(
        network=network,
        log_sigma=jnp.zeros((1,), dtype=jnp.float32),
        cov_type="single",
        squash=True,
        sigma_is_softplus=False,
        action_center=(0.0,),
        action_scale=(2.0,),
        fixed_log_sigma=None,
    )


def _q_net(key: jax.Array) -> MultilayerPerceptron:
    return MultilayerPerceptron(key, [4, 8, 1], ["tanh", "none"], False, "xavier_gaussian")


def _match(name: str, left: jax.Array, right: jax.Array, atol: float = 0.0) -> None:
    """Bitwise by default. ``atol`` covers a known float32 ulp, not a looser algorithm."""
    got = np.asarray(left)
    exp = np.asarray(right)
    assert got.shape == exp.shape
    if np.array_equal(got, exp):
        return
    abs_err = float(np.max(np.abs(got - exp)))
    assert abs_err <= atol, f"{name} abs {abs_err}"


def test_prefix_kernels_match_the_full_allocation():
    capacity = 2048
    size = 300
    n_actions = 4
    pad = 20
    n_episodes = size // pad
    key = jax.random.PRNGKey(7)
    key, k_policy, k_q, k_state, k_resample, k_retrace, k_mstep, k_critic = jax.random.split(key, 8)
    policy = _policy(k_policy)
    q_net = _q_net(k_q)
    states = jax.random.normal(k_state, (capacity, 3), dtype=jnp.float32)
    actions = jax.random.normal(jax.random.PRNGKey(1), (capacity, 1), dtype=jnp.float32)
    rewards = jax.random.normal(jax.random.PRNGKey(2), (capacity,), dtype=jnp.float32)
    next_states = states + 0.01
    dones = jnp.zeros((capacity,), dtype=jnp.float32).at[pad - 1 : size : pad].set(1.0)
    log_probs = jax.random.normal(jax.random.PRNGKey(3), (capacity,), dtype=jnp.float32)
    add_act = jnp.zeros((capacity, n_actions, 1), dtype=jnp.float32)
    add_q = jnp.zeros((capacity, n_actions), dtype=jnp.float32)
    add_pi = jnp.zeros((capacity, n_actions), dtype=jnp.float32)
    kept = (policy, q_net, states, k_resample, n_actions, jnp.int32(size))
    n_full = (capacity + RESAMPLE_CHUNK - 1) // RESAMPLE_CHUNK
    n_prefix = buckets.chunk_count(size, capacity, RESAMPLE_CHUNK)
    assert n_prefix < n_full
    full_a, full_q, full_pi = resample_buffer(kept, add_act, add_q, add_pi, n_full)
    pre_a, pre_q, pre_pi = resample_buffer(kept, add_act, add_q, add_pi, n_prefix)
    _match("add_act", pre_a[:size], full_a[:size])
    _match("add_q", pre_q[:size], full_q[:size])
    _match("add_pi", pre_pi[:size], full_pi[:size])
    assert np.all(np.asarray(pre_q[size:]) == 0)
    assert np.all(np.asarray(full_q[size:]) == 0)

    width = buckets.active_width(size, capacity)
    eta = jnp.float32(1.0)
    rank_full = _ranking_full(full_q, jnp.int32(size), eta, capacity)
    rank_prefix = _ranking_full(full_q, jnp.int32(size), eta, width)
    # log_softmax on a shorter batch differs by one float32 ulp from the full-buffer kernel.
    _match("ranking", rank_prefix[:size], rank_full[:size], atol=5e-7)

    n_stats_full = (capacity + _STATS_CHUNK - 1) // _STATS_CHUNK
    n_stats = buckets.chunk_count(size, capacity, _STATS_CHUNK)
    stats_full = _target_stats_all(policy, states, jnp.int32(size), n_stats_full)
    stats_prefix = _target_stats_all(policy, states, jnp.int32(size), n_stats)
    _match("target_stats", stats_prefix[:size], stats_full[:size])

    starts = np.zeros((n_episodes,), dtype=np.int32)
    lengths = np.full((n_episodes,), pad, dtype=np.int32)
    for index in range(n_episodes):
        starts[index] = index * pad
    full_slots = _slot_count(capacity, pad)
    slot_n = buckets.slot_width(n_episodes, full_slots, RETRACE_CHUNK)
    assert slot_n < full_slots
    starts_full = np.zeros((full_slots,), dtype=np.int32)
    lengths_full = np.zeros((full_slots,), dtype=np.int32)
    starts_full[:n_episodes] = starts
    lengths_full[:n_episodes] = lengths
    def _zeros() -> tuple[jax.Array, jax.Array]:
        # ``_retrace_slots`` donates these buffers, so each call needs its own.
        return jnp.zeros((capacity,), dtype=jnp.float32), jnp.zeros((capacity,), dtype=jnp.float32)
    common = (
        q_net,
        policy,
        states,
        actions,
        rewards,
        next_states,
        dones,
        log_probs,
    )
    tail = (
        k_retrace,
        jnp.float32(0.99),
        jnp.float32(1.0),
        5,
        4,
        pad,
        jnp.bool_(True),
    )
    q_full, tq_full = _retrace_slots(
        (*common, jnp.asarray(starts_full), jnp.asarray(lengths_full), *tail), *_zeros()
    )
    q_pre, tq_pre = _retrace_slots(
        (*common, jnp.asarray(starts_full[:slot_n]), jnp.asarray(lengths_full[:slot_n]), *tail),
        *_zeros(),
    )
    _match("retrace q", q_pre[:size], q_full[:size])
    _match("retrace tq", tq_pre[:size], tq_full[:size])

    tx_outer = optax.adam(1e-2)
    tx_inner = optax.adam(1e-2)

    def _m_args() -> tuple[object, ...]:
        # The scan donates the policy leaves, both Adam states and the temperature state.
        diff, static = eqx.partition(policy, eqx.is_inexact_array)
        diff = jax.tree.map(jnp.array, diff)
        multiplier = jnp.ones((2,), dtype=jnp.float32)
        eta_raw = jnp.zeros((), dtype=jnp.float32)
        tx_eta = optax.adam(1e-3)
        return (
            diff,
            static,
            tx_outer.init(diff),
            multiplier,
            tx_inner.init(multiplier),
            eta_raw,
            tx_eta.init(eta_raw),
            tx_outer,
            tx_inner,
            tx_eta,
            4,
            32,
            jnp.asarray([0.1, 1.0e-4], dtype=jnp.float32),
            0.0,
            0.1,
        )

    m_full = _mpo_scan(
        (states, full_a, stats_full, full_q, k_mstep, jnp.int32(size)),
        *_m_args(),
    )
    m_prefix = _mpo_scan(
        (states, pre_a, stats_prefix, pre_q, k_mstep, jnp.int32(size)),
        *_m_args(),
    )
    flat_prefix = jax.flatten_util.ravel_pytree(m_prefix[0])[0]
    flat_full = jax.flatten_util.ravel_pytree(m_full[0])[0]
    _match("m-step params", flat_prefix, flat_full, atol=1e-6)
    _match("m-step objective", m_prefix[6], m_full[6], atol=1e-6)

    tx = optax.adam(1e-3)

    def _c_args() -> tuple[object, ...]:
        cdiff, cstatic = eqx.partition(q_net, eqx.is_inexact_array)
        cdiff = jax.tree.map(jnp.array, cdiff)
        return cdiff, cstatic, tx.init(cdiff), tx, 3, 16

    c_full = _critic_scan((states, actions, tq_full, k_critic, jnp.int32(size)), *_c_args())
    c_prefix = _critic_scan((states, actions, tq_pre, k_critic, jnp.int32(size)), *_c_args())
    cstatic = eqx.partition(q_net, eqx.is_inexact_array)[1]
    flat_c_prefix = jax.flatten_util.ravel_pytree(c_prefix[0])[0]
    flat_c_full = jax.flatten_util.ravel_pytree(c_full[0])[0]
    _match("critic params", flat_c_prefix, flat_c_full, atol=1e-6)
    _match("critic loss", c_prefix[2], c_full[2], atol=1e-6)
    probe = jnp.arange(16)
    full_prediction = q_forward(eqx.combine(c_full[0], cstatic), states[probe], actions[probe])
    loss_full = jnp.mean(jnp.square(full_prediction - tq_full[probe]))
    loss_prefix = jnp.mean(
        jnp.square(q_forward(eqx.combine(c_prefix[0], cstatic), states[probe], actions[probe]) - tq_pre[probe])
    )
    _match("critic probe", loss_prefix, loss_full, atol=1e-6)
