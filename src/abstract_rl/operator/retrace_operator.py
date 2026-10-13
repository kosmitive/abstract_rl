"""Retrace(λ) action-value targets.

The return at time t is

    Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{k=0}^{B-1} γ^k (prod_{i=1}^{k} c_{t+i}) δ_{t+k}

with an empty product equal to 1 and ``c_s = λ min(1, π(a_s|s_s) / μ(a_s|s_s))``.
``c_t`` of the current step is not part of that product; the trace starts at the next action.
"""

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.replay_buffer import ReplayBuffer
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, log_prob
from abstract_rl.value_functions.fc_q_networks import estimate_v, q_forward

_CHUNK = 8
_V_SAMPLES = 10


class RetraceOperator(TrajectoryOperator):
    """Batched Retrace over the replay, using the current policy and the target Q."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        self.conf = mc.get("conf")
        self.mc = mc
        self.policy = mc.get("policy", False)
        self.q_network = mc.get("q_network", True)
        self.discount = float(mc["env"].discount())
        self.ret_lambda = float(self.conf["lambda"])
        self.bootstrap = int(self.conf.get_root("mem_bootstrap_steps"))

    def __repr__(self) -> str:
        return "retrace"

    def transform(self, trajectory: Trajectory) -> None:
        """Retrace targets for one trajectory, stored as ``q`` and ``tq``."""
        self.transform_all([trajectory])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """Write ``q`` and ``tq``. A device-backed collection is scored in episode chunks."""
        if isinstance(trajectories, TrajectoryCollection) and trajectories.device is not None:
            self.apply(trajectories.device, self.mc.rng.split())
            trajectories.scatter_field("q", trajectories.device.q)
            trajectories.scatter_field("tq", trajectories.device.tq)
            return
        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()
        for trajectory in trajectories:
            self._transform_one(trajectory)

    def _transform_one(self, trajectory: Trajectory) -> None:
        """Score a single episode with the same chunk kernel the buffer uses."""
        length = len(trajectory)
        replay = ReplayBuffer(
            trajectory.states.shape[-1],
            np.asarray(trajectory.actions).reshape(length, -1).shape[-1],
            length,
            1,
            length,
        )
        replay.add_trajectories([trajectory])
        self.apply(replay, self.mc.rng.split())
        trajectory["q"] = np.asarray(replay.q[:length])
        trajectory["tq"] = np.asarray(replay.tq[:length])

    def apply(self, replay: ReplayBuffer, key: PRNGKeyArray) -> None:
        """Write ``q`` and ``tq`` for every stored transition.

        Episodes are gathered on device into a slot count fixed by the buffer capacity, so the
        kernel shape does not follow the number of stored episodes. Empty slots are skipped.

        Args:
            replay: Replay whose behaviour log probabilities are the denominator of the trace.
            key: Random key for the action samples inside the value estimate.
        """
        if replay.size == 0 or not replay.episodes:
            return
        pad = replay.pad_limit
        slots = _slot_count(replay.capacity, pad)
        q = replay.q
        tq = replay.tq
        reset = True
        for group_start in range(0, len(replay.episodes), slots):
            group = replay.episodes[group_start : group_start + slots]
            starts = np.zeros((slots,), dtype=np.int32)
            lengths = np.zeros((slots,), dtype=np.int32)
            for index, (episode_start, length) in enumerate(group):
                starts[index] = int(episode_start)
                lengths[index] = min(int(length), pad)
            key, sub = jax.random.split(key)
            q, tq = _retrace_slots(
                (
                    self.q_network.params,
                    self.policy.params,
                    replay.states,
                    replay.actions,
                    replay.rewards,
                    replay.next_states,
                    replay.dones,
                    replay.log_probs,
                    jnp.asarray(starts),
                    jnp.asarray(lengths),
                    sub,
                    jnp.float32(self.discount),
                    jnp.float32(self.ret_lambda),
                    self.bootstrap,
                    _V_SAMPLES,
                    pad,
                    jnp.bool_(reset),
                ),
                q,
                tq,
            )
            reset = False
        replay.q = q
        replay.tq = tq


def retrace_accumulate(
    q: Float[Array, "episodes time"],
    td: Float[Array, "episodes time"],
    coefficients: Float[Array, "episodes time"],
    discount: float | Float[Array, ""],
    bootstrap: int,
    lengths: Array,
) -> Float[Array, "episodes time"]:
    """Retrace backup given Q, TD errors and clipped trace coefficients.

    Args:
        q: Current Q values.
        td: TD errors. Terminal bootstraps are already zero.
        coefficients: ``c_t = λ min(1, π/μ)``.
        discount: Discount factor.
        bootstrap: How many TD errors to include, counting the current one.
        lengths: Real length of each episode.

    Returns:
        Retrace targets, zero past ``lengths``.
    """
    time = q.shape[1]
    valid_now = jnp.arange(time)[None, :] < lengths[:, None]
    targets = jnp.where(valid_now, q + td, 0.0)
    coeff = jnp.ones_like(q)
    # ``bootstrap`` is a static integer. One scan step is the old ``k`` iteration: the trace
    # coefficient at t multiplies ``c_{t+k}`` and the TD error at ``t+k``.
    if bootstrap > 1:

        def step(
            carry: tuple[Float[Array, "episodes time"], Float[Array, "episodes time"]], k: Array
        ) -> tuple[tuple[Float[Array, "episodes time"], Float[Array, "episodes time"]], None]:
            targets_k, coeff_k = carry
            offset = jnp.arange(time) + k
            in_range = offset < time
            source = jnp.minimum(offset, time - 1)
            valid = in_range[None, :] & (offset[None, :] < lengths[:, None])
            updated = jnp.where(valid, coeff_k * discount * coefficients[:, source], 0.0)
            targets_k = targets_k + jnp.where(valid, updated * td[:, source], 0.0)
            coeff_k = jnp.where(in_range[None, :], updated, coeff_k)
            return (targets_k, coeff_k), None

        (targets, _coeff), _ = jax.lax.scan(step, (targets, coeff), jnp.arange(1, bootstrap))
    return targets


def _slot_count(capacity: int, pad_limit: int) -> int:
    """Episode slots for one compiled call. The count depends on capacity, not on the fill level."""
    guess = math.ceil(capacity / max(pad_limit, 1)) + _CHUNK
    guess = max(guess, _CHUNK)
    remainder = guess % _CHUNK
    if remainder:
        guess += _CHUNK - remainder
    return guess


def _retrace_chunk(
    q_net: MultilayerPerceptron,
    policy: GaussianParams,
    states: Float[Array, "chunk time obs"],
    actions: Float[Array, "chunk time act"],
    rewards: Float[Array, "chunk time"],
    next_states: Float[Array, "chunk time obs"],
    dones: Float[Array, "chunk time"],
    behaviour_log_prob: Float[Array, "chunk time"],
    lengths: Array,
    key: PRNGKeyArray,
    discount: Float[Array, ""],
    trace_lambda: Float[Array, ""],
    bootstrap: int,
    v_samples: int,
) -> tuple[Float[Array, "chunk time"], Float[Array, "chunk time"]]:
    episodes, time, _obs = states.shape
    flat = episodes * time
    flat_states = states.reshape((flat, states.shape[-1]))
    flat_actions = actions.reshape((flat, actions.shape[-1]))
    q = q_forward(q_net, flat_states, flat_actions).reshape((episodes, time))
    next_v = estimate_v(q_net, policy, next_states.reshape((flat, states.shape[-1])), key, v_samples)
    next_v = next_v.reshape((episodes, time)) * (1.0 - dones)
    pi_log_prob = log_prob(policy, flat_actions, states=flat_states).reshape((episodes, time))
    coefficients = jnp.minimum(jnp.exp(pi_log_prob - behaviour_log_prob), 1.0) * trace_lambda
    td = rewards + discount * next_v - q
    return q, retrace_accumulate(q, td, coefficients, discount, bootstrap, lengths)


def _scatter_add(
    flat: Float[Array, " cap"],
    rows: Float[Array, "slots time"],
    starts: Array,
    lengths: Array,
) -> Float[Array, " cap"]:
    """Add each episode row into the flat buffer. Padded steps contribute nothing."""
    time = rows.shape[1]
    offset = jnp.arange(time)
    index = starts[:, None] + offset[None, :]
    valid = (offset[None, :] < lengths[:, None]) & (index < flat.shape[0])
    safe = jnp.where(valid, index, 0)
    contrib = jnp.where(valid, rows, 0)
    return flat.at[safe.reshape(-1)].add(contrib.reshape(-1))


@eqx.filter_jit(donate="all-except-first")
def _retrace_slots(
    kept: tuple[Any, ...],
    q_buffer: Float[Array, " cap"],
    tq_buffer: Float[Array, " cap"],
) -> tuple[Float[Array, " cap"], Float[Array, " cap"]]:
    """Retrace for a fixed number of episode slots, scattered into the flat buffers.

    The first argument is kept (replay contents and the networks). ``q_buffer`` and ``tq_buffer``
    are donated and replaced. ``pad`` is the constructor time limit. Empty slots skip the Q network.
    A true ``reset`` flag clears the buffers before the scatter, so a second group can add to them.
    """
    (
        q_net,
        policy,
        states,
        actions,
        rewards,
        next_states,
        dones,
        behaviour_log_prob,
        starts,
        lengths,
        key,
        discount,
        trace_lambda,
        bootstrap,
        v_samples,
        pad,
        reset,
    ) = kept
    n_slots = starts.shape[0]
    n_chunks = n_slots // _CHUNK
    capacity = states.shape[0]
    time = jnp.arange(pad)

    def body(key_: PRNGKeyArray, index: Array) -> tuple[PRNGKeyArray, tuple[Array, Array]]:
        key_, sub = jax.random.split(key_)
        slot = jax.lax.dynamic_slice(starts, (index * _CHUNK,), (_CHUNK,))
        lens = jax.lax.dynamic_slice(lengths, (index * _CHUNK,), (_CHUNK,))

        def run(sub_key: PRNGKeyArray) -> tuple[Array, Array]:
            index_t = jnp.clip(slot[:, None] + time[None, :], 0, capacity - 1)
            mask = time[None, :] < lens[:, None]
            q_hat, tq_hat = _retrace_chunk(
                q_net,
                policy,
                states[index_t],
                actions[index_t],
                jnp.where(mask, rewards[index_t], 0),
                next_states[index_t],
                jnp.where(mask, dones[index_t], 1),
                jnp.where(mask, behaviour_log_prob[index_t], 0),
                lens,
                sub_key,
                discount,
                trace_lambda,
                bootstrap,
                v_samples,
            )
            return jnp.where(mask, q_hat, 0), jnp.where(mask, tq_hat, 0)

        def skip(_sub_key: PRNGKeyArray) -> tuple[Array, Array]:
            zeros = jnp.zeros((_CHUNK, pad), dtype=jnp.float32)
            return zeros, zeros

        q_hat, tq_hat = jax.lax.cond(jnp.any(lens > 0), run, skip, sub)
        return key_, (q_hat, tq_hat)

    _key, (q_hat, tq_hat) = jax.lax.scan(body, key, jnp.arange(n_chunks))
    q_rows = q_hat.reshape((n_slots, pad))
    tq_rows = tq_hat.reshape((n_slots, pad))
    q_base = jnp.where(reset, jnp.zeros_like(q_buffer), q_buffer)
    tq_base = jnp.where(reset, jnp.zeros_like(tq_buffer), tq_buffer)
    return _scatter_add(q_base, q_rows, starts, lengths), _scatter_add(tq_base, tq_rows, starts, lengths)
