"""Fixed-capacity replay stored as on-device arrays.

Transitions are packed at the front of the buffer. Sampling inside a jitted update indexes
``0 .. size`` and never sees the unused tail, so the compiled shapes stay constant as the
buffer fills.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


class ReplayBuffer:
    """Circular-free replay: drop oldest episodes from the front when the buffer is full."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, n_actions: int, time_limit: int) -> None:
        """Allocate the buffer.

        Args:
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            capacity: Maximum number of transitions.
            n_actions: Extra actions stored per state by the E-step.
            time_limit: Episode length used to pad Retrace batches.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self.time_limit = int(time_limit)
        # Retrace pads to this length. It stays at the constructor value unless a longer episode arrives,
        # so the compiled episode tensor does not change shape as the buffer fills.
        self.pad_limit = int(time_limit)
        self.size = 0
        self.episodes: list[tuple[int, int]] = []
        self.states: Float[Array, "cap obs"] = jnp.zeros((self.capacity, obs_dim), dtype=jnp.float32)
        self.actions: Float[Array, "cap act"] = jnp.zeros((self.capacity, act_dim), dtype=jnp.float32)
        self.rewards: Float[Array, " cap"] = jnp.zeros((self.capacity,), dtype=jnp.float32)
        self.next_states: Float[Array, "cap obs"] = jnp.zeros((self.capacity, obs_dim), dtype=jnp.float32)
        self.dones: Float[Array, " cap"] = jnp.zeros((self.capacity,), dtype=jnp.float32)
        self.log_probs: Float[Array, " cap"] = jnp.zeros((self.capacity,), dtype=jnp.float32)
        self.add_act: Float[Array, "cap k act"] = jnp.zeros((self.capacity, self.n_actions, act_dim), dtype=jnp.float32)
        self.add_q: Float[Array, "cap k"] = jnp.zeros((self.capacity, self.n_actions), dtype=jnp.float32)
        self.add_pi: Float[Array, "cap k"] = jnp.zeros((self.capacity, self.n_actions), dtype=jnp.float32)
        self.ranking: Float[Array, "cap k"] = jnp.zeros((self.capacity, self.n_actions), dtype=jnp.float32)
        self.q: Float[Array, " cap"] = jnp.zeros((self.capacity,), dtype=jnp.float32)
        self.tq: Float[Array, " cap"] = jnp.zeros((self.capacity,), dtype=jnp.float32)
        self.target_stats: Float[Array, "cap stats"] = jnp.zeros((self.capacity, 2 * act_dim), dtype=jnp.float32)

    def clear(self) -> None:
        """Drop every stored transition. The allocated arrays stay in place."""
        self.size = 0
        self.episodes.clear()

    def add_trajectories(self, trajectories: list[Trajectory] | Any) -> None:
        """Append episodes, dropping the oldest ones so the buffer stays within capacity.

        The episodes are packed on the host and written with one update per field. A per-episode
        device write copies the whole allocation each time, which dominates once the replay is large.
        """
        blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for trajectory in trajectories:
            states = np.asarray(trajectory.states, dtype=np.float32)
            if states.shape[0] == 0:
                continue
            blocks.append(
                (
                    states,
                    np.asarray(trajectory.actions, dtype=np.float32).reshape(-1, self.act_dim),
                    np.asarray(trajectory.rewards, dtype=np.float32).reshape(-1),
                    np.asarray(trajectory.next_states, dtype=np.float32),
                    np.asarray(trajectory.dones, dtype=np.float32).reshape(-1),
                    np.asarray(trajectory.log_likelihoods, dtype=np.float32).reshape(-1),
                )
            )
        if not blocks:
            return
        # An episode longer than the allocation still uses the one-at-a-time rule, which keeps its tail.
        if any(block[0].shape[0] > self.capacity for block in blocks):
            for block in blocks:
                self._add_episode(*block)
            return
        new_total = sum(block[0].shape[0] for block in blocks)
        drop_len = 0
        while self.episodes and self.size - drop_len + new_total > self.capacity:
            _start, length = self.episodes.pop(0)
            drop_len += length
        if drop_len:
            self._shift_prefix(drop_len)
        start = self.size
        stacked = tuple(np.concatenate([block[index] for block in blocks], axis=0) for index in range(6))
        self._write_block(start, *stacked)
        cursor = start
        longest = 0
        for block in blocks:
            length = int(block[0].shape[0])
            self.episodes.append((cursor, length))
            cursor += length
            longest = max(longest, length)
        self.size = cursor
        self.time_limit = max(self.time_limit, longest)
        self.pad_limit = max(self.pad_limit, longest)

    def _add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        length = int(states.shape[0])
        if length > self.capacity:
            states, actions, rewards, next_states, dones, log_probs = (
                states[-self.capacity :],
                actions[-self.capacity :],
                rewards[-self.capacity :],
                next_states[-self.capacity :],
                dones[-self.capacity :],
                log_probs[-self.capacity :],
            )
            length = self.capacity
            self.episodes.clear()
            self.size = 0
        drop_len = 0
        while self.episodes and self.size - drop_len + length > self.capacity:
            _oldest, dropped = self.episodes.pop(0)
            drop_len += dropped
        if drop_len:
            self._shift_prefix(drop_len)
        start = self.size
        self._write(start, states, actions, rewards, next_states, dones, log_probs)
        self.episodes.append((start, length))
        self.size += length
        self.time_limit = max(self.time_limit, length)
        self.pad_limit = max(self.pad_limit, length)

    def _write_block(
        self,
        start: int,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        """Write one contiguous host block. Callers pack every new episode into that block.

        The rows are padded to the allocation and the start index is a traced scalar, so filling
        the replay does not compile a new scatter.
        """
        count = self.capacity
        length = int(states.shape[0])
        kept = (
            jnp.asarray(_pad_rows(states, count)),
            jnp.asarray(_pad_rows(actions, count)),
            jnp.asarray(_pad_rows(rewards, count)),
            jnp.asarray(_pad_rows(next_states, count)),
            jnp.asarray(_pad_rows(dones, count)),
            jnp.asarray(_pad_rows(log_probs, count)),
            jnp.int32(start),
            jnp.int32(length),
        )
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.log_probs = _insert_transitions(
            kept,
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        )

    def _shift_prefix(self, drop_len: int) -> None:
        """Drop a prefix of transitions with one copy per field, then fix episode offsets.

        ``drop_len`` is traced. The shift is the same program at every fill level.
        """
        new_size = self.size - drop_len
        kept = (jnp.int32(drop_len), jnp.int32(new_size))
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
            self.add_act,
            self.add_q,
            self.add_pi,
            self.ranking,
            self.q,
            self.tq,
            self.target_stats,
        ) = _shift_transitions(
            kept,
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
            self.add_act,
            self.add_q,
            self.add_pi,
            self.ranking,
            self.q,
            self.tq,
            self.target_stats,
        )
        self.size = new_size
        self.episodes = [(start - drop_len, length) for start, length in self.episodes]

    def _write(
        self,
        start: int,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        self._write_block(start, states, actions, rewards, next_states, dones, log_probs)


def _pad_rows(rows: np.ndarray, count: int) -> np.ndarray:
    """Pad ``rows`` on the host so the device write has the allocation's leading size."""
    padded = rows
    match rows.shape[0] == count:
        case False:
            padded = np.zeros((count, *rows.shape[1:]), dtype=rows.dtype)
            padded[: rows.shape[0]] = rows
        case True:
            pass
    return padded


def _apply_update(dest: Array, rows: Array, start: Array, length: Array) -> Array:
    """Replace ``dest[start:start + length]`` with ``rows``. Indices are traced scalars."""
    count = dest.shape[0]
    position = jnp.arange(count)
    source = position - start
    valid = (source >= 0) & (source < length)
    updated = rows[jnp.clip(source, 0, count - 1)]
    mask = valid.reshape((count,) + (1,) * (dest.ndim - 1))
    return jnp.where(mask, updated, dest)


def _apply_shift(array: Array, drop_len: Array, new_size: Array) -> Array:
    """Move a prefix of ``array`` to the front. ``drop_len`` and ``new_size`` are traced."""
    count = array.shape[0]
    index = jnp.arange(count)
    moved = array[jnp.minimum(index + drop_len, count - 1)]
    mask = (index < new_size).reshape((count,) + (1,) * (array.ndim - 1))
    return jnp.where(mask, moved, array)


@eqx.filter_jit(donate="all-except-first")
def _insert_transitions(
    kept: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    states: Array,
    actions: Array,
    rewards: Array,
    next_states: Array,
    dones: Array,
    log_probs: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Insert one padded block into the six transition fields.

    The block and the traced start and length are kept. The six buffers are donated and
    replaced, so the caller must use the returned arrays.
    """
    row_states, row_actions, row_rewards, row_next, row_dones, row_log_probs, start, length = kept
    return (
        _apply_update(states, row_states, start, length),
        _apply_update(actions, row_actions, start, length),
        _apply_update(rewards, row_rewards, start, length),
        _apply_update(next_states, row_next, start, length),
        _apply_update(dones, row_dones, start, length),
        _apply_update(log_probs, row_log_probs, start, length),
    )


@eqx.filter_jit(donate="all-except-first")
def _shift_transitions(
    kept: tuple[Array, Array],
    states: Array,
    actions: Array,
    rewards: Array,
    next_states: Array,
    dones: Array,
    log_probs: Array,
    add_act: Array,
    add_q: Array,
    add_pi: Array,
    ranking: Array,
    q_values: Array,
    tq_values: Array,
    target_stats: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Drop a prefix of every replay field. The offsets are traced; the buffers are donated."""
    drop_len, new_size = kept
    return (
        _apply_shift(states, drop_len, new_size),
        _apply_shift(actions, drop_len, new_size),
        _apply_shift(rewards, drop_len, new_size),
        _apply_shift(next_states, drop_len, new_size),
        _apply_shift(dones, drop_len, new_size),
        _apply_shift(log_probs, drop_len, new_size),
        _apply_shift(add_act, drop_len, new_size),
        _apply_shift(add_q, drop_len, new_size),
        _apply_shift(add_pi, drop_len, new_size),
        _apply_shift(ranking, drop_len, new_size),
        _apply_shift(q_values, drop_len, new_size),
        _apply_shift(tq_values, drop_len, new_size),
        _apply_shift(target_stats, drop_len, new_size),
    )
