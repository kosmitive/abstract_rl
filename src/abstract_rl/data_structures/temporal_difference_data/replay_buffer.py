"""Fixed-capacity replay stored as on-device arrays.

Transitions are packed at the front of the buffer. Sampling inside a jitted update indexes
``0 .. size`` and never sees the unused tail, so the compiled shapes stay constant as the
buffer fills.
"""

from typing import Any

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
        """Append episodes, dropping the oldest ones so the buffer stays within capacity."""
        for trajectory in trajectories:
            self._add_episode(
                np.asarray(trajectory.states, dtype=np.float32),
                np.asarray(trajectory.actions, dtype=np.float32).reshape(-1, self.act_dim),
                np.asarray(trajectory.rewards, dtype=np.float32).reshape(-1),
                np.asarray(trajectory.next_states, dtype=np.float32),
                np.asarray(trajectory.dones, dtype=np.float32).reshape(-1),
                np.asarray(trajectory.log_likelihoods, dtype=np.float32).reshape(-1),
            )

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
        while self.episodes and self.size + length > self.capacity:
            self._drop_oldest()
        start = self.size
        self._write(start, states, actions, rewards, next_states, dones, log_probs)
        self.episodes.append((start, length))
        self.size += length
        self.time_limit = max(self.time_limit, length)

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
        end = start + len(states)
        self.states = self.states.at[start:end].set(jnp.asarray(states))
        self.actions = self.actions.at[start:end].set(jnp.asarray(actions))
        self.rewards = self.rewards.at[start:end].set(jnp.asarray(rewards))
        self.next_states = self.next_states.at[start:end].set(jnp.asarray(next_states))
        self.dones = self.dones.at[start:end].set(jnp.asarray(dones))
        self.log_probs = self.log_probs.at[start:end].set(jnp.asarray(log_probs))

    def _drop_oldest(self) -> None:
        _start, length = self.episodes.pop(0)
        new_size = self.size - length

        def shift(array: Array) -> Array:
            return array.at[:new_size].set(array[length : length + new_size])

        self.states = shift(self.states)
        self.actions = shift(self.actions)
        self.rewards = shift(self.rewards)
        self.next_states = shift(self.next_states)
        self.dones = shift(self.dones)
        self.log_probs = shift(self.log_probs)
        self.add_act = shift(self.add_act)
        self.add_q = shift(self.add_q)
        self.add_pi = shift(self.add_pi)
        self.ranking = shift(self.ranking)
        self.q = shift(self.q)
        self.tq = shift(self.tq)
        self.target_stats = shift(self.target_stats)
        self.size = new_size
        self.episodes = [(start - length, episode_length) for start, episode_length in self.episodes]
