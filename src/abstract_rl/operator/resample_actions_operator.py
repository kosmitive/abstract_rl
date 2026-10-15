"""Sample extra actions from the target policy and score them with the target Q."""

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
from abstract_rl.misc import buckets
from abstract_rl.misc.obs_norm import normalize_mc
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, log_prob, sample
from abstract_rl.value_functions.fc_q_networks import q_values

_CHUNK = 512


class ResampleOperator(TrajectoryOperator):
    """E-step action samples. Stored rows are scored in fixed-size chunks."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        self.conf = mc.get("conf")
        self.add_act = int(self.conf["add_acts"])
        self.mc = mc
        self.policy = mc.get("policy", True)
        self.q_net = mc.get("q_network", True)

    def __repr__(self) -> str:
        return "resample"

    def transform(self, trajectory: Trajectory) -> None:
        """Sample extra actions for one trajectory and store them on it."""
        states = normalize_mc(self.mc, trajectory.states)
        pieces_a: list[np.ndarray] = []
        pieces_q: list[np.ndarray] = []
        pieces_p: list[np.ndarray] = []
        offset = 0
        while offset < states.shape[0]:
            count = min(_CHUNK, int(states.shape[0]) - offset)
            actions, q_hat, log_density = resample_chunk(
                self.policy.params,
                self.q_net.params,
                _pad_rows(states, offset, count),
                self.mc.rng.split(),
                self.add_act,
            )
            pieces_a.append(np.asarray(actions[:count]))
            pieces_q.append(np.asarray(q_hat[:count]))
            pieces_p.append(np.asarray(log_density[:count]))
            offset += count
        trajectory["add_act"] = np.concatenate(pieces_a, axis=0)
        trajectory["add_q"] = np.concatenate(pieces_q, axis=0)
        trajectory["add_pi"] = np.concatenate(pieces_p, axis=0)

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """Score every stored state. A device-backed collection is one jitted pass."""
        if isinstance(trajectories, TrajectoryCollection) and trajectories.device is not None:
            self.apply(trajectories.device, self.mc.rng.split())
            trajectories.scatter_field("add_act", trajectories.device.add_act)
            trajectories.scatter_field("add_q", trajectories.device.add_q)
            trajectories.scatter_field("add_pi", trajectories.device.add_pi)
            return
        super().transform_all(trajectories)

    def apply(self, replay: ReplayBuffer, key: PRNGKeyArray) -> None:
        """Fill ``add_act``, ``add_q`` and ``add_pi`` for every stored state.

        The scan length is a power-of-two prefix of the allocation (the whole allocation once the
        buffer is full). Chunk ``i`` still draws ``split(key, n_full)[i]``, the same key the
        full-allocation scan uses, so the stored rows do not depend on the prefix length.

        Args:
            replay: Replay to update in place.
            key: Random key. One key per full-allocation chunk; only the filled chunks are run.
        """
        if replay.size == 0:
            return
        n_chunks = buckets.chunk_count(replay.size, replay.capacity, _CHUNK)
        actions, q_hat, log_density = resample_buffer(
            (
                self.policy.params,
                self.q_net.params,
                normalize_mc(self.mc, replay.states),
                key,
                self.add_act,
                jnp.int32(replay.size),
            ),
            replay.add_act,
            replay.add_q,
            replay.add_pi,
            n_chunks,
        )
        replay.add_act = actions
        replay.add_q = q_hat
        replay.add_pi = log_density


def _pad_rows(values: Array, offset: int, count: int) -> Array:
    padded = jnp.zeros((_CHUNK, *values.shape[1:]), dtype=values.dtype)
    return padded.at[:count].set(values[offset : offset + count])


def _resample(
    policy: GaussianParams,
    q_net: MultilayerPerceptron,
    states: Float[Array, "chunk obs"],
    key: PRNGKeyArray,
    n_actions: int,
) -> tuple[Float[Array, "chunk k act"], Float[Array, "chunk k"], Float[Array, "chunk k"]]:
    """Sample and score ``n_actions`` actions for each state."""
    actions = sample(policy, key, n_actions, states=states)
    return actions, q_values(q_net, states, actions), log_prob(policy, actions, states=states)


@eqx.filter_jit
def resample_chunk(
    policy: GaussianParams,
    q_net: MultilayerPerceptron,
    states: Float[Array, "chunk obs"],
    key: PRNGKeyArray,
    n_actions: int,
) -> tuple[Float[Array, "chunk k act"], Float[Array, "chunk k"], Float[Array, "chunk k"]]:
    """Jitted ``_resample`` for one host-side chunk, used when a trajectory is not on device."""
    return _resample(policy, q_net, states, key, n_actions)


@eqx.filter_jit(donate="all-except-first")
def resample_buffer(
    kept: tuple[Any, ...],
    add_act: Float[Array, "cap k act"],
    add_q: Float[Array, "cap k"],
    add_pi: Float[Array, "cap k"],
    n_chunks: int,
) -> tuple[Float[Array, "cap k act"], Float[Array, "cap k"], Float[Array, "cap k"]]:
    """Score the first ``n_chunks`` blocks. Rows at or past ``size`` are zero.

    ``n_chunks`` is static, so each power-of-two prefix compiles once. Keys are split for the
    whole allocation; block ``i`` uses the same key it would in a full scan. The previous
    ``add_*`` buffers are donated and overwritten. The policy, Q network and states stay live.
    """
    policy, q_net, states, key, n_actions, size = kept
    capacity = states.shape[0]
    n_full = (capacity + _CHUNK - 1) // _CHUNK
    keys = jax.random.split(key, n_full)

    def body(carry: Array, index: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        start = index * _CHUNK
        rows = jnp.minimum(jnp.arange(_CHUNK) + start, capacity - 1)
        actions, q_hat, log_density = _resample(policy, q_net, states[rows], keys[index], n_actions)
        valid = (jnp.arange(_CHUNK) + start) < size
        actions = jnp.where(valid[:, None, None], actions, 0)
        q_hat = jnp.where(valid[:, None], q_hat, 0)
        log_density = jnp.where(valid[:, None], log_density, 0)
        return carry, (actions, q_hat, log_density)

    _carry, (actions, q_hat, log_density) = jax.lax.scan(body, jnp.int32(0), jnp.arange(n_chunks))
    act = actions.shape[-1]
    flat_actions = actions.reshape((n_chunks * _CHUNK, n_actions, act))
    flat_q = q_hat.reshape((n_chunks * _CHUNK, n_actions))
    flat_pi = log_density.reshape((n_chunks * _CHUNK, n_actions))
    prefix = min(n_chunks * _CHUNK, capacity)
    return (
        jnp.zeros_like(add_act).at[:prefix].set(flat_actions[:prefix]),
        jnp.zeros_like(add_q).at[:prefix].set(flat_q[:prefix]),
        jnp.zeros_like(add_pi).at[:prefix].set(flat_pi[:prefix]),
    )
