"""Retrace(λ) action-value targets.

The return at time t is

    Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{k=0}^{B-1} γ^k (prod_{i=1}^{k} c_{t+i}) δ_{t+k}

with an empty product equal to 1 and ``c_s = λ min(1, π(a_s|s_s) / μ(a_s|s_s))``.
``c_t`` of the current step is not part of that product; the trace starts at the next action.
"""

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

        Args:
            replay: Replay whose behaviour log probabilities are the denominator of the trace.
            key: Random key for the action samples inside the value estimate.
        """
        if replay.size == 0 or not replay.episodes:
            return
        time_limit = replay.time_limit
        q_rows: list[Array] = []
        tq_rows: list[Array] = []
        indices: list[Array] = []
        for start in range(0, len(replay.episodes), _CHUNK):
            group = replay.episodes[start : start + _CHUNK]
            key, sub = jax.random.split(key)
            padded = _pad(replay, group, time_limit)
            q_hat, tq_hat = _retrace_chunk(
                self.q_network.params,
                self.policy.params,
                *padded,
                sub,
                jnp.float32(self.discount),
                jnp.float32(self.ret_lambda),
                self.bootstrap,
                _V_SAMPLES,
            )
            for row, (episode_start, length) in enumerate(group):
                if length <= 0:
                    continue
                length = min(length, time_limit)
                indices.append(jnp.arange(episode_start, episode_start + length))
                q_rows.append(q_hat[row, :length])
                tq_rows.append(tq_hat[row, :length])
        if not indices:
            return
        index = jnp.concatenate(indices)
        replay.q = replay.q.at[index].set(jnp.concatenate(q_rows))
        replay.tq = replay.tq.at[index].set(jnp.concatenate(tq_rows))


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
    for k in range(1, bootstrap):
        if k >= time:
            break
        valid = (jnp.arange(time - k)[None, :] + k) < lengths[:, None]
        updated = coeff[:, : time - k] * discount * coefficients[:, k:]
        updated = jnp.where(valid, updated, 0.0)
        targets = targets.at[:, : time - k].add(jnp.where(valid, updated * td[:, k:], 0.0))
        coeff = coeff.at[:, : time - k].set(updated)
    return targets


def _pad(
    replay: ReplayBuffer, group: list[tuple[int, int]], time_limit: int
) -> tuple[
    Float[Array, "chunk time obs"],
    Float[Array, "chunk time act"],
    Float[Array, "chunk time"],
    Float[Array, "chunk time obs"],
    Float[Array, "chunk time"],
    Float[Array, "chunk time"],
    Array,
]:
    """Pad a chunk of episodes to ``[_CHUNK, time_limit]``."""
    obs, act = replay.obs_dim, replay.act_dim
    states = np.zeros((_CHUNK, time_limit, obs), dtype=np.float32)
    actions = np.zeros((_CHUNK, time_limit, act), dtype=np.float32)
    rewards = np.zeros((_CHUNK, time_limit), dtype=np.float32)
    next_states = np.zeros((_CHUNK, time_limit, obs), dtype=np.float32)
    dones = np.ones((_CHUNK, time_limit), dtype=np.float32)
    log_probs = np.zeros((_CHUNK, time_limit), dtype=np.float32)
    lengths = np.zeros((_CHUNK,), dtype=np.int32)
    for index, (start, length) in enumerate(group):
        length = min(int(length), time_limit)
        lengths[index] = length
        states[index, :length] = np.asarray(replay.states[start : start + length])
        actions[index, :length] = np.asarray(replay.actions[start : start + length])
        rewards[index, :length] = np.asarray(replay.rewards[start : start + length])
        next_states[index, :length] = np.asarray(replay.next_states[start : start + length])
        dones[index, :length] = np.asarray(replay.dones[start : start + length])
        log_probs[index, :length] = np.asarray(replay.log_probs[start : start + length])
    return (
        jnp.asarray(states),
        jnp.asarray(actions),
        jnp.asarray(rewards),
        jnp.asarray(next_states),
        jnp.asarray(dones),
        jnp.asarray(log_probs),
        jnp.asarray(lengths),
    )


@eqx.filter_jit
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
