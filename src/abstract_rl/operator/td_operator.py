"""Temporal-difference returns and errors, bootstrapping through time-limit truncations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class TemporalDifferenceOperator(TrajectoryOperator):
    """TD(1) returns and one-step TD errors from the target value function."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        self.conf = mc.get("conf")
        self.v_net = mc.get("v_network", True)
        self.env = mc["env"]

    def __repr__(self) -> str:
        return "td"

    def transform(self, trajectory: Trajectory) -> None:
        self.transform_all([trajectory])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()
        if not trajectories:
            return
        tv, td = temporal_difference(
            self.v_net.params,
            trajectories,
            float(self.env.discount()),
        )
        for trajectory, row_tv, row_td in zip(trajectories, np.asarray(tv), np.asarray(td)):
            length = len(trajectory)
            trajectory["tv"] = row_tv[:length]
            trajectory["td"] = row_td[:length]


def temporal_difference(
    v_net: MultilayerPerceptron,
    trajectories: list[Trajectory],
    discount: float,
) -> tuple[Float[Array, "episodes time"], Float[Array, "episodes time"]]:
    """Vectorised TD returns and errors for a list of episodes.

    Args:
        v_net: Value network. Evaluated once on every stored state.
        trajectories: Episodes. Shorter ones are padded; padding does not affect real prefixes.
        discount: Discount factor.

    Returns:
        Returns and TD errors, each ``[episodes, max_length]``.
    """
    lengths = np.asarray([len(trajectory) for trajectory in trajectories], dtype=np.int32)
    max_len = int(lengths.max())
    obs = trajectories[0].all_states.shape[-1]
    all_states = np.zeros((len(trajectories), max_len + 1, obs), dtype=np.float32)
    rewards = np.zeros((len(trajectories), max_len), dtype=np.float32)
    dones = np.ones((len(trajectories), max_len), dtype=np.float32)
    for index, trajectory in enumerate(trajectories):
        length = lengths[index]
        all_states[index, : length + 1] = np.asarray(trajectory.all_states, dtype=np.float32)
        rewards[index, :length] = np.asarray(trajectory.rewards, dtype=np.float32).reshape(-1)
        dones[index, :length] = np.asarray(trajectory.dones, dtype=np.float32).reshape(-1)
    return _td_jit(
        v_net,
        jnp.asarray(all_states),
        jnp.asarray(rewards),
        jnp.asarray(dones),
        jnp.asarray(lengths),
        jnp.float32(discount),
    )


@eqx.filter_jit
def _td_jit(
    v_net: MultilayerPerceptron,
    all_states: Float[Array, "episodes time_plus obs"],
    rewards: Float[Array, "episodes time"],
    dones: Float[Array, "episodes time"],
    lengths: Array,
    discount: Float[Array, ""],
) -> tuple[Float[Array, "episodes time"], Float[Array, "episodes time"]]:
    values = v_net.forward(all_states).reshape(all_states.shape[0], all_states.shape[1])
    episodes = jnp.arange(lengths.shape[0])
    last_index = jnp.clip(lengths, 0, values.shape[1] - 1)
    prev_index = jnp.clip(lengths - 1, 0, dones.shape[1] - 1)
    alive = lengths > 0
    last_v = jnp.where(alive, values[episodes, last_index] * (1.0 - dones[episodes, prev_index]), 0.0)
    time = jnp.arange(rewards.shape[1])[::-1]

    def body(
        carry: tuple[Float[Array, " episodes"], Float[Array, " episodes"]], t: Array
    ) -> tuple[
        tuple[Float[Array, " episodes"], Float[Array, " episodes"]],
        tuple[Float[Array, " episodes"], Float[Array, " episodes"]],
    ]:
        next_value, tv_next = carry
        active = t < lengths
        at_end = t == (lengths - 1)
        next_value = jnp.where(at_end, last_v, next_value)
        tv_next = jnp.where(at_end, last_v, tv_next)
        mask = 1.0 - dones[:, t]
        reward = rewards[:, t]
        value = values[:, t]
        tv_t = reward + discount * mask * tv_next
        td_t = reward + discount * next_value * mask - value
        tv_t = jnp.where(active, tv_t, 0.0)
        td_t = jnp.where(active, td_t, 0.0)
        next_value = jnp.where(active, value, next_value)
        tv_next = jnp.where(active, tv_t, tv_next)
        return (next_value, tv_next), (tv_t, td_t)

    _, (tv_rev, td_rev) = jax.lax.scan(
        body,
        (jnp.zeros(lengths.shape[0]), jnp.zeros(lengths.shape[0])),
        time,
    )
    return tv_rev[::-1].T, td_rev[::-1].T
