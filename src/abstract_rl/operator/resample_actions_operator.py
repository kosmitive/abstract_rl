"""Sample extra actions from the target policy and score them with the target Q."""

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
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, log_prob, sample
from abstract_rl.value_functions.fc_q_networks import q_values

_CHUNK = 512


class ResampleOperator(TrajectoryOperator):
    """E-step action samples. The whole buffer is scored in fixed-size chunks."""

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
        states = jnp.asarray(trajectory.states, dtype=jnp.float32)
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

        Args:
            replay: Replay to update in place.
            key: Random key, split once per chunk.
        """
        offset = 0
        while offset < replay.size:
            key, sub = jax.random.split(key)
            count = min(_CHUNK, replay.size - offset)
            states = _pad_rows(replay.states, offset, count)
            actions, q_hat, log_density = resample_chunk(
                self.policy.params, self.q_net.params, states, sub, self.add_act
            )
            end = offset + count
            replay.add_act = replay.add_act.at[offset:end].set(actions[:count])
            replay.add_q = replay.add_q.at[offset:end].set(q_hat[:count])
            replay.add_pi = replay.add_pi.at[offset:end].set(log_density[:count])
            offset = end


def _pad_rows(values: Array, offset: int, count: int) -> Array:
    padded = jnp.zeros((_CHUNK, *values.shape[1:]), dtype=values.dtype)
    return padded.at[:count].set(values[offset : offset + count])


@eqx.filter_jit
def resample_chunk(
    policy: GaussianParams,
    q_net: MultilayerPerceptron,
    states: Float[Array, "chunk obs"],
    key: PRNGKeyArray,
    n_actions: int,
) -> tuple[Float[Array, "chunk k act"], Float[Array, "chunk k"], Float[Array, "chunk k"]]:
    """Sample and score ``n_actions`` actions for each state. Q is vmapped over the action axis."""
    actions = sample(policy, key, n_actions, states=states)
    return actions, q_values(q_net, states, actions), log_prob(policy, actions, states=states)
