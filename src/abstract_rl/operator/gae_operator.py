"""Generalised advantage estimation."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class GeneralizedAdvantagesEstimationOperator(TrajectoryOperator):
    """GAE(λ) advantages from the TD errors of a trajectory."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        self.conf = mc.get("conf")
        self.v_net = mc.get("v_network", True)
        self.env = mc["env"]
        self.gae_lambda = float(self.conf["lambda"])

    def __repr__(self) -> str:
        return "gae"

    def transform(self, trajectory: Trajectory) -> None:
        self.transform_all([trajectory])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()
        if not trajectories:
            return
        advantages = generalized_advantage(
            trajectories,
            float(self.env.discount()),
            self.gae_lambda,
        )
        for trajectory, row in zip(trajectories, np.asarray(advantages)):
            trajectory["a"] = row[: len(trajectory)]


def generalized_advantage(
    trajectories: list[Trajectory], discount: float, gae_lambda: float
) -> Float[Array, "episodes time"]:
    """Vectorised GAE. Each trajectory must already carry a ``td`` field.

    Args:
        trajectories: Episodes with TD errors.
        discount: Discount factor.
        gae_lambda: GAE trace coefficient.

    Returns:
        Advantages, shape ``[episodes, max_length]``.
    """
    lengths = np.asarray([len(trajectory) for trajectory in trajectories], dtype=np.int32)
    max_len = int(lengths.max())
    td = np.zeros((len(trajectories), max_len), dtype=np.float32)
    dones = np.ones((len(trajectories), max_len), dtype=np.float32)
    for index, trajectory in enumerate(trajectories):
        length = int(lengths[index])
        td[index, :length] = np.asarray(trajectory["td"], dtype=np.float32).reshape(-1)
        dones[index, :length] = np.asarray(trajectory.dones, dtype=np.float32).reshape(-1)
    return _gae_jit(jnp.asarray(td), jnp.asarray(dones), jnp.asarray(lengths), jnp.float32(discount * gae_lambda))


@jax.jit
def _gae_jit(
    td: Float[Array, "episodes time"],
    dones: Float[Array, "episodes time"],
    lengths: Array,
    decay: Float[Array, ""],
) -> Float[Array, "episodes time"]:
    time = jnp.arange(td.shape[1])[::-1]

    def body(
        advantage_next: Float[Array, " episodes"], t: Array
    ) -> tuple[Float[Array, " episodes"], Float[Array, " episodes"]]:
        active = t < lengths
        at_end = t == (lengths - 1)
        advantage_next = jnp.where(at_end, 0.0, advantage_next)
        advantage_t = td[:, t] + (1.0 - dones[:, t]) * decay * advantage_next
        advantage_t = jnp.where(active, advantage_t, 0.0)
        advantage_next = jnp.where(active, advantage_t, advantage_next)
        return advantage_next, advantage_t

    _, reversed_advantages = jax.lax.scan(body, jnp.zeros(lengths.shape[0]), time)
    return reversed_advantages[::-1].T
