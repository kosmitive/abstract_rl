"""Non-parametric E-step weights."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.replay_buffer import ReplayBuffer
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc import buckets
from abstract_rl.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.policy.non_parametric.non_parametric_variational_policy_builder import (
    NonParametricVariationalPolicyBuilder,
)


class VariationalRankingOperator(TrajectoryOperator):
    """Per-state normalised weights ``log q(a|s) = Q(s,a)/η - logsumexp_a Q(s,a)/η``."""

    def __init__(self, mc: ModelConfiguration) -> None:
        self.conf = mc["conf"]
        self.mc = mc
        self.policy = mc.get("policy", True)
        self.num_add_actions = self.conf.get_root("res_add_acts")
        self.b = NonParametricVariationalPolicyBuilder(self.mc)

    def __repr__(self) -> str:
        return "reweighing"

    def hook(self) -> None:
        self.b.calc_eta()

    def transform(self, trajectory: Trajectory) -> None:
        """Normalise one trajectory's Q samples into ``grad_ranking``."""
        eta = float(np.reshape(self.b.eta, ()))
        trajectory["grad_ranking"] = np.asarray(e_step_log_weights(trajectory["add_q"], eta))

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """Write ``grad_ranking`` for every stored transition."""
        if isinstance(trajectories, TrajectoryCollection) and trajectories.device is not None:
            self.apply(trajectories.device)
            trajectories.scatter_field("grad_ranking", trajectories.device.ranking)
            return
        super().transform_all(trajectories)

    def apply(self, replay: ReplayBuffer) -> None:
        """Write ``ranking`` for the stored extra actions. ``hook`` must have run first.

        The softmax covers a power-of-two prefix of the allocation. It is per state. A shorter
        prefix matches the full-buffer rows up to one float32 ulp, because the batched kernel
        depends on the row count; at full width the kernel is the full-buffer one. ``hook`` has
        already fit η on every stored row.
        """
        if replay.size == 0:
            return
        eta = jnp.float32(float(np.reshape(self.b.eta, ())))
        width = buckets.active_width(replay.size, int(replay.add_q.shape[0]))
        replay.ranking = _ranking_full(replay.add_q, jnp.int32(replay.size), eta, width)


@eqx.filter_jit
def _ranking_full(
    q_values: Float[Array, "cap k"], size: Array, eta: Float[Array, ""], width: int
) -> Float[Array, "cap k"]:
    """Per-state log weights for the first ``width`` rows. Rows past ``size`` are zero.

    ``width`` is static. Log-softmax is independent across rows. A shorter batch matches those
    rows up to one float32 ulp; the full allocation is the steady-state kernel.
    """
    weights = jax.nn.log_softmax(q_values[:width] / eta, axis=-1)
    valid = jnp.arange(width) < size
    head = jnp.where(valid[:, None], weights, 0)
    return jnp.zeros_like(q_values).at[:width].set(head)


def e_step_log_weights(q_values: Float[Array, "batch k"] | np.ndarray, eta: float) -> Float[Array, "batch k"]:
    """Log of the non-parametric variational distribution, normalised per state.

    Args:
        q_values: Q samples, shape ``[batch, k]``.
        eta: Temperature. The weights are ``softmax(Q / eta)``.

    Returns:
        Log weights that sum to one over the action axis in linear space.
    """
    return jax.nn.log_softmax(jnp.asarray(q_values, dtype=jnp.float32) / jnp.float32(eta), axis=-1)
