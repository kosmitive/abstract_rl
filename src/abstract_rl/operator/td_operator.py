import jax
import jax.numpy as jnp
from jaxtyping import Array

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class TemporalDifferenceOperator(TrajectoryOperator):
    """
    Represents a generalized advantage estimation operator based upon https://arxiv.org/abs/1506.02438.
    """

    def __repr__(self) -> str:
        return "td"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new generalized advantage estimator.
        :param mc: The model configuration to use.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf

        # get discount
        self.env = mc["env"]

    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state (gives the target v network).
        :param trajectory: trajectory to transform.
        """

        v_net = state.get("v_network", True)

        # obtain rewards
        discount = self.env.discount()
        t_states = trajectory.states
        t_rewards = trajectory.rewards
        dones = trajectory.dones
        v = v_net.forward(t_states)
        mask = 1 - dones

        # bootstrap from the observed next state if an episode was truncated (e.g. time limit) instead of treating the
        # end of the trajectory as terminal; inside an episode the next value is v[t + 1]
        next_v = v_net.forward(trajectory.next_states) * mask
        td_errors = t_rewards + discount * next_v - v

        def step(tv_next: Array, xs: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
            r, m, last_v, end = xs
            tv_next = jnp.where(end, last_v, tv_next)
            tv = r + discount * m * tv_next
            return tv, tv

        xs = (t_rewards[:, 0], mask[:, 0], next_v[:, 0], trajectory.ends)
        _, tv = jax.lax.scan(step, jnp.zeros((), t_rewards.dtype), xs, reverse=True)

        return trajectory.set("tv", tv).set("td", td_errors)
