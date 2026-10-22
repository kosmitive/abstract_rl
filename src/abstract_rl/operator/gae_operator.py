import jax
import jax.numpy as jnp
from jaxtyping import Array

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class GeneralizedAdvantagesEstimationOperator(TrajectoryOperator):
    """
    Represents a generalized advantage estimation operator based upon https://arxiv.org/abs/1506.02438.
    """

    def __repr__(self) -> str:
        return "gae"

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
        self.gae_lambda = float(self.conf["lambda"])

    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectory: trajectory to transform.
        """

        # obtain rewards
        discount = self.env.discount()

        # use v approximation to get advantages
        td_errors = trajectory["td"][:, 0]
        dones = trajectory.dones
        mask = 1 - dones[:, 0]

        # get combined ratio
        gae_gamma_lambda = discount * self.gae_lambda

        def step(a_next: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            td, m, end = xs
            # an advantage never crosses the end of an episode
            a_next = jnp.where(end, 0.0, a_next)
            a = td + m * gae_gamma_lambda * a_next
            return a, a

        _, a = jax.lax.scan(step, jnp.zeros((), td_errors.dtype), (td_errors, mask, trajectory.ends), reverse=True)
        return trajectory.set("a", a)
