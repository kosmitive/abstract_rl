from typing import Any

import jax.numpy as jnp

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.typing import DTYPE
from abstract_rl.policy.non_parametric.non_parametric_policy import NonParametricPolicy


class NonParametricVariationalPolicyBuilder:
    """
    Represents a non parametric variational policy builder. It can be fed with data samples and reweights them
    to a non parametric distribution by solving the optimization problem from REPS
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        self.mc = mc
        conf = mc["conf"]
        self.policy_config = conf
        self.non_parametric_variational_eps = conf["eps"]
        self.non_parametric_variational_lr = conf["lr"]
        self.non_parametric_variational_init_eta = conf["init_eta"]
        self.non_parametric_policy_max_steps = conf["max_steps"]
        self.max_steps = conf["max_steps"]

    def init_state(self) -> dict[str, Any]:
        """The traced temperature, warm started from rnk_init_eta."""
        return {"eta": jnp.asarray([self.non_parametric_variational_init_eta], DTYPE)}

    def calc_eta(self, state: ModelState, tc: TrajectoryCollection) -> ModelState:
        """
        Calculate the optimal eta using the find optimal temperature function, over the valid rows of the collection.
        """
        t = tc.trajectories()

        # append all q values and likelihoods
        q_tensor = t["add_q"].reshape((-1, t["add_q"].shape[-1]))
        eta = find_optimal_temperature(
            q_tensor,
            self.non_parametric_variational_eps,
            state.get("eta")[0],
            self.non_parametric_variational_lr,
            mask=t.mask.reshape(-1),
        )
        return state.set("eta", eta)

    def build(self, state: ModelState, trajectory: Trajectory) -> NonParametricPolicy:
        """
        Takes a trajectory or batch data and builds the correct samples.
        :param trajectory: Tje trajectory to transform.
        :return: A non parametric policy.
        """

        exp_samples = trajectory["add_q"] / state.get("eta")
        return NonParametricPolicy(exp_samples)
