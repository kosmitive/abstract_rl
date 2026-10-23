from typing import Any

import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.mixed import to_mx
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
        self.tar_network = mc.get("q_network", True)
        self.tar_policy = mc.get("policy", True)
        self.eta: Any = [self.non_parametric_variational_init_eta]
        self.log = self.mc["log"]
        self.log.create_field("eta", 1)

    def calc_eta(self) -> None:
        """
        Calculate the optimal eta using the find optimal temperature function.
        """
        tc = self.mc.get("tc")

        # append all q values and likelihoods
        q_tensor = np.concatenate(tuple([q["add_q"] for q in tc.trajectories()]))
        eta = find_optimal_temperature(
            to_mx(q_tensor), self.non_parametric_variational_eps, self.eta[0], self.non_parametric_variational_lr
        )

        self.eta = np.array(eta)
        self.log.log({"eta": self.eta})

    def build(self, trajectory: Trajectory) -> NonParametricPolicy:
        """
        Takes a trajectory or batch data and builds the correct samples.
        :param trajectory: Tje trajectory to transform.
        :return: A non parametric policy.
        """

        exp_samples = trajectory["add_q"] / self.eta
        return NonParametricPolicy(exp_samples)
