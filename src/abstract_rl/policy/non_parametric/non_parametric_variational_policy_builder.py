from typing import Any

import mlx.core as mx
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
        self.eta_mx = mx.array([self.non_parametric_variational_init_eta], dtype=mx.float32)
        self.log = self.mc["log"]
        self.log.create_field("eta", 1)

    def calc_eta(self, add_q: mx.array | None = None) -> None:
        """
        Calculate the optimal eta using the find optimal temperature function.
        :param add_q: The sampled q values [NxA] to use, those of all states of the memory if None.
        """
        if add_q is None:
            # the sampled q values of all states, as the annotation stored them on the device
            add_q = self.mc.get("tc").annotation("add_q")
        eta = find_optimal_temperature(
            add_q, self.non_parametric_variational_eps, self.eta[0], self.non_parametric_variational_lr
        )

        self.eta_mx = eta
        self.eta = np.array(eta)
        self.log.log({"eta": self.eta})

    def log_weights(self, add_q: mx.array) -> mx.array:
        """
        The log weights of the non parametric policy, normalized per state on the device:
        log q(a|s) = Q(s,a) / eta - log sum_a' exp(Q(s,a') / eta).
        """
        samples = to_mx(add_q) / self.eta_mx
        return samples - mx.logsumexp(samples, axis=1, keepdims=True)

    def build(self, trajectory: Trajectory) -> NonParametricPolicy:
        """
        Takes a trajectory or batch data and builds the correct samples.
        :param trajectory: Tje trajectory to transform.
        :return: A non parametric policy.
        """

        exp_samples = trajectory["add_q"] / self.eta
        return NonParametricPolicy(exp_samples)
