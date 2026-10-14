"""Temperature of the non-parametric E-step."""

import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc import buckets
from abstract_rl.policy.non_parametric.non_parametric_policy import NonParametricPolicy


class NonParametricVariationalPolicyBuilder:
    """Fits the temperature η of the non-parametric variational distribution on the replay."""

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
        self.eta = np.asarray([self.non_parametric_variational_init_eta], dtype=np.float32)
        self.log = self.mc["log"]
        self.log.create_field("eta", 1)

    def calc_eta(self) -> None:
        """Minimise the dual on every stored Q sample and log the temperature.

        The mean is over the stored rows, not over the M-step minibatch. The device array is
        sliced to the same power-of-two prefix as the other buffer-wide kernels; rows past
        ``size`` stay masked out of the mean.
        """
        collection = self.mc.get("tc")
        device = getattr(collection, "device", None)
        stored = None
        if device is not None and device.size:
            stored = int(device.size)
            width = buckets.active_width(stored, int(device.add_q.shape[0]))
            q_values = device.add_q[:width]
        else:
            q_values = np.concatenate([np.asarray(traj["add_q"]) for traj in collection.trajectories()])
        eta = find_optimal_temperature(
            q_values,
            float(self.non_parametric_variational_eps),
            float(np.reshape(self.eta, (-1,))[0]),
            self.non_parametric_variational_lr,
            int(self.max_steps),
            stored,
        )
        self.eta = np.asarray(eta, dtype=np.float32)
        self.log.log({"eta": self.eta.reshape(-1)})

    def build(self, trajectory: Trajectory) -> NonParametricPolicy:
        """Un-normalised log weights ``Q / η`` for one trajectory.

        Args:
            trajectory: Trajectory carrying ``add_q``.

        Returns:
            A non-parametric policy over those samples.
        """
        return NonParametricPolicy(np.asarray(trajectory["add_q"]) / self.eta)
