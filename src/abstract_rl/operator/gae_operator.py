import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
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
        self.v_net = mc.get("v_network", True)

        # get discount
        self.env = mc["env"]
        self.gae_lambda = np.float32(self.conf["lambda"])

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator. The backward recursion is
        sequential and runs on the host in float32.
        :param trajectory: trajectory to transform.
        """

        tl = len(trajectory)

        # obtain rewards
        discount = self.env.discount()

        # use v approximation to get advantages
        td_errors = np.asarray(trajectory["td"], dtype=np.float32).reshape(-1)
        mask = 1 - np.asarray(trajectory.dones, dtype=np.float32).reshape(-1)

        # reserve space for advantages
        a = np.zeros(tl + 1, dtype=np.float32)

        # get combined ratio
        gae_gamma_lambda = np.float32(discount * self.gae_lambda)

        for t in reversed(range(tl)):
            a[t] = td_errors[t] + mask[t] * gae_gamma_lambda * a[t + 1]

        trajectory["a"] = a[:-1]
