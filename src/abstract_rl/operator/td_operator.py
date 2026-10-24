import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.mixed import to_mx
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
        self.v_net = mc.get("v_network", True)

        # get discount
        self.env = mc["env"]

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator. The values come from the
        (MLX) value network, the sequential recursion of the targets runs on the host in float32.
        :param trajectory: trajectory to transform.
        """

        # obtain rewards
        discount = np.float32(self.env.discount())
        rewards = np.asarray(trajectory.rewards, dtype=np.float32).reshape(-1)
        mask = 1 - np.asarray(trajectory.dones, dtype=np.float32).reshape(-1)

        # values of all states and the last observed state, which bootstraps truncated (e.g. time limit) trajectories
        # instead of treating the end of the trajectory as terminal
        t_states = to_mx(trajectory.all_states)
        all_v = np.array(self.v_net.forward(t_states)).reshape(-1)
        v = all_v[:-1]
        last_v = all_v[-1] * mask[-1]

        tl = len(trajectory)
        tv = np.zeros(tl + 1, dtype=np.float32)
        tv[tl] = last_v
        for t in reversed(range(tl)):
            tv[t] = rewards[t] + discount * mask[t] * tv[t + 1]

        # the value of the following state, the last one is bootstrapped
        next_v = np.concatenate([v[1:], [last_v]]).astype(np.float32)
        td_errors = rewards + discount * next_v * mask - v

        trajectory["tv"] = tv[:-1]
        trajectory["td"] = td_errors.reshape(-1, 1)
