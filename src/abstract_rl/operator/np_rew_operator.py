from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import TrajectoryOperator, store_rows
from abstract_rl.policy.non_parametric.non_parametric_variational_policy_builder import (
    NonParametricVariationalPolicyBuilder,
)


class VariationalRankingOperator(TrajectoryOperator):
    """
    Represents a variational reweighting operator see https://arxiv.org/abs/1806.06920.
    """

    def __repr__(self) -> str:
        return "reweighing"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new variational reweighting operator.
        :param mc: The model configuration to use.
        """
        self.conf = mc["conf"]
        self.mc = mc
        self.policy = mc.get("policy", True)
        self.num_add_actions = self.conf.get_root("res_add_acts")
        self.b = NonParametricVariationalPolicyBuilder(self.mc)

    def hook(self) -> None:
        self.b.calc_eta()

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """
        trajectory["grad_ranking"] = self.b.log_weights(trajectory["add_q"])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """
        Annotates all trajectories with the normalized log weights (grad_ranking) of their sampled actions, computed on
        the device over the concatenated q values.
        """
        if isinstance(trajectories, TrajectoryCollection):
            store_rows(trajectories, {"grad_ranking": self.b.log_weights(trajectories.annotation("add_q"))})
        else:
            for t in trajectories:
                self.transform(t)
