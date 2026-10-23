from collections.abc import Iterable

import mlx.core as mx

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.mixed import to_mx


class PolicyOptimizationAlgorithm:
    """
    Represents a policy optimization algorithm. It is splitted into two phases for the  scheme from the original
    papers of relative entropy policy search and maximum a posteriori policy optimization. Note that algorithms like
    trust region policy optimization or proximal policy optimization can be implemented efficiently.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new policy approximation algorithm. It makes the model config and from that also the running
        config available to the derived class itself.
        :param mc: The model configuration, containing all important elements, like env or similar.
        """
        assert isinstance(mc, ModelConfiguration)
        self.mc = mc
        self.conf = mc["conf"]

    def obj(self, trajectories: Iterable[Trajectory]) -> mx.array:
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        obj = mx.zeros(1)
        num_steps = 0
        for traj in trajectories:
            policy = self.mc.get("policy", False)

            # calculate the objective
            t_states = to_mx(traj.states)
            t_actions = to_mx(traj.actions)
            log_likelihoods = to_mx(traj.log_likelihoods)
            suff_stats = policy.forward(t_states)
            ll = policy.log_prob(t_actions, suff_stats=suff_stats)
            old_ll = log_likelihoods

            # calculate the ratio term
            a_vals = to_mx(traj["a"])
            r = mx.exp(ll - old_ll)
            obj = (num_steps * obj + mx.sum(r * a_vals)) / (num_steps + len(traj))
            num_steps += len(traj)

        return obj
