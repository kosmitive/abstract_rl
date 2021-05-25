import torch

from abstract_rl.src.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.src.algorithms.mixed.trust_region_gradient import TrustRegionGradient


class TRPOAlgorithm(PolicyOptimizationAlgorithm, TrustRegionGradient):
    """
    Represents a trust region policy optimization algorithm, where it uses v-trace and generalized advantage
    estimation for providing off policy advantages and then used these values to maximize the estimated objective,
    while full filling the kl constraint. See https://arxiv.org/abs/1502.05477 for more details.
    """

    def __init__(self, mc):
        PolicyOptimizationAlgorithm.__init__(self, mc)
        conf = mc.get('conf')

        # trpo vars
        trpo_eps = conf['tr_eps']
        trpo_cg_max_k = conf['cg_max_k']
        trpo_cg_residual_tol = conf['cg_res_tol']
        trpo_dvp_damping = conf['dvp_damping']
        self.policy = self.mc.get('policy', False)

        TrustRegionGradient.__init__(self, self.policy, trpo_eps, trpo_cg_max_k, trpo_cg_residual_tol, trpo_dvp_damping, 'max')

    def tr_distance(self, batch):
        """Pass back KL divergence for training divergence.

        :param batch: The batch to use.
        :return: The KL divergence.
        """
        return self.policy.kl_divergence(batch, est='mean', mode='m-projection')

    def tr_obj(self, batch):
        """Pass back the likelihood ratio gradient.

        :param batch: The batch to use.
        :return: The likelihood ratio gradient
        """
        return self.policy.likelihood_ratio(batch) #- self.trpo_entropy * self.policy.entropy(batch)
