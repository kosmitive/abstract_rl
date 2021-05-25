from torch.nn import Module


class Policy(Module):
    """
    Represents a policy, e.g a gaussian policy or a eps greedy policy
    """

    def __init__(self, mc):
        Module.__init__(self)
        self.mc = mc

    def get_suff_stats_dim(self):
        """
        Obtain the dimension of sufficient statistics vector for the policy.
        :return: give dimension of suff stats vector.
        """
        raise NotImplementedError

    def forward(self, states):
        """
        Simply produces sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, get_suff_stats_dim()]
        """
        raise NotImplementedError

    def sample_actions(self, states=None, suff_stats=None, num_actions=1):
        """
        Can be used to obtain action samples for all states.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :param num_actions: The number of actions or K.
        :return: A matrix [D, K] containing action samples generated by the suff stats of the specific state.
        """
        raise NotImplementedError