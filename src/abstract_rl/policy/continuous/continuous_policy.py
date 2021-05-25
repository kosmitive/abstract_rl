import torch
from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.src.policy.policy import Policy


class ContinuousPolicy(Policy, Syncable):
    """
    Represents a continuous policy, e.g a gaussian policy
    """

    def log_prob(self, actions, states=None, suff_stats=None):
        """
        Can be used to obtain log probability of actions relative to states.
        :param actions: A matrix of size [D, K] containing actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :return: A matrix [D, K] containing the log prob for each action in each row of
            actions matched to the row and the the suff stats of the specific state.
        """
        raise NotImplementedError

    def clone(self):
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """

        raise NotImplementedError

    def sync(self, v):
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        raise NotImplementedError

    def kl_divergence(self, batch, suff_stats=None, mode='i-projection', est='sum'):
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param batch:
        :return:
        """

        tp = self.mc.get('policy', True)

        # calc sufficient statistics
        states = torch.Tensor(batch.states)
        if suff_stats is None:
            suff_stats = self.forward(states)

        actions = torch.Tensor(batch.actions)
        log_likelihoods = tp.log_prob(actions, states).detach()
        new_lls = self.log_prob(actions, suff_stats=suff_stats)

        # use I projection
        if mode == 'm-projection':
            p = log_likelihoods
            q = new_lls

        # use M Projection
        elif mode == 'i-projection':
            p = new_lls
            q = log_likelihoods

        else: raise NotImplementedError

        # build kl terms adn sum up
        kl = torch.exp(p) * (p - q)

        if est == 'sum': kl_est = torch.sum
        elif est == 'mean':  kl_est = torch.mean
        else: raise NotImplementedError

        return kl_est(kl)

    def likelihood_ratio(self, batch):
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        policy = self.mc.get('policy', False)

        # calculate the objective
        states = torch.Tensor(batch.states)
        actions = torch.Tensor(batch.actions)
        suff_stats = policy.forward(states)
        ll = policy.log_prob(actions, suff_stats=suff_stats)
        old_ll = torch.Tensor(batch.log_likelihoods)

        # calculate the ratio term
        a_vals = torch.Tensor(batch['a'])
        mean_a = a_vals.mean()
        std_a = a_vals.std()
        advantages = (a_vals - mean_a) / std_a
        r = torch.exp(ll - old_ll)
        obj = torch.mean(r * advantages)
        return -obj
