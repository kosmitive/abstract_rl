import math
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.mixed import to_mx
from abstract_rl.policy.policy import Policy

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def normal_log_prob(x: mx.array, mu: mx.array, sigma: mx.array) -> mx.array:
    """Elementwise log density of ``N(mu, sigma^2)``, as ``torch.distributions.Normal.log_prob``."""
    return -((x - mu) ** 2) / (2 * sigma**2) - mx.log(sigma) - LOG_SQRT_2PI


def normal_kl(p_mu: mx.array, p_sigma: mx.array, q_mu: mx.array, q_sigma: mx.array) -> mx.array:
    """Elementwise ``KL(N(p_mu, p_sigma^2) || N(q_mu, q_sigma^2))``, as ``torch.distributions.kl.kl_divergence``."""
    var_ratio = (p_sigma / q_sigma) ** 2
    t1 = ((p_mu - q_mu) / q_sigma) ** 2
    return 0.5 * (var_ratio + t1 - 1 - mx.log(var_ratio))


class ContinuousPolicy(Policy, Syncable):
    """
    Represents a continuous policy, e.g a gaussian policy
    """

    def log_prob(
        self, actions: mx.array, states: mx.array | None = None, suff_stats: mx.array | None = None
    ) -> mx.array:
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

    def clone(self) -> "ContinuousPolicy":
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """

        raise NotImplementedError

    def sync(self, v: Any) -> Any:
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        raise NotImplementedError

    def mode(self, states: mx.array | None = None, suff_stats: mx.array | None = None) -> mx.array:
        """
        The most likely action of each state.
        :return: A matrix [D, A].
        """
        raise NotImplementedError

    def act(self, states: mx.array, opt: bool = False) -> tuple[mx.array, mx.array]:
        """
        Action and its log likelihood for a batch of states in one compiled call, used by the rollout. The parameters
        and the random state are inputs of the compiled function, so it stays valid after every update.
        :param states: A matrix of size [D, Ds].
        :param opt: Use the mode instead of a sample.
        :return: The actions and their log likelihoods.
        """
        fns: dict[bool, Callable] = self.__dict__.setdefault("_act_fns", {})
        if opt not in fns:

            def fn(s: mx.array) -> tuple[mx.array, mx.array]:
                suff_stats = self.forward(s)
                a = self.mode(suff_stats=suff_stats) if opt else self.sample_actions(suff_stats=suff_stats)
                return a, self.log_prob(a, suff_stats=suff_stats)

            state = [self.state, mx.random.state]
            fns[opt] = mx.compile(fn, inputs=state, outputs=[mx.random.state])
        return fns[opt](states)

    def kl_divergence(
        self, batch: Batch, suff_stats: mx.array | None = None, mode: str = "i-projection", est: str = "sum"
    ) -> mx.array:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param batch:
        :return:
        """

        tp = self.mc.get("policy", True)

        # calc sufficient statistics
        states = to_mx(batch.states)
        if suff_stats is None:
            suff_stats = self.forward(states)

        actions = to_mx(batch.actions)
        log_likelihoods = mx.stop_gradient(tp.log_prob(actions, states))
        new_lls = self.log_prob(actions, suff_stats=suff_stats)

        # use I projection
        if mode == "m-projection":
            p = log_likelihoods
            q = new_lls

        # use M Projection
        elif mode == "i-projection":
            p = new_lls
            q = log_likelihoods

        else:
            raise NotImplementedError

        # build kl terms adn sum up
        kl = mx.exp(p) * (p - q)

        if est == "sum":
            return mx.sum(kl)
        elif est == "mean":
            return mx.mean(kl)
        raise NotImplementedError

    def likelihood_ratio(self, batch: Batch) -> mx.array:
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        policy = self.mc.get("policy", False)

        # calculate the objective
        states = to_mx(batch.states)
        actions = to_mx(batch.actions)
        suff_stats = policy.forward(states)
        ll = policy.log_prob(actions, suff_stats=suff_stats).reshape(-1)
        old_ll = to_mx(batch.log_likelihoods).reshape(-1)

        # calculate the ratio term (unbiased standard deviation as torch.std)
        a_vals = to_mx(batch["a"]).reshape(-1)
        mean_a = a_vals.mean()
        std_a = mx.sqrt(mx.sum((a_vals - mean_a) ** 2) / (a_vals.size - 1))
        advantages = (a_vals - mean_a) / (std_a + 1e-8)
        r = mx.exp(ll - old_ll)
        obj = mx.mean(r * advantages)
        return -obj
