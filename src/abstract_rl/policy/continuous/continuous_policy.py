from collections.abc import Callable
from typing import Any

import torch

from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.mixed import compile_attr, torch_compile_enabled
from abstract_rl.policy.policy import Policy


class ContinuousPolicy(Policy, Syncable):
    """
    Represents a continuous policy, e.g a gaussian policy
    """

    # eager forward kept for the KL, which TRPO double-backwards through
    _forward: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, mc: Any) -> None:
        """Save an eager forward. ``torch_compile`` compiles the public forward and the surrogate."""
        super().__init__(mc)
        # the Fisher-vector product double-backwards through the KL, which torch.compile cannot do
        self._forward = self.forward
        if not torch_compile_enabled(mc.get("conf")):
            return
        # rollout batches have length 1 and the update does not, so the batch dim stays dynamic
        compile_attr(self, "forward", dynamic=True)
        # the surrogate sees one training batch size
        compile_attr(self, "likelihood_ratio", dynamic=False)

    def log_prob(
        self, actions: torch.Tensor, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def mode(self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None) -> torch.Tensor:
        """
        The most likely action of each state.
        :return: A matrix [D, A].
        """
        raise NotImplementedError

    def kl_divergence(
        self,
        batch: Batch,
        suff_stats: torch.Tensor | None = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> torch.Tensor:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param batch:
        :return:
        """

        tp = self.mc.get("policy", True)

        # calc sufficient statistics
        states = torch.Tensor(batch.states)
        if suff_stats is None:
            suff_stats = self._forward(states)

        actions = torch.Tensor(batch.actions)
        log_likelihoods = tp.log_prob(actions, states).detach()
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
        kl = torch.exp(p) * (p - q)

        if est == "sum":
            kl_est = torch.sum
        elif est == "mean":
            kl_est = torch.mean
        else:
            raise NotImplementedError

        return kl_est(kl)

    def likelihood_ratio(self, batch: Batch) -> torch.Tensor:
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        policy = self.mc.get("policy", False)

        # calculate the objective
        states = torch.as_tensor(batch.states, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        suff_stats = policy._forward(states)
        ll = policy.log_prob(actions, suff_stats=suff_stats).view(-1)
        old_ll = torch.as_tensor(batch.log_likelihoods, dtype=torch.float32).view(-1)

        # calculate the ratio term (all terms are flat vectors of length D)
        a_vals = torch.as_tensor(batch["a"], dtype=torch.float32).view(-1)
        mean_a = a_vals.mean()
        std_a = a_vals.std()
        advantages = (a_vals - mean_a) / (std_a + 1e-8)
        r = torch.exp(ll - old_ll)
        obj = torch.mean(r * advantages)
        return -obj
