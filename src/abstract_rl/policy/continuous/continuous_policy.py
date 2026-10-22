from typing import Any

import jax
import jax.numpy as jnp

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import (
    batch_mask,
    masked_mean,
    masked_reduce,
    masked_std,
)
from abstract_rl.misc.typing import DTYPE
from abstract_rl.policy.policy import Policy


class ContinuousPolicy(Policy, Syncable):
    """
    Represents a continuous policy, e.g a gaussian policy
    """

    def log_prob(self, actions: Any, states: Any = None, suff_stats: Any = None) -> Any:
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

    def mode(self, states: Any = None, suff_stats: Any = None) -> Any:
        """
        The most likely action of each state.
        :return: A matrix [D, A].
        """
        raise NotImplementedError

    def kl_divergence(
        self,
        state: ModelState,
        batch: Any,
        suff_stats: Any = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> Any:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param state: The traced model state (gives the target policy).
        :param batch:
        :return:
        """

        tp = state.get("policy", True)

        # calc sufficient statistics
        states = jnp.asarray(batch.states, DTYPE)
        if suff_stats is None:
            suff_stats = self.forward(states)

        actions = jnp.asarray(batch.actions, DTYPE)
        log_likelihoods = jax.lax.stop_gradient(tp.log_prob(actions, states))
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
        kl = jnp.exp(p) * (p - q)
        return masked_reduce(kl, batch_mask(batch), est)

    def likelihood_ratio(self, state: ModelState, batch: Any) -> Any:
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param state: The traced model state (gives the current policy).
            The policy is read from ``state`` (like torch ``mc.get``), so differentiate w.r.t.
            ``state.set("policy", p)``, not ``self``.
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        policy = state.get("policy", False)

        # calculate the objective
        states = jnp.asarray(batch.states, DTYPE)
        actions = jnp.asarray(batch.actions, DTYPE)
        suff_stats = policy.forward(states)
        ll = policy.log_prob(actions, suff_stats=suff_stats).reshape(-1)
        old_ll = jnp.asarray(batch.log_likelihoods, DTYPE).reshape(-1)

        # calculate the ratio term
        a_vals = jnp.asarray(batch["a"], DTYPE)
        mean_a = masked_mean(a_vals, batch_mask(batch))
        std_a = masked_std(a_vals, batch_mask(batch))
        advantages = (a_vals - mean_a) / (std_a + 1e-8)
        r = jnp.exp(ll - old_ll)
        obj = masked_mean(r * advantages, batch_mask(batch))
        return -obj
