from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import (
    PolicyOptimizationAlgorithm,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.typing import DTYPE


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """
    Implementation of Maximum A Posteriori Policy optimization (mpo), e.g. see the original paper for more
    information https://arxiv.org/abs/1806.06920. It estimates the q values using a critic based on retrace, which
    is not inherent to this algorithm, so it is not defined in this class. From these q values a variational
    distribution is fitted. This 'reweighted' policy is then used to fit the current iterated policy to the optimal
    one in a supervised fashion.
    """

    def __repr__(self) -> str:
        return "Maximum A Posteriori Policy Optimization"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new maximum a posteriori policy optimization algorithm.
        :param mc: The model configuration, containing all important elements, like env or similar.
        """
        PolicyOptimizationAlgorithm.__init__(self, mc)

        # create lagrange multiplier
        policy = self.mc.get("policy", target=False)
        num_lagrange_multipliers = policy.get_suff_stats_dim()

        # inner and outer learning rated, trust regions and batch size shortcuts
        self.outer_lr = self.conf["outer_lr"]
        self.inner_lr = self.conf["inner_lr"]
        self.trust_regions = self.conf["trust_regions"]
        self.split_objective = self.conf["coordinate_ascent_objective"]
        self.split_constraint = self.conf["coordinate_ascent_constraint"]
        self.cli = CliPrinter().instance
        self.l2_reg = self.conf.get("l2_reg", 0.0)

        self.num_params = 1 if not self.split_constraint else num_lagrange_multipliers
        self.fmtstr = "%10i | %10.5f" + self.num_params * " | %10.5f x %10.5f"
        self.titlestr = "%10s | %10s" + self.num_params * " | %23s"

    def init_state(self, policy: Any) -> dict[str, Any]:
        """Lagrange multipliers (log space, initialised to one like torch) and the persistent Adam states."""
        lagrange_mults = jnp.ones(self.num_params, DTYPE)
        return {
            "lagrange_mults": lagrange_mults,
            # optimizers are persistent over all steps, so adam can accumulate its moment estimates
            "opt_outer": optax.adam(self.outer_lr).init(eqx.filter(policy, eqx.is_inexact_array)),
            "opt_inner": optax.adam(self.inner_lr).init(lagrange_mults),
        }

    def sample_v(self, key: PRNGKeyArray) -> list[Any]:
        """Sample new trust regions.

        :return: the new sampled trust regions.
        """

        samp_trust_regions = self.trust_regions
        v: list[Any] = [None] * len(samp_trust_regions)
        for d in range(len(samp_trust_regions)):
            if isinstance(samp_trust_regions[d], list):
                lo, hi = samp_trust_regions[d]
                v[d] = jax.random.uniform(jax.random.fold_in(key, d), (), DTYPE, lo, hi)
            else:
                v[d] = samp_trust_regions[d]
        return v

    def shuffle_up_stats(self, state: ModelState, batch: Any, policy: Any) -> list[Any]:
        """Use this to create a list, where each sufficient statistic from the
        normal policy is mixed with the target ones, so the objective and
        constraint can be effectively splitted up.

        :param batch: The batch to use for obtaining the sufficient statistics
        :return:
        """
        # get target and correct policy
        t_states = jnp.asarray(batch.states, DTYPE)
        t_policy = state.get("policy", target=True)

        # target stats
        target_suff_stats = jax.lax.stop_gradient(t_policy.forward(t_states))
        suff_stats = policy.forward(t_states)
        l_suff_stats = [suff_stats]

        # for each dimension
        num_dims = target_suff_stats.shape[1]
        for d in range(num_dims):
            lst_els = []
            if d > 0:
                lst_els.append(target_suff_stats[:, :d])
            lst_els.append(suff_stats[:, d : d + 1])
            if d < num_dims - 1:
                lst_els.append(target_suff_stats[:, d + 1 :])
            l_suff_stats.append(jnp.concatenate(lst_els, axis=1))

        return l_suff_stats

    def main_obj(self, state: ModelState, batch: Any, l_suff_stats: list[Any], policy: Any) -> Any:
        """Obtain the objective based on the list of sufficient statistics.

        :param batch: The batch to use.
        :param l_suff_stats: A list of sufficient statistics.
        :return A list of constraints
        """

        # split objective if wanted
        if self.split_objective:
            # build the objective in a coordinate ascent fashion
            # iterate over the dimensions
            num_dims = len(l_suff_stats) - 1
            obj = jnp.zeros(())
            for d in range(num_dims):
                part_obj = self.__lagrange_obj(batch, l_suff_stats[d + 1], policy)
                obj = obj + part_obj

        else:
            obj = self.__lagrange_obj(batch, l_suff_stats[0], policy)

        return obj

    def main_constraints(
        self, state: ModelState, batch: Any, l_suff_stats: list[Any], policy: Any, v: list[Any]
    ) -> list[Any]:
        """Obtain all constraints using the list of sufficient statistics and the batch.

        :param batch: The batch to use.
        :param l_suff_stats: A list of sufficient statistics.
        :return A list of constraints
        """

        kl = "m-projection"

        # split constraint if wanted
        if self.split_constraint:
            num_dims = len(l_suff_stats) - 1
            constraints = []
            for d in range(num_dims):
                c = v[d] - policy.kl_divergence(state, batch, l_suff_stats[d + 1], kl, "mean")
                constraints.append(c)

        else:
            constraints = [(v[0] - policy.kl_divergence(state, batch, l_suff_stats[0], kl, "mean"))]

        return constraints

    def obj(self, state: ModelState, batch: Any, policy: Any, lagrange_mults: Any, v: list[Any]) -> Any:  # type: ignore[override]
        """
        Calculate the objective based on the objective and the splitted kl divergence constraints.
        :param batch: Batch to estimate objective and constraints
        :return: A un-detached version of the objective
        """

        l_suff_stats = self.shuffle_up_stats(state, batch, policy)
        obj = self.main_obj(state, batch, l_suff_stats, policy)
        constraints = self.main_constraints(state, batch, l_suff_stats, policy, v)
        for i, c in enumerate(constraints):
            obj = obj + jnp.exp(lagrange_mults[i]) * c

        return obj

    def perform_mult_steps(
        self, state: ModelState, key: PRNGKeyArray, num_steps: int, tc: Any, batch_size: int
    ) -> tuple[ModelState, dict[str, Any]]:
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        k_v, key = jax.random.split(key)
        v = self.sample_v(k_v)

        # perform a min step
        def body(carry: Any, s: Any) -> Any:
            state, key = carry
            key, sub = jax.random.split(key)
            batch = tc.sample(sub, batch_size, ["add_act", "grad_ranking", "add_pi"])
            state, metrics = self.perform_step(state, s, batch, v)
            return (state, key), metrics

        (state, _), metrics = jax.lax.scan(body, (state, key), jnp.arange(num_steps))
        return state, metrics

    def perform_step(self, state: ModelState, step: Any, batch: Any, v: list[Any]) -> tuple[ModelState, dict[str, Any]]:
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        # perform a max step on the policy, then a min step on the lagrange multipliers
        state = self.__max_step(state, batch, v)
        state, obj, constraints = self.__min_step(state, batch, v)

        # the logged objective (without the multiplier terms, as torch printed it) and constraints are the ones of the
        # min step (spec P4)
        return state, {"obj": obj, "lagrange_mults": state.get("lagrange_mults"), "constraints": jnp.stack(constraints)}

    def __lagrange_obj(self, batch: Any, suff_stats: Any, policy: Any) -> Any:
        """
        Calculated the reweighted objective needed for the min max problem. This is only partly
        the objective as the lagrange constraints are missing.
        :param batch: The batch to use for estimating the objective.
        :param suff_stats: A matrix containing for each element of the batch a vector of the
               sufficient statistics.
        :return: The calculated objective.
        """

        t_add_act = jnp.asarray(batch["add_act"], DTYPE)
        t_add_q = jnp.asarray(batch["grad_ranking"], DTYPE)

        # weighted maximum likelihood: E_s [ sum_a q(a|s) log pi(a|s) ], where q(a|s) is normalized per state
        ll = policy.log_prob(t_add_act, suff_stats=suff_stats)
        rew_vals = jnp.exp(t_add_q) * ll
        obj = jnp.mean(jnp.sum(rew_vals, axis=1))

        # the torch version regularised with the detached parameters (no gradient)
        w = jax.lax.stop_gradient(get_flat_params_from(policy))
        return obj - self.l2_reg * jnp.linalg.norm(w)

    def __max_step(self, state: ModelState, batch: Any, v: list[Any]) -> ModelState:
        """
        Maximizes the Lagrangian with respect to the policy parameters only (one Adam step on the negated objective).
        """

        policy = state.get("policy")
        lagrange_mults = state.get("lagrange_mults")
        opt = optax.adam(self.outer_lr)
        grads = eqx.filter_grad(lambda p: -self.obj(state, batch, p, lagrange_mults, v))(policy)
        updates, opt_state = opt.update(grads, state.get("opt_outer"), eqx.filter(policy, eqx.is_inexact_array))
        return state.set("policy", eqx.apply_updates(policy, updates)).set("opt_outer", opt_state)

    def __min_step(self, state: ModelState, batch: Any, v: list[Any]) -> tuple[ModelState, Any, list[Any]]:
        """
        Minimizes the Lagrangian with respect to the Lagrange multipliers only. They enter as exp(lm), so they
        stay positive.
        """

        policy = state.get("policy")
        opt = optax.adam(self.inner_lr)

        def objective(lagrange_mults: Any) -> Any:
            # same value as self.obj, but the objective without the multiplier terms is kept for logging
            l_suff_stats = self.shuffle_up_stats(state, batch, policy)
            main = self.main_obj(state, batch, l_suff_stats, policy)
            constraints = self.main_constraints(state, batch, l_suff_stats, policy, v)
            obj = main
            for i, c in enumerate(constraints):
                obj = obj + jnp.exp(lagrange_mults[i]) * c
            return obj, (main, constraints)

        (_, (obj, constraints)), grads = jax.value_and_grad(objective, has_aux=True)(state.get("lagrange_mults"))
        updates, opt_state = opt.update(grads, state.get("opt_inner"))
        state = state.set("lagrange_mults", state.get("lagrange_mults") + updates).set("opt_inner", opt_state)
        return state, obj, constraints
