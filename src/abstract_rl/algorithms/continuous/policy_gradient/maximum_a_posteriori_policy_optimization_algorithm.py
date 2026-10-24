from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import (
    PolicyOptimizationAlgorithm,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.mixed import to_mx

# M-steps per compiled call of perform_mult_steps, also the interval of its log lines
M_STEPS_PER_CALL = 10


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """
    Implementation of Maximum A Posteriori Policy optimization (mpo), e.g. see the original paper for more
    information https://arxiv.org/abs/1806.06920. It estimates the q values using a critic based on retrace, which
    is not inherent to this algorithm, so it is not defined in this class. From these q values a variational
    distribution is fitted. This 'reweighted' policy is then used to fit the current iterated policy to the optimal
    one in a supervised fashion.

    One step of the M-step, the Adam step of the policy followed by the Adam step of the Lagrange multipliers, is one
    compiled MLX function. perform_mult_steps runs M_STEPS_PER_CALL of them, each gathering its batch from a flat
    replay on the device, in one compiled function.
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

        num_params = 1 if not self.split_constraint else num_lagrange_multipliers
        # the multipliers and the sampled trust regions live in trees, which the compiled step reads and writes
        self._lm_state: dict[str, list[mx.array]] = {"lm": [mx.ones(1) for _ in range(num_params)]}
        self._v_state: dict[str, list[mx.array]] = {"v": []}
        self.v = self.sample_v()

        # optimizers are persistent over all steps, so adam can accumulate its moment estimates (the reference
        # implementation corrects the bias of the moments, mlx only does that if asked to)
        self.opt_outer = optim.Adam(learning_rate=self.outer_lr, bias_correction=True)
        self.opt_outer.init(policy.trainable_parameters())
        self.opt_inner = optim.Adam(learning_rate=self.inner_lr, bias_correction=True)
        self.opt_inner.init(self._lm_state)
        self._compiled: dict[str, Callable] = {}

        self.fmtstr = "%10i | %10.5f" + num_params * " | %10.5f x %10.5f"
        self.titlestr = "%10s | %10s" + num_params * " | %23s"

    def sample_v(self) -> list[Any]:
        """Sample new trust regions.

        :return: the new sampled trust regions.
        """

        samp_trust_regions = self.trust_regions
        v: list[Any] = [None] * len(samp_trust_regions)
        for d in range(len(samp_trust_regions)):
            if isinstance(samp_trust_regions[d], list):
                v[d] = np.random.uniform(samp_trust_regions[d][0], samp_trust_regions[d][1])  # noqa: NPY002
            else:
                v[d] = samp_trust_regions[d]

        self.v = v
        self._v_state["v"] = [mx.array(float(el)) for el in v]
        return self.v

    @property
    def lagrange_mults(self) -> list[mx.array]:
        """The (log) Lagrange multipliers, they enter as exp(lm)."""
        return self._lm_state["lm"]

    def shuffle_up_stats(self, batch: Batch) -> list[mx.array]:
        """Use this to create a list, where each sufficient statistic from the
        normal policy is mixed with the target ones, so the objective and
        constraint can be effectively splitted up.

        :param batch: The batch to use for obtaining the sufficient statistics
        :return:
        """

        # get target and correct policy
        t_states = to_mx(batch.states)
        t_policy = self.mc.get("policy", target=True)
        policy = self.mc.get("policy", target=False)

        # target stats
        target_suff_stats = mx.stop_gradient(t_policy.forward(t_states))
        suff_stats = policy.forward(t_states)
        l_suff_stats = []
        l_suff_stats.append(suff_stats)

        # for each dimension
        num_dims = target_suff_stats.shape[1]
        for d in range(num_dims):
            lst_els = []
            if d > 0:
                lst_els.append(target_suff_stats[:, :d])
            lst_els.append(suff_stats[:, d : d + 1].reshape([-1, 1]))
            if d < num_dims - 1:
                lst_els.append(target_suff_stats[:, d + 1 :])
            l_suff_stats.append(mx.concatenate(lst_els, axis=1))

        return l_suff_stats

    def main_obj(self, batch: Batch, l_suff_stats: list[mx.array]) -> mx.array:
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
            obj = mx.zeros(1)
            for d in range(num_dims):
                part_obj = self.__lagrange_obj(batch, l_suff_stats[d + 1])
                obj = obj + part_obj

        else:
            obj = self.__lagrange_obj(batch, l_suff_stats[0])

        return obj

    def main_constraints(self, batch: Batch, l_suff_stats: list[mx.array]) -> list[mx.array]:
        """Obtain all constraints using the list of sufficient statistics and the batch.

        :param batch: The batch to use.
        :param l_suff_stats: A list of sufficient statistics.
        :return A list of constraints
        """

        kl = "m-projection"
        policy = self.mc.get("policy", target=False)

        # split constraint if wanted
        if self.split_constraint:
            num_dims = len(l_suff_stats) - 1
            constraints = []
            for d in range(num_dims):
                c = self._v_state["v"][d] - policy.kl_divergence(batch, l_suff_stats[d + 1], kl, "mean")
                constraints.append(c)

        else:
            constraints = [(self._v_state["v"][0] - policy.kl_divergence(batch, l_suff_stats[0], kl, "mean"))]

        return constraints

    def obj(self, batch: Batch) -> mx.array:  # type: ignore[override]
        """
        Calculate the objective based on the objective and the splitted kl divergence constraints.
        :param batch: Batch to estimate objective and constraints
        :return: A un-detached version of the objective
        """

        l_suff_stats = self.shuffle_up_stats(batch)
        obj = self.main_obj(batch, l_suff_stats)
        constraints = self.main_constraints(batch, l_suff_stats)
        for lm, c in zip(self.lagrange_mults, constraints, strict=False):
            obj = obj + mx.exp(lm) * c

        return obj

    def perform_mult_steps(self, num_steps: int, tc: TrajectoryCollection, batch_size: int) -> None:
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        self.sample_v()
        self.cli.empty().print(
            self.titlestr % ("k", "objective", *[f"lm_{i} x constr_{i}" for i in range(len(self.lagrange_mults))])
        )

        # the indices of all batches, drawn in the order and with the calls of one replay.sample per step
        replay = tc.flatten(["add_act", "grad_ranking"])
        indices = mx.array(np.stack([replay.draw(batch_size) for _ in range(num_steps)]), dtype=mx.int32)
        step_fn, log_fn = self._step_fns()
        logs = []
        for s in range(0, num_steps, M_STEPS_PER_CALL):
            # the first step of a chunk and its log values as in perform_step, then the others in one compiled call
            tree = replay.gather(indices[s])
            step_fn(tree)
            obj, *constraints = log_fn(tree)
            pairs = [v for pair in zip(self.lagrange_mults, constraints, strict=False) for v in pair]
            if s + 1 < min(s + M_STEPS_PER_CALL, num_steps):
                self._compiled["steps"](replay.tree, indices[s + 1 : s + M_STEPS_PER_CALL])
            mx.async_eval(
                obj, pairs, self.mc.get("policy", target=False).parameters(), self._lm_state, self.opt_outer.state
            )
            logs.append((s, obj, pairs))

        # the log lines of perform_step, printed once all steps are queued
        for s, obj, pairs in logs:
            self.cli.print(self.fmtstr % (s, obj.item(), *[v.item() for v in pairs]))

    def _step_fns(self) -> tuple[Callable, Callable]:
        """The compiled M-step and the compiled objective and constraints for the log line."""
        if not self._compiled:
            policy = self.mc.get("policy", target=False)
            t_policy = self.mc.get("policy", target=True)

            def step(tree: dict[str, Any]) -> None:
                batch = Batch.from_tree(tree)

                # max step: one adam step of the policy on the negated objective
                def neg_obj(params: Any) -> mx.array:
                    policy.update(params)
                    return -self.obj(batch).sum()

                params = policy.trainable_parameters()
                policy.update(self.opt_outer.apply_gradients(mx.grad(neg_obj)(params), params))

                # min step: one adam step of the lagrange multipliers with the updated policy
                def lm_obj(lm_state: dict[str, list[mx.array]]) -> mx.array:
                    self._lm_state["lm"] = lm_state["lm"]
                    return self.obj(batch).sum()

                lm_state = {"lm": list(self._lm_state["lm"])}
                lm_grads = mx.grad(lm_obj)(lm_state)
                self._lm_state["lm"] = self.opt_inner.apply_gradients(lm_grads, lm_state)["lm"]

            def log_values(tree: dict[str, Any]) -> list[mx.array]:
                batch = Batch.from_tree(tree)
                l_suff_stats = self.shuffle_up_stats(batch)
                obj = self.main_obj(batch, l_suff_stats)
                constraints = self.main_constraints(batch, l_suff_stats)
                return [obj, *constraints]

            state = [policy.state, self.opt_outer.state, self.opt_inner.state, self._lm_state]
            inputs = [*state, t_policy.state, self._v_state]
            self._compiled["step"] = mx.compile(step, inputs=inputs, outputs=state)
            self._compiled["log"] = mx.compile(log_values, inputs=inputs)

            def steps(tree: dict[str, Any], indices: mx.array) -> None:
                # one step per row of indices. The log values stay a separate call: in the same graph they share
                # subexpressions with the step, which changes its fused CPU kernels and the rounding.
                for j in range(indices.shape[0]):
                    step(tree_map(lambda v, j=j: mx.take(v, indices[j], axis=0), tree))

            self._compiled["steps"] = mx.compile(steps, inputs=inputs, outputs=state)
        return self._compiled["step"], self._compiled["log"]

    def perform_step(self, step: int, batch: Batch) -> None:
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        # perform a max step of the policy and a min step of the multipliers
        step_fn, log_fn = self._step_fns()
        tree = batch.to_tree(["add_act", "grad_ranking"])
        step_fn(tree)

        # objective and constraints after the step, only for the log line
        if step % 10 == 0:
            obj, *constraints = log_fn(tree)
            vals = [cast(float, v.item()) for pair in zip(self.lagrange_mults, constraints, strict=False) for v in pair]
            self.cli.print(self.fmtstr % (step, obj.item(), *vals))
        else:
            # the next batch is drawn while this step runs
            mx.async_eval(self.mc.get("policy", target=False).parameters(), self._lm_state, self.opt_outer.state)

    def __lagrange_obj(self, batch: Batch, suff_stats: mx.array) -> mx.array:
        """
        Calculated the reweighted objective needed for the min max problem. This is only partly
        the objective as the lagrange constraints are missing.
        :param batch: The batch to use for estimating the objective.
        :param suff_stats: A matrix containing for each element of the batch a vector of the
               sufficient statistics.
        :return: The calculated objective.
        """

        policy = self.mc.get("policy", target=False)

        t_add_act = to_mx(batch["add_act"])
        t_add_q = to_mx(batch["grad_ranking"])

        # weighted maximum likelihood: E_s [ sum_a q(a|s) log pi(a|s) ], where q(a|s) is normalized per state
        ll = policy.log_prob(t_add_act, suff_stats=suff_stats)
        rew_vals = mx.exp(t_add_q) * ll
        obj = mx.mean(mx.sum(rew_vals, axis=1))

        w = get_flat_params_from(policy)
        return obj - self.l2_reg * mx.linalg.norm(w)
