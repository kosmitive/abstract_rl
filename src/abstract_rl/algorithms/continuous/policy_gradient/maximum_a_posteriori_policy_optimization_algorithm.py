from typing import Any

import numpy as np
import torch
import torch.optim as optim

from abstract_rl.algorithms.continuous.policy_gradient.mpo_loss import (
    ALPHA_FLOOR,
    ETA_FLOOR,
    decoupled_kl_diagonal,
    decoupled_kl_full,
    dual_descent_loss,
    dual_shapes,
    inverse_softplus,
    lagrangian_penalty,
    normalized_weights,
    project_positive,
    weighted_log_likelihood,
)
from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    find_optimal_temperature,
    temperature_dual,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy


def _constraint_value(spec: Any) -> float:
    """A fixed bound, or one draw from a ``[low, high]`` range."""

    if isinstance(spec, list | tuple):
        return float(np.random.uniform(float(spec[0]), float(spec[1])))
    return float(spec)


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """
    Maximum a posteriori policy optimization with a non-parametric E-step and a decoupled Gaussian M-step.

    Each update samples actions from the target policy, scores them with the target critic and reweights them with a
    per-state softmax of ``Q / eta``. ``eta`` and the mean / covariance multipliers are trained by descending their
    duals alongside the policy. The policy loss is the weighted log likelihood of those actions, with no gradient
    through the critic or the sampled actions. Set ``eta_solver`` to ``scipy`` to freeze ``eta`` at the L-BFGS dual
    solution instead of learning it by gradient steps.
    """

    def __repr__(self) -> str:
        return "Maximum A Posteriori Policy Optimization"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new maximum a posteriori policy optimization algorithm.
        :param mc: The model configuration, containing all important elements, like env or similar.
        """
        PolicyOptimizationAlgorithm.__init__(self, mc)

        policy = self.mc.get("policy", target=False)
        self.policy_lr = float(self.conf["outer_lr"])
        self.dual_lr = float(self.conf["inner_lr"])
        self.l2_reg = float(self.conf.get("l2_reg", 0.0))
        self.num_actions = int(self.conf.get_root("res_add_acts"))
        self.eps_eta = float(self.conf.get_root("rnk_eps"))
        self.eta_solver = str(self.conf.get("eta_solver", "dual"))
        self.per_dim = bool(self.conf.get("per_dim_constraint", True))
        grad_clip = self.conf.get("grad_clip", 40.0)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        ratio = self.conf.get("samples_per_insert", None)
        self.samples_per_insert = None if ratio is None else float(ratio)
        self.fallback_steps = int(self.conf.get_root("policy_opt_steps"))
        self._eps_mean_spec = self.conf.get("epsilon_mean", 5e-3)
        self._eps_std_spec = self.conf.get("epsilon_stddev", 1e-5)
        self.scipy_steps = int(self.conf.get_root("rnk_max_steps"))
        self.target_update = str(self.conf.get("target_update", "hard"))
        self.polyak_tau = float(self.conf.get("polyak_tau", 0.005))
        self.policy_period = int(self.conf.get_root("policy_sync_steps"))
        self.critic_period = int(self.conf.get_root("val_sync_steps"))
        self.cov_type = str(policy.cov_type)
        self.action_dim = int(mc.env.action_dim)
        self.learner_steps = 0
        self.eps_mean = _constraint_value(self._eps_mean_spec)
        self.eps_std = _constraint_value(self._eps_std_spec)

        init_eta = float(self.conf.get_root("rnk_init_eta"))
        mean_shape, std_shape = dual_shapes(self.action_dim, self.cov_type, self.per_dim)
        raw_one = inverse_softplus(torch.tensor(1.0))
        self.raw_eta = torch.nn.Parameter(inverse_softplus(torch.tensor(init_eta)))
        self.raw_alpha_mean = torch.nn.Parameter(raw_one.expand(mean_shape).clone())
        self.raw_alpha_std = torch.nn.Parameter(raw_one.expand(std_shape).clone())

        self.opt_policy = optim.Adam(policy.parameters(), lr=self.policy_lr)
        dual_params = [self.raw_alpha_mean, self.raw_alpha_std]
        if self.eta_solver == "dual":
            dual_params.append(self.raw_eta)
        self.opt_dual = optim.Adam(dual_params, lr=self.dual_lr)
        self.cli = CliPrinter().instance

    def eta(self) -> torch.Tensor:
        """Temperature ``eta = max(softplus(raw), 1e-8)``."""

        return project_positive(self.raw_eta, ETA_FLOOR)

    def alpha_mean(self) -> torch.Tensor:
        """Mean-constraint multiplier, ``softplus`` clamped at zero."""

        return project_positive(self.raw_alpha_mean, ALPHA_FLOOR)

    def alpha_std(self) -> torch.Tensor:
        """Covariance-constraint multiplier, ``softplus`` clamped at zero."""

        return project_positive(self.raw_alpha_std, ALPHA_FLOOR)

    def sample_constraints(self) -> None:
        """Draw this epoch's mean and covariance bounds when the config gives a range."""

        self.eps_mean = _constraint_value(self._eps_mean_spec)
        self.eps_std = _constraint_value(self._eps_std_spec)

    def set_eta(self, eta: float) -> None:
        """Freeze the temperature at a solved value (the scipy E-step).

        :param eta: The positive temperature to install.
        """

        raw = inverse_softplus(torch.tensor(max(float(eta), ETA_FLOOR)))
        with torch.no_grad():
            self.raw_eta.copy_(raw)

    def solve_eta_scipy(self, q_values: torch.Tensor) -> None:
        """Minimize the temperature dual with L-BFGS and install the solution.

        :param q_values: Q values of shape ``[batch, num_actions]``.
        """

        solved = find_optimal_temperature(
            q_values.detach(), self.eps_eta, float(self.eta().detach()), max_steps=self.scipy_steps
        )
        self.set_eta(float(solved.reshape(-1)[0]))

    def num_updates(self, num_inserts: int, batch_size: int) -> int:
        """How many learner steps to take for the transitions inserted this epoch.

        :param num_inserts: Transitions added to the replay.
        :param batch_size: Transitions drawn per learner step.
        :return: The step count. ``samples_per_insert`` wins over the fixed ``policy_opt_steps``.
        """

        if self.samples_per_insert is None:
            return self.fallback_steps
        return max(1, round(num_inserts * self.samples_per_insert / max(batch_size, 1)))

    def prepare_targets(self) -> bool:
        """Hard-copy or Polyak-average the target networks. Returns whether the critic target changed."""

        step = self.learner_steps
        critic_changed = False
        match self.target_update:
            case "polyak":
                self.mc.polyak_targets(self.polyak_tau)
                critic_changed = True
            case "hard":
                if step % self.policy_period == 0:
                    self.mc.copy_targets(["policy"])
                if step % self.critic_period == 0:
                    self.mc.copy_targets(["q_network"])
                    critic_changed = True
            case _:
                raise ValueError(f"unknown target update {self.target_update}")
        self.learner_steps = step + 1
        return critic_changed

    def perform_mult_steps(self, num_steps: int, tc: TrajectoryCollection, batch_size: int) -> None:
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately.
        """

        self.sample_constraints()
        for step in range(num_steps):
            batch = tc.sample(batch_size)
            self.perform_step(step, batch)

    def perform_step(self, step: int, batch: Batch) -> None:
        """One policy step and one dual step on a batch of states.

        Actions are drawn from the target policy and scored by the target critic. Neither the weights nor the target
        networks receive gradients.
        """

        states = torch.as_tensor(batch.states, dtype=torch.float32)
        policy = self.mc.get("policy", target=False)
        target_policy = self.mc.get("policy", target=True)
        target_q = self.mc.get("q_network", target=True)

        with torch.no_grad():
            target_stats = target_policy.forward(states)
            actions = target_policy.sample_actions(suff_stats=target_stats, num_actions=self.num_actions)
            q_values = target_q.q_val_multi(states, actions)

        eta = self.eta().detach() if self.eta_solver == "scipy" else self.eta()
        weights = normalized_weights(q_values, eta)
        online_stats = policy.forward(states)
        log_prob = policy.log_prob(actions, suff_stats=online_stats)
        likelihood = weighted_log_likelihood(log_prob, weights)
        kl_mean, kl_std = self._decoupled_kl(policy, online_stats, target_stats)

        policy_loss = -likelihood
        policy_loss = policy_loss + lagrangian_penalty(kl_mean, self.alpha_mean())
        policy_loss = policy_loss + lagrangian_penalty(kl_std, self.alpha_std())
        if self.l2_reg:
            policy_loss = policy_loss + self.l2_reg * torch.norm(get_flat_params_from(policy))

        match self.eta_solver:
            case "dual":
                temperature_loss = temperature_dual(q_values.detach(), self.eta(), self.eps_eta)
            case "scipy":
                temperature_loss = online_stats.new_zeros(())
            case _:
                raise ValueError(f"unknown eta solver {self.eta_solver}")
        dual_loss = temperature_loss + dual_descent_loss(kl_mean, self.alpha_mean(), self.eps_mean)
        dual_loss = dual_loss + dual_descent_loss(kl_std, self.alpha_std(), self.eps_std)

        self.opt_policy.zero_grad()
        self.opt_dual.zero_grad()
        policy_loss.backward()
        dual_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        self.opt_policy.step()
        self.opt_dual.step()

        if step % 10 == 0:
            self.cli.print(
                f"{step:6d} | ll {float(likelihood.detach()): .4f} | eta {float(self.eta().detach()): .4f}"
                f" | kl_mu {float(kl_mean.detach().mean()): .5f} | kl_std {float(kl_std.detach().mean()): .6f}"
                f" | a_mu {float(self.alpha_mean().detach().mean()): .4f}"
                f" | a_std {float(self.alpha_std().detach().mean()): .4f}"
            )

    def _decoupled_kl(
        self, policy: Any, online_stats: torch.Tensor, target_stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cov_type == "full":
            assert isinstance(policy, DiagonalGaussianPolicy)
            mu_online, factor_online = policy.cholesky(online_stats)
            mu_target, factor_target = policy.cholesky(target_stats)
            return decoupled_kl_full(mu_online, factor_online, mu_target, factor_target, self.per_dim)
        return decoupled_kl_diagonal(online_stats, target_stats, self.per_dim)
