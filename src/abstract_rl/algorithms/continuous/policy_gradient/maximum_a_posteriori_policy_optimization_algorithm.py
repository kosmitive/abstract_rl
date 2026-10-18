from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import temperature_dual
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from


def project_positive(raw: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained dual parameter to a strictly positive value.

    Args:
        raw: Unconstrained parameter.

    Returns:
        ``softplus(raw)`` clamped at ``1e-8``.
    """

    return F.softplus(raw).clamp(min=1e-8)


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, so a parameter starts at ``value`` after ``project_positive``.

    Args:
        value: Positive target value.

    Returns:
        The unconstrained pre-image.
    """

    return torch.log(torch.expm1(value))


def decoupled_kl_diagonal(
    online: torch.Tensor, target: torch.Tensor, per_dim: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoupled KL of a diagonal Gaussian, optionally one constraint per action dimension.

    Rows of ``online`` and ``target`` are ``[mean, std]``. The mean term pairs the online mean with the
    target standard deviation. The covariance term pairs the target mean with the online standard deviation.
    Both are ``KL(target || mixed)``.

    Args:
        online: Online sufficient statistics, shape ``[batch, 2 * action_dim]``.
        target: Target sufficient statistics, same shape.
        per_dim: Keep a separate constraint for each action dimension.

    Returns:
        Mean-term KL and covariance-term KL. Each is a scalar, or a vector of length ``action_dim`` when
        ``per_dim`` is set.
    """

    action_dim = online.shape[-1] // 2
    mu_online, std_online = online[..., :action_dim], online[..., action_dim:]
    mu_target, std_target = target[..., :action_dim], target[..., action_dim:]
    reference = Normal(mu_target, std_target)
    kl_mean = kl_divergence(reference, Normal(mu_online, std_target))
    kl_cov = kl_divergence(reference, Normal(mu_target, std_online))
    if per_dim:
        return kl_mean.mean(dim=0), kl_cov.mean(dim=0)
    return kl_mean.sum(dim=-1).mean(), kl_cov.sum(dim=-1).mean()


def decoupled_kl_full(
    mu_online: torch.Tensor,
    factor_online: torch.Tensor,
    mu_target: torch.Tensor,
    factor_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoupled KL of a full-covariance Gaussian, as one scalar per term.

    Both terms are the plain ``MultivariateNormal`` KL. The mean term swaps in the online mean and keeps
    the target factor. The covariance term swaps in the online factor and keeps the target mean.

    Args:
        mu_online: Online mean, shape ``[batch, action_dim]``.
        factor_online: Online Cholesky factor, shape ``[batch, action_dim, action_dim]``.
        mu_target: Target mean, shape ``[batch, action_dim]``.
        factor_target: Target Cholesky factor, shape ``[batch, action_dim, action_dim]``.

    Returns:
        Scalar mean-term KL and scalar covariance-term KL.
    """

    reference = MultivariateNormal(mu_target, scale_tril=factor_target)
    kl_mean = kl_divergence(reference, MultivariateNormal(mu_online, scale_tril=factor_target)).mean()
    kl_cov = kl_divergence(reference, MultivariateNormal(mu_target, scale_tril=factor_online)).mean()
    return kl_mean, kl_cov


def _constraint_value(spec: Any) -> float:
    """A fixed bound, or one draw from a ``[low, high]`` range."""

    if isinstance(spec, list | tuple):
        return float(np.random.uniform(float(spec[0]), float(spec[1])))
    return float(spec)


def _multiplier_shape(action_dim: int, cov_type: str, per_dim: bool) -> tuple[int, ...]:
    """Shape of one dual multiplier. A full covariance is always a single scalar."""

    match cov_type:
        case "full":
            return (1,)
        case _:
            return (action_dim,) if per_dim else (1,)


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """
    Maximum a posteriori policy optimization with a non-parametric E-step and a decoupled Gaussian M-step.

    Each update samples actions from the target policy, scores them with the target critic and reweights them with a
    per-state softmax of Q divided by eta. Eta and the mean / covariance multipliers descend their duals on the same
    step as the policy. The policy loss is the weighted log likelihood of those actions, with no gradient through the
    critic or the sampled actions. A diagonal covariance can constrain each action dimension on its own. A full
    covariance uses one scalar KL for the mean term and one for the covariance term.
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
        self.l2_reg = float(self.conf["l2_reg"])
        self.num_actions = int(self.conf.get_root("res_add_acts"))
        self.eps_eta = float(self.conf.get_root("rnk_eps"))
        self.per_dim = bool(self.conf["per_dim_constraint"])
        self.grad_clip = float(self.conf["grad_clip"])
        self.samples_per_insert = float(self.conf["samples_per_insert"])
        self._eps_mean_spec = self.conf["epsilon_mean"]
        self._eps_std_spec = self.conf["epsilon_stddev"]
        self.cov_type = str(policy.cov_type)
        self.action_dim = int(mc.env.action_dim)
        self.eps_mean = _constraint_value(self._eps_mean_spec)
        self.eps_std = _constraint_value(self._eps_std_spec)

        init_eta = float(self.conf.get_root("rnk_init_eta"))
        shape = _multiplier_shape(self.action_dim, self.cov_type, self.per_dim)
        raw_one = inverse_softplus(torch.tensor(1.0))
        self.raw_eta = torch.nn.Parameter(inverse_softplus(torch.tensor(init_eta)))
        self.raw_alpha_mean = torch.nn.Parameter(raw_one.expand(shape).clone())
        self.raw_alpha_std = torch.nn.Parameter(raw_one.expand(shape).clone())

        self.opt_policy = optim.Adam(policy.parameters(), lr=self.policy_lr)
        self.opt_dual = optim.Adam([self.raw_eta, self.raw_alpha_mean, self.raw_alpha_std], lr=self.dual_lr)
        self.cli = CliPrinter().instance

    def eta(self) -> torch.Tensor:
        """Temperature, softplus of the raw parameter clamped at 1e-8."""

        return project_positive(self.raw_eta)

    def alpha_mean(self) -> torch.Tensor:
        """Mean-constraint multiplier, softplus clamped at 1e-8."""

        return project_positive(self.raw_alpha_mean)

    def alpha_std(self) -> torch.Tensor:
        """Covariance-constraint multiplier, softplus clamped at 1e-8."""

        return project_positive(self.raw_alpha_std)

    def sample_constraints(self) -> None:
        """Draw this epoch's mean and covariance bounds when the config gives a range."""

        self.eps_mean = _constraint_value(self._eps_mean_spec)
        self.eps_std = _constraint_value(self._eps_std_spec)

    def num_updates(self, num_inserts: int, batch_size: int) -> int:
        """How many learner steps to take for the transitions inserted this epoch.

        :param num_inserts: Transitions added to the replay.
        :param batch_size: Transitions drawn per learner step.
        :return: ``samples_per_insert`` replay samples per inserted transition, at least one step.
        """

        return max(1, round(num_inserts * self.samples_per_insert / max(batch_size, 1)))

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

        eta = self.eta()
        weights = torch.softmax(q_values.detach() / eta.detach(), dim=1)
        online_stats = policy.forward(states)
        log_prob = policy.log_prob(actions, suff_stats=online_stats)
        likelihood = (weights * log_prob).sum(dim=1).mean()
        kl_mean, kl_cov = self._decoupled_kl(policy, online_stats, target_stats)
        alpha_mean, alpha_std = self.alpha_mean(), self.alpha_std()

        policy_loss = -likelihood
        policy_loss = policy_loss + (alpha_mean.detach() * kl_mean).sum()
        policy_loss = policy_loss + (alpha_std.detach() * kl_cov).sum()
        if self.l2_reg:
            policy_loss = policy_loss + self.l2_reg * torch.norm(get_flat_params_from(policy))

        dual_loss = temperature_dual(q_values.detach(), eta, self.eps_eta)
        dual_loss = dual_loss + (alpha_mean * (self.eps_mean - kl_mean.detach())).sum()
        dual_loss = dual_loss + (alpha_std * (self.eps_std - kl_cov.detach())).sum()

        self.opt_policy.zero_grad()
        self.opt_dual.zero_grad()
        policy_loss.backward()
        dual_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        self.opt_policy.step()
        self.opt_dual.step()

        if step % 10 == 0:
            self.cli.print(
                f"{step:6d} | ll {float(likelihood.detach()): .4f} | eta {float(eta.detach()): .4f}"
                f" | kl_mu {float(kl_mean.detach().mean()): .5f} | kl_std {float(kl_cov.detach().mean()): .6f}"
                f" | a_mu {float(alpha_mean.detach().mean()): .4f} | a_std {float(alpha_std.detach().mean()): .4f}"
            )

    def _decoupled_kl(
        self, policy: Any, online_stats: torch.Tensor, target_stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        match self.cov_type:
            case "full":
                mu_online, factor_online = policy._mu_factor(online_stats)
                mu_target, factor_target = policy._mu_factor(target_stats)
                return decoupled_kl_full(mu_online, factor_online, mu_target, factor_target)
            case _:
                return decoupled_kl_diagonal(online_stats, target_stats, self.per_dim)
