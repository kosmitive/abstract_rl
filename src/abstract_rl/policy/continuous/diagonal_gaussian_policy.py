import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Parameter

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import SIGMA_MIN, SQUASH_BOUND, SQUASH_U_MAX


def head_activations(act_fn: Any, num_actions: int) -> tuple[list[Any], list[Any]]:
    """Resolve the activation functions of the policy head of a network with ``[mu_1..mu_A, sigma_1..sigma_A]`` outputs.

    The last entry of ``act_fn`` is either one activation name for all outputs, a pair ``[mu_act, sigma_act]`` which is
    expanded to one entry per output, or a list with one name per output. The activation of every output is applied by
    the network exactly once, the policy itself only adds a softplus to sigma if the network did not (see
    ``sigma_is_softplus``).
    :param act_fn: The per layer list of activation names of the network configuration.
    :param num_actions: The action dimension A.
    :return: The activation list for the network (last entry expanded) and the activation names of the sigma outputs.
    """
    act_fn = list(act_fn)
    last = act_fn[-1]
    if isinstance(last, list | tuple):
        last = list(last)
        if len(last) == 2 and num_actions > 1:
            last = [last[0]] * num_actions + [last[1]] * num_actions
        elif len(last) not in (2, 2 * num_actions):
            raise ValueError(f"the activations of the last layer need 2 or {2 * num_actions} entries, got {len(last)}")
        sigma_acts = last[len(last) // 2 :]
        if len(act_fn) > 1:
            act_fn[-1] = last
        else:
            # a single entry applies to every layer, so it can not be a per output list
            raise ValueError("per output activations require one entry per layer")
    else:
        sigma_acts = [last]
    return act_fn, sigma_acts


def sigma_is_softplus(sigma_acts: Sequence[Any]) -> bool:
    """Whether the network already applies a softplus to all sigma outputs, which makes them positive."""
    flags = {isinstance(a, str) and a.lower() == "softplus" for a in sigma_acts}
    if len(flags) != 1:
        raise ValueError("either all or none of the sigma outputs may use a softplus activation")
    return flags.pop()


class DiagonalGaussianPolicy(ContinuousPolicy):
    """
    Gaussian policy with a diagonal covariance for action spaces of any dimension A. The sufficient statistics are
    ``[mu_1..mu_A, sigma_1..sigma_A]``. Optionally the sample is squashed with a tanh into the per dimension bounds
    ``[low, high]`` of the action space, ``a = center + scale * tanh(u)`` with ``u ~ N(mu, sigma)``, and the density
    contains the matching log-determinant of the Jacobian.

    ``cov_type`` ``full`` replaces the standard deviations with a state-dependent Cholesky factor ``L(s)``. The network
    emits the mean and the lower triangle packed in row-major order. The diagonal of ``L`` is ``softplus(raw) +
    SIGMA_MIN`` and the off-diagonal entries stay raw. Samples are ``u = mu + L z`` with ``z ~ N(0, I)``. Log
    probability, entropy and the KL are those of ``N(mu, L L^T)``. The squash is the same componentwise map, and the
    KL stays the KL of the Gaussian on ``u``.

    Action shapes: samples and additional actions are ``[D, K, A]`` (K actions per state), a single action per state
    can also be passed as ``[D, A]``. Log probabilities are ``[D, K]``.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new diagonal gaussian policy.
        :param mc: The model configuration to use.
        """
        super().__init__(mc)
        assert isinstance(mc, ModelConfiguration)

        self.mc = mc
        run_conf = mc.get("conf")
        self.conf = run_conf
        self.cov_type = run_conf["cov_type"]
        self.cli = CliPrinter().instance
        self.action_dim = mc.env.action_dim
        a_dim = self.action_dim

        if self.cov_type == "state":
            self.num_output_net = 2 * a_dim
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            self.register_buffer("cov", torch.full([1, a_dim], float(np.log(run_conf["sigma"]))))
            self.num_output_net = a_dim
        elif self.cov_type == "single":
            # one state independent standard deviation per action dimension, stored in log space
            self.cov = Parameter(torch.full([1, a_dim], float(np.log(run_conf.get("sigma", 1.0)))))
            self.num_output_net = a_dim
        elif self.cov_type == "full":
            # mean and the packed lower-triangular Cholesky factor L(s)
            self.num_output_net = a_dim + a_dim * (a_dim + 1) // 2
        else:
            raise NotImplementedError(f"unknown covariance type {self.cov_type}")

        # optionally squash the gaussian with tanh into the action bounds of the environment
        self.squash = bool(run_conf.get("squash", False))
        low = np.asarray(mc.env.action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(mc.env.action_space.high, dtype=np.float64).reshape(-1)
        if self.squash and not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("squashing needs a bounded action space")
        center, scale = np.zeros(a_dim), np.ones(a_dim)
        if self.squash:
            center, scale = (high + low) / 2, (high - low) / 2
        self.action_center: torch.Tensor
        self.action_scale: torch.Tensor
        self.register_buffer("action_center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))

        # create policy network, the activations of the mu / sigma outputs are applied inside of the network.
        # a full covariance keeps one activation name per layer and softplus-es only diag L
        sd = mc.env.observation_dim
        structure = np.concatenate(([sd], run_conf["structure"], [self.num_output_net]))
        if self.cov_type == "full":
            af = list(run_conf["act_fn"])
            last = af[-1]
            if isinstance(last, list | tuple):
                raise ValueError(
                    "the full-covariance head uses one activation name; softplus is applied to the Cholesky diagonal"
                )
            sigma_acts: list[Any] = [last]
        else:
            af, sigma_acts = head_activations(run_conf["act_fn"], a_dim)
        self._sigma_is_softplus = sigma_is_softplus(sigma_acts)
        self.network = MultilayerPerceptron(
            mc, hidden_neurons=structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"]
        )
        self.add_module("nn", self.network)
        if self.cov_type in ("state", "full"):
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            with torch.no_grad():
                bias = self.network.layers[-1].bias
                assert bias is not None
                if self.cov_type == "state":
                    bias[a_dim:] = raw
                else:
                    bias[[a_dim + i for i in self._diag_index()]] = raw

    def get_suff_stats_dim(self) -> int:
        """
        :return: 2 * A, the means followed by the standard deviations, or A + A(A+1)/2 for a full Cholesky factor.
        """
        a_dim = self.action_dim
        if self.cov_type == "full":
            return a_dim + a_dim * (a_dim + 1) // 2
        return 2 * a_dim

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Produces the sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2A], ``[mu, sigma]`` with sigma strictly positive, or ``[mu, packed L]`` when
            ``cov_type`` is ``full``.
        """
        a_dim = self.action_dim
        if self.cov_type == "full":
            out = self.network.forward(states)
            mu, raw = out[:, :a_dim], out[:, a_dim:]
            return torch.cat([mu, self._with_positive_diagonal(raw)], dim=1)
        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, :a_dim], out[:, a_dim:]
            # the configured activation of the sigma outputs already is a softplus: only guard against underflow,
            # otherwise the unconstrained output is made positive here (never both, a second softplus floors sigma)
            sigma = torch.clamp(sigma, min=SIGMA_MIN) if self._sigma_is_softplus else F.softplus(sigma) + SIGMA_MIN
        else:
            mu = self.network.forward(states)
            sigma = torch.exp(self.cov).expand_as(mu)
        return torch.cat([mu, sigma], dim=1)

    def _diag_index(self) -> list[int]:
        """Row-major index of each diagonal entry inside a packed lower triangle."""
        return [i * (i + 3) // 2 for i in range(self.action_dim)]

    def _with_positive_diagonal(self, packed: torch.Tensor) -> torch.Tensor:
        """``softplus(raw) + SIGMA_MIN`` on the diagonal of a packed Cholesky factor. Off-diagonals stay raw."""
        idx = self._diag_index()
        diagonal = packed[:, idx]
        diagonal = torch.clamp(diagonal, min=SIGMA_MIN) if self._sigma_is_softplus else F.softplus(diagonal) + SIGMA_MIN
        index = torch.tensor(idx, device=packed.device).view(1, -1).expand(packed.shape[0], -1)
        return packed.scatter(1, index, diagonal)

    def _mu_factor(self, suff_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and the lower-triangular Cholesky factor ``L`` of a full-covariance statistic row."""
        a_dim = self.action_dim
        mu, packed = suff_stats[:, :a_dim], suff_stats[:, a_dim:]
        rows, cols = torch.tril_indices(a_dim, a_dim, device=suff_stats.device)
        flat_index = (rows * a_dim + cols).view(1, -1).expand(packed.shape[0], -1)
        flat = packed.new_zeros(packed.shape[0], a_dim * a_dim)
        factor = flat.scatter(1, flat_index, packed).view(packed.shape[0], a_dim, a_dim)
        return mu, factor

    def _split(self, suff_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return suff_stats[:, : self.action_dim], suff_stats[:, self.action_dim :]

    def _suff_stats(self, states: torch.Tensor | None, suff_stats: torch.Tensor | None) -> torch.Tensor:
        assert (states is None) != (suff_stats is None), "pass either states or suff_stats"
        if suff_stats is None:
            assert states is not None
            return self.forward(states)
        return suff_stats

    def log_prob(
        self, actions: torch.Tensor, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Log probability of actions relative to states.
        :param actions: A tensor of size [D, K, A], or [D, A] for one action per state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :return: A matrix [D, K] containing the log probability of each action.
        """
        stats = self._suff_stats(states, suff_stats)
        if self.cov_type == "full":
            return self._full_log_prob(actions, stats)
        mu, sigma = self._split(stats)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        mu, sigma = mu.unsqueeze(1), sigma.unsqueeze(1)
        if not self.squash:
            return Normal(mu, sigma).log_prob(actions).sum(dim=-1)

        # change of variables a = center + scale * tanh(u): log p(a) = log N(u) - sum log(scale * (1 - tanh(u)^2))
        y = torch.clamp((actions - self.action_center) / self.action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = torch.atanh(y)
        log_jac = torch.log(self.action_scale * (1 - y.pow(2)))
        return (Normal(mu, sigma).log_prob(u) - log_jac).sum(dim=-1)

    def _full_log_prob(self, actions: torch.Tensor, suff_stats: torch.Tensor) -> torch.Tensor:
        """Log density of ``N(mu, L L^T)``, with the same squash as the diagonal policy."""
        mu, factor = self._mu_factor(suff_stats)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        dist = MultivariateNormal(mu.unsqueeze(1), scale_tril=factor.unsqueeze(1))
        if not self.squash:
            return dist.log_prob(actions)
        y = torch.clamp((actions - self.action_center) / self.action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = torch.atanh(y)
        log_jac = torch.log(self.action_scale * (1 - y.pow(2))).sum(dim=-1)
        return dist.log_prob(u) - log_jac

    def mode(self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None) -> torch.Tensor:
        """
        :return: A matrix [D, A] with the most likely (mean) action of each state.
        """
        mu, _ = self._split(self._suff_stats(states, suff_stats))
        return self.action_center + self.action_scale * torch.tanh(mu) if self.squash else mu

    def sample_actions(
        self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None, num_actions: int = 1
    ) -> torch.Tensor:
        """
        Samples K actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :param num_actions: The number of actions K.
        :return: A tensor [D, K, A] of (reparametrised) action samples.
        """
        stats = self._suff_stats(states, suff_stats)
        if self.cov_type == "full":
            mu, factor = self._mu_factor(stats)
            u = MultivariateNormal(mu, scale_tril=factor).rsample((num_actions,)).transpose(0, 1)
        else:
            mu, sigma = self._split(stats)
            u = Normal(mu.unsqueeze(1), sigma.unsqueeze(1)).rsample([num_actions]).transpose(0, 1).squeeze(2)
        if self.squash:
            u = self.action_center + self.action_scale * torch.tanh(torch.clamp(u, -SQUASH_U_MAX, SQUASH_U_MAX))
        return u.contiguous()

    def clone(self) -> "DiagonalGaussianPolicy":
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """
        v = DiagonalGaussianPolicy(self.mc)
        self.sync(v)
        return v

    def sync(self, v: "DiagonalGaussianPolicy") -> "DiagonalGaussianPolicy":
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        v.load_state_dict(self.state_dict())
        return v

    def entropy(self, batch: Batch) -> torch.Tensor:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        stats = self.forward(torch.Tensor(batch.states))
        if self.cov_type == "full":
            # 1/2 * A * log(2 pi e) + sum log diag(L), mean over the batch
            _, factor = self._mu_factor(stats)
            log_diag = torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)
            return (0.5 * self.action_dim * math.log(2 * math.pi * math.e) + log_diag).mean()
        _, sigma = self._split(stats)
        return (0.5 * math.log(2 * math.pi * math.e) + torch.log(sigma)).sum(dim=1).mean()

    def kl_divergence(
        self,
        batch: Batch,
        suff_stats: torch.Tensor | None = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> torch.Tensor:
        """
        Analytic KL divergence between the gaussian of the current policy (or the passed sufficient statistics) and the
        target policy. A diagonal covariance sums the per-dimension KL. A full covariance is the KL of
        ``N(mu, L L^T)``, the total of the mean and covariance terms. TRPO constrains that KL, and MPO constrains the
        same total. For the tanh squashed policy this equals the KL of the underlying gaussians, as the (shared)
        squashing is a bijection.
        :param batch: The batch containing the states.
        :param suff_stats: Optional sufficient statistics replacing the current policy.
        :param mode: i-projection computes KL(current, target), m-projection KL(target, current).
        :param est: Sum or mean over the states of the batch.
        """
        states = torch.Tensor(batch.states)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        if self.cov_type == "full":
            mu, factor = self._mu_factor(suff_stats)
            with torch.no_grad():
                t_mu, t_factor = self._mu_factor(self.mc.get("policy", True).forward(states))
            current_full = MultivariateNormal(mu, scale_tril=factor)
            target_full = MultivariateNormal(t_mu, scale_tril=t_factor)
            if mode == "i-projection":
                kl = torch.distributions.kl.kl_divergence(current_full, target_full)
            elif mode == "m-projection":
                kl = torch.distributions.kl.kl_divergence(target_full, current_full)
            else:
                raise NotImplementedError(mode)
            if est == "sum":
                return kl.sum()
            elif est == "mean":
                return kl.mean()
            raise NotImplementedError(est)
        mu, sigma = self._split(suff_stats)
        with torch.no_grad():
            t_mu, t_sigma = self._split(self.mc.get("policy", True).forward(states))

        current, target = Normal(mu, sigma), Normal(t_mu, t_sigma)
        if mode == "i-projection":
            kl = torch.distributions.kl.kl_divergence(current, target)
        elif mode == "m-projection":
            kl = torch.distributions.kl.kl_divergence(target, current)
        else:
            raise NotImplementedError(mode)
        kl = kl.sum(dim=1)

        if est == "sum":
            return kl.sum()
        elif est == "mean":
            return kl.mean()
        raise NotImplementedError(est)
