import math
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.mixed import to_mx
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import (
    LOG_SQRT_2PI,
    ContinuousPolicy,
    normal_kl,
    normal_log_prob,
)
from abstract_rl.policy.continuous.host_actor import (
    FrozenHost,
    HostActor,
    host_actor,
)
from abstract_rl.policy.continuous.host_actor import (
    frozen_host as freeze_host_actor,
)
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

    ``cov_type`` ``full`` (a state-dependent Cholesky factor) is not available in the MLX port: MLX has no gradient
    of the Cholesky decomposition and of the triangular solve, and both only run on the CPU.

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
            # constant standard deviation, stored in log space and not trained (frozen below)
            self.cov = mx.full([1, a_dim], float(np.log(run_conf["sigma"])))
            self.num_output_net = a_dim
        elif self.cov_type == "single":
            # one state independent standard deviation per action dimension, stored in log space
            self.cov = mx.full([1, a_dim], float(np.log(run_conf.get("sigma", 1.0))))
            self.num_output_net = a_dim
        elif self.cov_type == "full":
            raise NotImplementedError(
                "cov_type full is not supported by the MLX port: mlx has no vjp for cholesky / triangular_solve "
                "and runs them on the CPU only, use policy_cov_type=state, single or fixed"
            )
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
        # constants, the underscore keeps them out of the parameter tree
        self._action_center = mx.array(center.astype(np.float32))
        self._action_scale = mx.array(scale.astype(np.float32))

        # create policy network, the activations of the mu / sigma outputs are applied inside of the network.
        sd = mc.env.observation_dim
        structure = np.concatenate(([sd], run_conf["structure"], [self.num_output_net]))
        af, sigma_acts = head_activations(run_conf["act_fn"], a_dim)
        self._sigma_is_softplus = sigma_is_softplus(sigma_acts)
        self.network = MultilayerPerceptron(
            mc, hidden_neurons=structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"]
        )
        if self.cov_type == "state":
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            layer = self.network.layers[-1]
            # covariance outputs are the bias alone, so sigma = policy_sigma at every state
            rows = mx.arange(layer.weight.shape[0])[:, None]
            layer.weight = mx.where(rows >= a_dim, 0.0, layer.weight)
            layer.bias = mx.where(mx.arange(layer.bias.shape[0]) >= a_dim, raw, layer.bias)
        if self.cov_type == "fixed":
            self.freeze(recurse=False, keys=["cov"])

    @property
    def action_center(self) -> mx.array:
        return self._action_center

    @property
    def action_scale(self) -> mx.array:
        return self._action_scale

    def get_suff_stats_dim(self) -> int:
        """
        :return: 2 * A, the means followed by the standard deviations.
        """
        return 2 * self.action_dim

    def forward(self, states: mx.array) -> mx.array:
        """
        Produces the sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2A], ``[mu, sigma]`` with sigma strictly positive.
        """
        a_dim = self.action_dim
        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, :a_dim], out[:, a_dim:]
            # the configured activation of the sigma outputs already is a softplus: only guard against underflow,
            # otherwise the unconstrained output is made positive here (never both, a second softplus floors sigma)
            sigma = mx.maximum(sigma, SIGMA_MIN) if self._sigma_is_softplus else nn.softplus(sigma) + SIGMA_MIN
        else:
            mu = self.network.forward(states)
            sigma = mx.broadcast_to(mx.exp(self.cov), mu.shape)
        return mx.concatenate([mu, sigma], axis=1)

    def _split(self, suff_stats: mx.array) -> tuple[mx.array, mx.array]:
        return suff_stats[:, : self.action_dim], suff_stats[:, self.action_dim :]

    def _suff_stats(self, states: mx.array | None, suff_stats: mx.array | None) -> mx.array:
        assert (states is None) != (suff_stats is None), "pass either states or suff_stats"
        if suff_stats is None:
            assert states is not None
            return self.forward(states)
        return suff_stats

    def log_prob(
        self, actions: mx.array, states: mx.array | None = None, suff_stats: mx.array | None = None
    ) -> mx.array:
        """
        Log probability of actions relative to states.
        :param actions: A tensor of size [D, K, A], or [D, A] for one action per state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :return: A matrix [D, K] containing the log probability of each action.
        """
        stats = self._suff_stats(states, suff_stats)
        mu, sigma = self._split(stats)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        mu, sigma = mu[:, None, :], sigma[:, None, :]
        if not self.squash:
            return normal_log_prob(actions, mu, sigma).sum(axis=-1)

        # change of variables a = center + scale * tanh(u): log p(a) = log N(u) - sum log(scale * (1 - tanh(u)^2))
        y = mx.clip((actions - self._action_center) / self._action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = mx.arctanh(y)
        log_jac = mx.log(self._action_scale * (1 - y**2))
        return (normal_log_prob(u, mu, sigma) - log_jac).sum(axis=-1)

    def mode(self, states: mx.array | None = None, suff_stats: mx.array | None = None) -> mx.array:
        """
        :return: A matrix [D, A] with the most likely (mean) action of each state.
        """
        mu, _ = self._split(self._suff_stats(states, suff_stats))
        return self._action_center + self._action_scale * mx.tanh(mu) if self.squash else mu

    def sample_noise(self, num_states: int, num_actions: int = 1) -> mx.array:
        """The standard normal noise of sample_actions for num_states states, a tensor [D, K, A]."""
        return mx.random.normal((num_states, num_actions, self.action_dim))

    def sample_actions(
        self,
        states: mx.array | None = None,
        suff_stats: mx.array | None = None,
        num_actions: int = 1,
        noise: mx.array | None = None,
    ) -> mx.array:
        """
        Samples K actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :param num_actions: The number of actions K.
        :param noise: The noise to use, see sample_noise. Drawn here if None.
        :return: A tensor [D, K, A] of (reparametrised) action samples.
        """
        stats = self._suff_stats(states, suff_stats)
        mu, sigma = self._split(stats)
        z = noise if noise is not None else self.sample_noise(stats.shape[0], num_actions)
        u = mu[:, None, :] + sigma[:, None, :] * z
        if self.squash:
            u = self._action_center + self._action_scale * mx.tanh(mx.clip(u, -SQUASH_U_MAX, SQUASH_U_MAX))
        return u

    def host_actor(self) -> HostActor | None:
        squash = (np.array(self._action_center), np.array(self._action_scale), SQUASH_U_MAX) if self.squash else None
        return host_actor(self, (1, 1, self.action_dim), squash)

    def frozen_host(self, key: tuple[int, int], num_actions: int) -> FrozenHost | None:
        """NumPy snapshot of this policy with ``num_actions`` of noise from ``key``, or None if unsupported.

        The returned actor does not read the live parameters or the global random key again.
        """
        squash = (np.array(self._action_center), np.array(self._action_scale), SQUASH_U_MAX) if self.squash else None
        return freeze_host_actor(self, (1, 1, self.action_dim), squash, key, num_actions)

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
        v.update(self.parameters())
        return v

    def entropy(self, batch: Batch) -> mx.array:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        stats = self.forward(to_mx(batch.states))
        _, sigma = self._split(stats)
        return (0.5 + LOG_SQRT_2PI + mx.log(sigma)).sum(axis=1).mean()

    def kl_divergence(
        self,
        batch: Batch,
        suff_stats: mx.array | None = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> mx.array:
        """
        Analytic KL divergence between the gaussian of the current policy (or the passed sufficient statistics) and the
        target policy, the sum of the per-dimension KL. For the tanh squashed policy this equals the KL of the
        underlying gaussians, as the (shared) squashing is a bijection.
        :param batch: The batch containing the states.
        :param suff_stats: Optional sufficient statistics replacing the current policy.
        :param mode: i-projection computes KL(current, target), m-projection KL(target, current).
        :param est: Sum or mean over the states of the batch.
        """
        states = to_mx(batch.states)
        if suff_stats is None:
            suff_stats = self.forward(states)
        mu, sigma = self._split(suff_stats)
        t_mu, t_sigma = self._split(mx.stop_gradient(self.mc.get("policy", True).forward(states)))

        if mode == "i-projection":
            kl = normal_kl(mu, sigma, t_mu, t_sigma)
        elif mode == "m-projection":
            kl = normal_kl(t_mu, t_sigma, mu, sigma)
        else:
            raise NotImplementedError(mode)
        kl = kl.sum(axis=1)

        if est == "sum":
            return kl.sum()
        elif est == "mean":
            return kl.mean()
        raise NotImplementedError(est)
