import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import batch_mask, masked_mean, masked_reduce
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import (
    SIGMA_MIN,
    SQUASH_BOUND,
    SQUASH_U_MAX,
    normal_entropy,
    normal_kl,
    normal_log_prob,
)


def _mvn_log_prob(value: Array, loc: Array, scale_tril: Array) -> Array:
    """``torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(value)``."""
    diff = value - loc
    factor = jnp.broadcast_to(scale_tril, diff.shape + diff.shape[-1:])
    z = jax.scipy.linalg.solve_triangular(factor, diff[..., None], lower=True)[..., 0]
    half_log_det = jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    return -0.5 * (value.shape[-1] * math.log(2 * math.pi) + jnp.sum(z**2, axis=-1)) - half_log_det


def _mvn_kl(p_loc: Array, p_tril: Array, q_loc: Array, q_tril: Array) -> Array:
    """``torch.distributions.kl.kl_divergence(MultivariateNormal(p...), MultivariateNormal(q...))``."""
    n = p_loc.shape[-1]
    half_term1 = jnp.log(jnp.diagonal(q_tril, axis1=-2, axis2=-1)).sum(-1) - jnp.log(
        jnp.diagonal(p_tril, axis1=-2, axis2=-1)
    ).sum(-1)
    term2 = jnp.sum(jax.scipy.linalg.solve_triangular(q_tril, p_tril, lower=True) ** 2, axis=(-2, -1))
    term3 = jnp.sum(jax.scipy.linalg.solve_triangular(q_tril, (q_loc - p_loc)[..., None], lower=True)[..., 0] ** 2, -1)
    return half_term1 + 0.5 * (term2 + term3 - n)


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

    cov_type: str = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    num_output_net: int = eqx.field(static=True)
    squash: bool = eqx.field(static=True)
    action_center: tuple[float, ...] = eqx.field(static=True)
    action_scale: tuple[float, ...] = eqx.field(static=True)
    _sigma_is_softplus: bool = eqx.field(static=True)
    network: MultilayerPerceptron
    # log sigma of cov_type fixed: a constant, so it is static (torch buffer) and never a parameter
    _fixed_cov: tuple[float, ...] | None = eqx.field(static=True, default=None)
    # log sigma of cov_type single
    cov: Float[Array, "1 A"] | None = None

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray) -> None:
        """
        Initializes a new diagonal gaussian policy.
        :param mc: The model configuration to use.
        :param key: The random key of the network initialization.
        """
        assert isinstance(mc, ModelConfiguration)

        run_conf = mc.get("conf")
        self.cov_type = run_conf["cov_type"]
        self.action_dim = mc.env.action_dim
        a_dim = self.action_dim

        if self.cov_type == "state":
            self.num_output_net = 2 * a_dim
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            self._fixed_cov = (float(np.log(run_conf["sigma"])),) * a_dim
            self.num_output_net = a_dim
        elif self.cov_type == "single":
            # one state independent standard deviation per action dimension, stored in log space
            self.cov = jnp.full([1, a_dim], np.log(run_conf.get("sigma", 1.0)), DTYPE)
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
        # float32 like the torch buffers, kept as static tuples so they are constants and never parameters
        self.action_center = tuple(float(c) for c in center.astype(np.float32))
        self.action_scale = tuple(float(s) for s in scale.astype(np.float32))

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
        network = MultilayerPerceptron(
            mc, hidden_neurons=structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"], key=key
        )
        if self.cov_type in ("state", "full"):
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            layer = network.layers[-1]
            # covariance outputs are the bias alone, so sigma and L(s) = policy_sigma * I at every state
            weight = layer.weight.at[a_dim:].set(0.0)
            if self.cov_type == "state":
                bias = layer.bias.at[a_dim:].set(raw)
            else:
                diag = np.array([a_dim + i for i in self._diag_index()])
                bias = layer.bias.at[a_dim:].set(0.0).at[diag].set(raw)
            network = eqx.tree_at(lambda n: (n.layers[-1].weight, n.layers[-1].bias), network, (weight, bias))
        self.network = network

    @property
    def _center(self) -> Float[Array, "A"]:
        return jnp.asarray(self.action_center, DTYPE)

    @property
    def _scale(self) -> Float[Array, "A"]:
        return jnp.asarray(self.action_scale, DTYPE)

    def get_suff_stats_dim(self) -> int:
        """
        :return: 2 * A, the means followed by the standard deviations, or A + A(A+1)/2 for a full Cholesky factor.
        """
        a_dim = self.action_dim
        if self.cov_type == "full":
            return a_dim + a_dim * (a_dim + 1) // 2
        return 2 * a_dim

    def forward(self, states: Float[Array, "D Ds"]) -> Float[Array, "D S"]:
        """
        Produces the sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2A], ``[mu, sigma]`` with sigma strictly positive, or ``[mu, packed L]`` when
            ``cov_type`` is ``full``.
        """
        states = jnp.asarray(states, DTYPE)
        a_dim = self.action_dim
        if self.cov_type == "full":
            out = self.network.forward(states)
            mu, raw = out[:, :a_dim], out[:, a_dim:]
            return jnp.concatenate([mu, self._with_positive_diagonal(raw)], axis=1)
        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, :a_dim], out[:, a_dim:]
            # the configured activation of the sigma outputs already is a softplus: only guard against underflow,
            # otherwise the unconstrained output is made positive here (never both, a second softplus floors sigma)
            sigma = jnp.maximum(sigma, SIGMA_MIN) if self._sigma_is_softplus else jax.nn.softplus(sigma) + SIGMA_MIN
        else:
            mu = self.network.forward(states)
            log_sigma = self.cov if self.cov_type == "single" else jnp.asarray(self._fixed_cov, DTYPE)[None]
            sigma = jnp.broadcast_to(jnp.exp(log_sigma), mu.shape)
        return jnp.concatenate([mu, sigma], axis=1)

    def _diag_index(self) -> list[int]:
        """Row-major index of each diagonal entry inside a packed lower triangle."""
        return [i * (i + 3) // 2 for i in range(self.action_dim)]

    def _with_positive_diagonal(self, packed: Float[Array, "D L"]) -> Float[Array, "D L"]:
        """``softplus(raw) + SIGMA_MIN`` on the diagonal of a packed Cholesky factor. Off-diagonals stay raw."""
        idx = np.array(self._diag_index())
        diagonal = packed[:, idx]
        if self._sigma_is_softplus:
            diagonal = jnp.maximum(diagonal, SIGMA_MIN)
        else:
            diagonal = jax.nn.softplus(diagonal) + SIGMA_MIN
        return packed.at[:, idx].set(diagonal)

    def _mu_factor(self, suff_stats: Float[Array, "D S"]) -> tuple[Float[Array, "D A"], Float[Array, "D A A"]]:
        """Mean and the lower-triangular Cholesky factor ``L`` of a full-covariance statistic row."""
        a_dim = self.action_dim
        mu, packed = suff_stats[:, :a_dim], suff_stats[:, a_dim:]
        rows, cols = np.tril_indices(a_dim)
        factor = jnp.zeros((packed.shape[0], a_dim, a_dim), packed.dtype).at[:, rows, cols].set(packed)
        return mu, factor

    def _split(self, suff_stats: Float[Array, "D S"]) -> tuple[Float[Array, "D A"], Float[Array, "D _"]]:
        return suff_stats[:, : self.action_dim], suff_stats[:, self.action_dim :]

    def _suff_stats(self, states: Any, suff_stats: Any) -> Float[Array, "D S"]:
        assert (states is None) != (suff_stats is None), "pass either states or suff_stats"
        if suff_stats is None:
            assert states is not None
            return self.forward(states)
        return suff_stats

    def log_prob(self, actions: Any, states: Any = None, suff_stats: Any = None) -> Float[Array, "D K"]:
        """
        Log probability of actions relative to states.
        :param actions: A tensor of size [D, K, A], or [D, A] for one action per state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :return: A matrix [D, K] containing the log probability of each action.
        """
        stats = self._suff_stats(states, suff_stats)
        actions = jnp.asarray(actions, DTYPE)
        if self.cov_type == "full":
            return self._full_log_prob(actions, stats)
        mu, sigma = self._split(stats)
        if actions.ndim == 2:
            actions = actions[:, None]
        mu, sigma = mu[:, None], sigma[:, None]
        if not self.squash:
            return normal_log_prob(actions, mu, sigma).sum(axis=-1)

        # change of variables a = center + scale * tanh(u): log p(a) = log N(u) - sum log(scale * (1 - tanh(u)^2))
        y = jnp.clip((actions - self._center) / self._scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        log_jac = jnp.log(self._scale * (1 - y**2))
        return (normal_log_prob(u, mu, sigma) - log_jac).sum(axis=-1)

    def _full_log_prob(self, actions: Any, suff_stats: Float[Array, "D S"]) -> Float[Array, "D K"]:
        """Log density of ``N(mu, L L^T)``, with the same squash as the diagonal policy."""
        mu, factor = self._mu_factor(suff_stats)
        if actions.ndim == 2:
            actions = actions[:, None]
        mu, factor = mu[:, None], factor[:, None]
        if not self.squash:
            return _mvn_log_prob(actions, mu, factor)
        y = jnp.clip((actions - self._center) / self._scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        log_jac = jnp.log(self._scale * (1 - y**2)).sum(axis=-1)
        return _mvn_log_prob(u, mu, factor) - log_jac

    def mode(self, states: Any = None, suff_stats: Any = None) -> Float[Array, "D A"]:
        """
        :return: A matrix [D, A] with the most likely (mean) action of each state.
        """
        mu, _ = self._split(self._suff_stats(states, suff_stats))
        return self._center + self._scale * jnp.tanh(mu) if self.squash else mu

    def sample_actions(
        self, key: PRNGKeyArray, states: Any = None, suff_stats: Any = None, num_actions: int = 1
    ) -> Float[Array, "D K A"]:
        """
        Samples K actions for each state.
        :param key: The random key.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :param num_actions: The number of actions K.
        :return: A tensor [D, K, A] of (reparametrised) action samples.
        """
        stats = self._suff_stats(states, suff_stats)
        eps = jax.random.normal(key, (stats.shape[0], num_actions, self.action_dim), DTYPE)
        if self.cov_type == "full":
            mu, factor = self._mu_factor(stats)
            u = mu[:, None, :] + jnp.einsum("dij,dkj->dki", factor, eps)
        else:
            mu, sigma = self._split(stats)
            u = mu[:, None, :] + eps * sigma[:, None, :]
        if self.squash:
            u = self._center + self._scale * jnp.tanh(jnp.clip(u, -SQUASH_U_MAX, SQUASH_U_MAX))
        return u

    def clone(self) -> "DiagonalGaussianPolicy":
        """
        Clones the current instance. Modules are immutable, so the clone is the instance itself.
        :return: A new identical clones instance of the current.
        """
        return self

    def sync(self, v: "DiagonalGaussianPolicy") -> "DiagonalGaussianPolicy":
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        return self

    def entropy(self, state: ModelState | None, batch: Any) -> Float[Array, ""]:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        stats = self.forward(jnp.asarray(batch.states, DTYPE))
        mask = batch_mask(batch)
        if self.cov_type == "full":
            # 1/2 * A * log(2 pi e) + sum log diag(L), mean over the batch
            _, factor = self._mu_factor(stats)
            log_diag = jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=-1)
            return masked_mean(0.5 * self.action_dim * math.log(2 * math.pi * math.e) + log_diag, mask)
        _, sigma = self._split(stats)
        return masked_mean(normal_entropy(sigma).sum(axis=1), mask)

    def kl_divergence(
        self,
        state: ModelState,
        batch: Any,
        suff_stats: Any = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> Float[Array, ""]:
        """
        Analytic KL divergence between the gaussian of the current policy (or the passed sufficient statistics) and the
        target policy. A diagonal covariance sums the per-dimension KL. A full covariance is the KL of
        ``N(mu, L L^T)``, the total of the mean and covariance terms. TRPO constrains that KL, and MPO constrains the
        same total. For the tanh squashed policy this equals the KL of the underlying gaussians, as the (shared)
        squashing is a bijection.
        :param state: The traced model state (gives the target policy).
        :param batch: The batch containing the states.
        :param suff_stats: Optional sufficient statistics replacing the current policy.
        :param mode: i-projection computes KL(current, target), m-projection KL(target, current).
        :param est: Sum or mean over the states of the batch.
        """
        states = jnp.asarray(batch.states, DTYPE)
        mask = batch_mask(batch)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        t_stats = jax.lax.stop_gradient(state.get("policy", True).forward(states))
        if self.cov_type == "full":
            mu, factor = self._mu_factor(suff_stats)
            t_mu, t_factor = self._mu_factor(t_stats)
            if mode == "i-projection":
                kl_full = _mvn_kl(mu, factor, t_mu, t_factor)
            elif mode == "m-projection":
                kl_full = _mvn_kl(t_mu, t_factor, mu, factor)
            else:
                raise NotImplementedError(mode)
            return masked_reduce(kl_full, mask, est)
        mu, sigma = self._split(suff_stats)
        t_mu, t_sigma = self._split(t_stats)

        if mode == "i-projection":
            kl = normal_kl(mu, sigma, t_mu, t_sigma)
        elif mode == "m-projection":
            kl = normal_kl(t_mu, t_sigma, mu, sigma)
        else:
            raise NotImplementedError(mode)
        kl = kl.sum(axis=1)
        return masked_reduce(kl, mask, est)
