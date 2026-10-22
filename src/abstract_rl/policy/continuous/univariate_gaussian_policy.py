import math
from typing import Any, cast

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

# pre-squash samples are limited to |u| <= 5, so tanh stays invertible in float32
SQUASH_U_MAX = 5.0
SQUASH_BOUND = math.tanh(SQUASH_U_MAX)
SIGMA_MIN = 1e-6


def normal_log_prob(value: Array, loc: Array, scale: Array) -> Array:
    """``torch.distributions.Normal(loc, scale).log_prob(value)``."""
    return -((value - loc) ** 2) / (2 * scale**2) - jnp.log(scale) - math.log(math.sqrt(2 * math.pi))


def normal_kl(p_loc: Array, p_scale: Array, q_loc: Array, q_scale: Array) -> Array:
    """``torch.distributions.kl.kl_divergence(Normal(p_loc, p_scale), Normal(q_loc, q_scale))``."""
    var_ratio = (p_scale / q_scale) ** 2
    t1 = ((p_loc - q_loc) / q_scale) ** 2
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))


def normal_entropy(scale: Array) -> Array:
    """``torch.distributions.Normal(loc, scale).entropy()``."""
    return 0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(scale)


def _sigma_head_is_softplus(act_fn: list[str | list[str]]) -> bool:
    """Whether the configured activation of the last network layer already applies a softplus to the sigma output.

    ``act_fn`` is the per layer list of the network configuration, an entry of the last layer is either one activation
    name for all outputs or a list with one name per output, e.g. ``["tanh", "tanh", ["none", "softplus"]]`` (outputs
    are ``[mu, sigma]``).
    """
    last = act_fn[-1] if len(act_fn) > 1 else act_fn[0]
    if isinstance(last, (list, tuple)):
        last = last[1] if len(last) > 1 else last[0]
    return isinstance(last, str) and last.lower() == "softplus"


class UnivariateGaussianPolicy(ContinuousPolicy):
    """
    Represents a univariate gaussian policy. This only supports one action dimension.
    """

    cov_type: str = eqx.field(static=True)
    num_output_net: int = eqx.field(static=True)
    squash: bool = eqx.field(static=True)
    action_scale: float = eqx.field(static=True)
    _sigma_is_softplus: bool = eqx.field(static=True)
    network: MultilayerPerceptron
    # log sigma of cov_type fixed: a constant, so it is static (torch buffer) and never a parameter
    _fixed_cov: float | None = eqx.field(static=True, default=None)
    # log sigma of cov_type single: the only parameter besides the network
    cov: Float[Array, "1 1"] | None = None

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray) -> None:
        """
        Initializes a new uni variate gaussian policy.
        :param mc: The model configuration to use.
        :param key: The random key of the network initialization.
        """
        assert isinstance(mc, ModelConfiguration)

        run_conf = mc.get("conf")
        self.cov_type = run_conf["cov_type"]

        if self.cov_type == "state":
            self.num_output_net = 2
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            init_log_sigma = np.log(run_conf["sigma"])
            self._fixed_cov = float(init_log_sigma)
            self.num_output_net = 1
        elif self.cov_type == "single":
            # single state independent standard deviation, stored in log space
            init_log_sigma = np.log(run_conf.get("sigma", 1.0))
            self.cov = jnp.full([1, 1], init_log_sigma, DTYPE)
            self.num_output_net = 1
        else:
            raise NotImplementedError(f"unknown covariance type {self.cov_type}")

        # optionally squash the gaussian with tanh into the (symmetric) action bounds of the environment, so the
        # policy and the q function only ever see actions which the environment actually executes
        self.squash = bool(run_conf.get("squash", False))
        self.action_scale = float(np.max(np.abs(mc.env.action_space.high)))

        # create policy network
        sd = mc.env.observation_dim
        policy_net_structure: np.ndarray = np.concatenate(
            cast(Any, ([sd], run_conf["structure"], [self.num_output_net]))
        )

        af = run_conf["act_fn"]
        self._sigma_is_softplus = _sigma_head_is_softplus(af)
        network = MultilayerPerceptron(
            mc,
            hidden_neurons=policy_net_structure,
            act_fn=af,
            layer_norm=run_conf["layer_norm"],
            init=run_conf["init"],
            key=key,
        )
        if self.cov_type == "state":
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            bias = network.layers[-1].bias
            network = eqx.tree_at(lambda n: n.layers[-1].bias, network, bias.at[1].set(raw))
        self.network = network

    @property
    def cov_variable(self) -> Float[Array, "1 1"]:
        """The log standard deviation of the single and fixed covariance types."""
        return self.cov if self.cov_type == "single" else jnp.full([1, 1], self._fixed_cov, DTYPE)

    def get_suff_stats_dim(self) -> int:
        """
        Obtain the dimension of sufficient statistics vector for a univariate gaussian.
        :return: depending on the configuration 1 for mean-only and 2 for mean and covariance.
        """

        return 2

    def forward(self, states: Float[np.ndarray | Array, "D Ds"]) -> Float[Array, "D 2"]:
        """
        Simply produces sufficient statistics of the univariate gaussian policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2]
        """
        states = jnp.asarray(states, DTYPE)

        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, 0:1], out[:, 1:2]
            # if the activation function of the network already is a softplus on the sigma head, only guard against an
            # underflow to 0, otherwise the unconstrained output is made strictly positive with a softplus
            sigma = jnp.maximum(sigma, SIGMA_MIN) if self._sigma_is_softplus else jax.nn.softplus(sigma) + SIGMA_MIN
            return jnp.concatenate([mu, sigma], 1)

        else:
            mu = self.network.forward(states)
            fw = jnp.exp(self.cov_variable)
            sigma = jnp.broadcast_to(fw, mu.shape)
            return jnp.concatenate([mu, sigma], 1)

    def _suff_stats(self, states: Any, suff_stats: Any) -> Float[Array, "D 2"]:
        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        return suff_stats

    def log_prob(
        self, actions: Float[np.ndarray | Array, "D K"], states: Any = None, suff_stats: Any = None
    ) -> Float[Array, "D K"]:
        """
        Can be used to obtain log probability of actions relative to states.
        :param actions: A matrix of size [D, K] containing actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :return: A matrix [D, K] containing the log prob for each action in each row of
            actions matched to the row and the the suff stats of the specific state.
        """
        suff_stats = self._suff_stats(states, suff_stats)
        actions = jnp.asarray(actions, DTYPE)

        # extract mu and sigma depending on settings
        mu, sigma = suff_stats[:, 0:1], suff_stats[:, 1:2]
        if not self.squash:
            return normal_log_prob(actions, mu, sigma)

        # change of variables a = c * tanh(u): log p(a) = log N(u) - log(c * (1 - tanh(u)^2))
        y = jnp.clip(actions / self.action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        ll = normal_log_prob(u, mu, sigma) - jnp.log(self.action_scale * (1 - y**2))
        return ll

    def mode(self, states: Any = None, suff_stats: Any = None) -> Float[Array, "D 1"]:
        suff_stats = self._suff_stats(states, suff_stats)
        mu = suff_stats[:, 0:1]
        return self.action_scale * jnp.tanh(mu) if self.squash else mu

    def sample_actions(
        self, key: PRNGKeyArray, states: Any = None, suff_stats: Any = None, num_actions: int = 1
    ) -> Float[Array, "D K"]:
        """
        Can be used to obtain action samples for all states.
        :param key: The random key.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :param num_actions: The number of actions or K.
        :return: A matrix [D, K] containing action samples generated by the suff stats of the specific state.
        """
        suff_stats = self._suff_stats(states, suff_stats)

        # extract mu and sigma depending on settings, reparametrised sample mu + sigma * eps (torch rsample)
        mu, sigma = suff_stats[:, 0:1], suff_stats[:, 1:2]
        eps = jax.random.normal(key, (mu.shape[0], num_actions), DTYPE)
        samples = mu + eps * sigma
        if self.squash:
            samples = self.action_scale * jnp.tanh(jnp.clip(samples, -SQUASH_U_MAX, SQUASH_U_MAX))
        return samples

    def clone(self) -> "UnivariateGaussianPolicy":
        """
        Clones the current instance. Modules are immutable, so the clone is the instance itself.
        :return: A new identical clones instance of the current.
        """
        return self

    def sync(self, v: "UnivariateGaussianPolicy") -> "UnivariateGaussianPolicy":
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        return self

    def entropy(self, state: ModelState | None, batch: Any) -> Float[Array, ""]:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        policy = state.get("policy", False)
        t_states = jnp.asarray(batch.states, DTYPE)
        suff_stats = policy.forward(t_states)

        sigma = suff_stats[:, 1]
        ent = masked_mean(normal_entropy(sigma), batch_mask(batch))
        return ent

    def kl_divergence(
        self, state: ModelState, batch: Any, suff_stats: Any = None, mode: str = "i-projection", est: str = "sum"
    ) -> Float[Array, ""]:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param state: The traced model state (gives the target policy).
        :param batch:
        :return:
        """

        # calc sufficient statistics
        states = jnp.asarray(batch.states, DTYPE)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)

        # create this distribution
        mu, sigma = suff_stats[:, 0], suff_stats[:, 1]

        # get the old sufficient statistics
        t_policy = state.get("policy", True)
        t_suff_stats = jax.lax.stop_gradient(t_policy.forward(states))
        t_mu, t_sigma = t_suff_stats[:, 0], t_suff_stats[:, 1]

        # use I projection
        if mode == "i-projection":
            kl = normal_kl(mu, sigma, t_mu, t_sigma)

        # use M Projection
        elif mode == "m-projection":
            kl = normal_kl(t_mu, t_sigma, mu, sigma)

        else:
            raise NotImplementedError

        return masked_reduce(kl, batch_mask(batch), est)
