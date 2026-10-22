import math
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy

# pre-squash samples are limited to |u| <= 5, so tanh stays invertible in float32
SQUASH_U_MAX = 5.0
SQUASH_BOUND = math.tanh(SQUASH_U_MAX)
SIGMA_MIN = 1e-6


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

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new uni variate gaussian policy.
        :param mc: The model configuration to use.
        """
        super().__init__(mc)
        assert isinstance(mc, ModelConfiguration)

        self.mc = mc
        run_conf = mc.get("conf")
        self.conf = run_conf
        self.cov_type = run_conf["cov_type"]
        self.cli = CliPrinter().instance

        self.num_output_net: int | None = None
        self.cov_variable: Any = None
        if self.cov_type == "state":
            self.num_output_net = 2
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            init_log_sigma = np.log(run_conf["sigma"])
            self.cov: torch.Tensor
            self.register_buffer("cov", torch.full([1, 1], init_log_sigma))
            self.cov_variable = self.cov
            self.num_output_net = 1
        elif self.cov_type == "single":
            # single state independent standard deviation, stored in log space
            init_log_sigma = np.log(run_conf.get("sigma", 1.0))
            self.cov = Parameter(torch.full([1, 1], init_log_sigma), requires_grad=True)
            self.cov_variable = self.cov
            self.num_output_net = 1
        else:
            raise NotImplementedError(f"unknown covariance type {self.cov_type}")

        # optionally squash the gaussian with tanh into the (symmetric) action bounds of the environment, so the
        # policy and the q function only ever see actions which the environment actually executes
        self.squash = bool(run_conf.get("squash", False))
        self.action_scale = np.max(np.abs(mc.env.action_space.high))

        # create policy network
        sd = mc.env.observation_dim
        policy_net_structure: np.ndarray = np.concatenate(
            cast(Any, ([sd], run_conf["structure"], [self.num_output_net]))
        )

        af = run_conf["act_fn"]
        self._sigma_is_softplus = _sigma_head_is_softplus(af)
        self.network = MultilayerPerceptron(
            mc, hidden_neurons=policy_net_structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"]
        )
        self.add_module("nn", self.network)
        if self.cov_type == "state":
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            with torch.no_grad():
                bias = self.network.layers[-1].bias
                assert bias is not None
                bias[1] = raw

    def get_suff_stats_dim(self) -> int:
        """
        Obtain the dimension of sufficient statistics vector for a univariate gaussian.
        :return: depending on the configuration 1 for mean-only and 2 for mean and covariance.
        """

        return 2

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Simply produces sufficient statistics of the univariate gaussian policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2]
        """

        if self.cov_type == "state":
            mu, sigma = torch.split(self.network.forward(states), 1, dim=1)
            # if the activation function of the network already is a softplus on the sigma head, only guard against an
            # underflow to 0, otherwise the unconstrained output is made strictly positive with a softplus
            sigma = torch.clamp(sigma, min=SIGMA_MIN) if self._sigma_is_softplus else F.softplus(sigma) + SIGMA_MIN
            return torch.cat([mu, sigma], 1)

        else:
            mu = self.network.forward(states)
            fw = torch.exp(self.cov_variable)
            sigma = fw.expand_as(mu)
            return torch.cat([mu, sigma], 1)

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

        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)

        # extract mu and sigma depending on settings
        mu, sigma = torch.split(suff_stats, 1, dim=1)
        std_normal = Normal(mu[:, 0], sigma[:, 0])
        if not self.squash:
            ll = std_normal.log_prob(actions.t())
            return ll.t().contiguous()

        # change of variables a = c * tanh(u): log p(a) = log N(u) - log(c * (1 - tanh(u)^2))
        y = torch.clamp(actions / self.action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = torch.atanh(y)
        ll = std_normal.log_prob(u.t()).t() - torch.log(self.action_scale * (1 - y.pow(2)))
        return ll

    def mode(self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None) -> torch.Tensor:

        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        mu = suff_stats[:, 0:1]
        return self.action_scale * torch.tanh(mu) if self.squash else mu

    def sample_actions(
        self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None, num_actions: int = 1
    ) -> torch.Tensor:
        """
        Can be used to obtain action samples for all states.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :param num_actions: The number of actions or K.
        :return: A matrix [D, K] containing action samples generated by the suff stats of the specific state.
        """

        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)

        # extract mu and sigma depending on settings
        mu, sigma = torch.split(suff_stats, 1, dim=1)
        std_normal = Normal(mu[:, 0], sigma[:, 0])
        samples = std_normal.rsample([num_actions]).t().contiguous()
        if self.squash:
            samples = self.action_scale * torch.tanh(torch.clamp(samples, -SQUASH_U_MAX, SQUASH_U_MAX))
        return samples

    def clone(self) -> "UnivariateGaussianPolicy":
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = UnivariateGaussianPolicy(self.mc)
        self.sync(v)
        return v

    def sync(self, v: "UnivariateGaussianPolicy") -> "UnivariateGaussianPolicy":
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
        policy = self.mc.get("policy", False)
        t_states = torch.Tensor(batch.states)
        suff_stats = policy.forward(t_states)

        mu, sigma = torch.split(suff_stats, 1, dim=1)
        std_normal = Normal(mu[:, 0], sigma[:, 0])
        ent = std_normal.entropy().mean()
        return ent

    def kl_divergence(
        self, batch: Batch, suff_stats: torch.Tensor | None = None, mode: str = "i-projection", est: str = "sum"
    ) -> torch.Tensor:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param batch:
        :return:
        """

        # calc sufficient statistics
        states = torch.Tensor(batch.states)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)

        # create this distribution
        mu, sigma = torch.split(suff_stats, 1, dim=1)
        std_normal = Normal(mu[:, 0], sigma[:, 0])

        # create this distribution
        # get the old sufficient statistics
        with torch.no_grad():
            t_policy = self.mc.get("policy", True)
            t_suff_stats = t_policy.forward(states)
            t_mu, t_sigma = torch.split(t_suff_stats, 1, dim=1)
            t_std_normal = Normal(t_mu[:, 0], t_sigma[:, 0])

        # use I projection
        if mode == "i-projection":
            p = std_normal
            q = t_std_normal

        # use M Projection
        elif mode == "m-projection":
            q = std_normal
            p = t_std_normal

        else:
            raise NotImplementedError

        kl = torch.distributions.kl.kl_divergence(p, q)

        if est == "sum":
            return kl.sum()
        elif est == "mean":
            return kl.mean()
        else:
            raise NotImplementedError
