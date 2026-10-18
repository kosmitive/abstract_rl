from typing import Any, cast

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import Parameter

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy


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
        self.cov_variable: torch.Tensor | None = None
        if self.cov_type == "state":
            self.num_output_net = 2
        elif self.cov_type == "fixed":
            init_sigma = np.log(np.exp([run_conf["sigma"]]) - 1)
            self.cov_variable = torch.Tensor([init_sigma])
            self.cov_variable.backward()
            self.num_output_net = 0
        elif self.cov_type == "single":
            self.cov_variable = Parameter(torch.zeros([1, 1]), requires_grad=True)
            self.cov_variable.backward()
            self.num_output_net = 1
            self.__setattr__("cov", self.cov_variable)

        # create policy network
        sd = mc.env.observation_dim
        policy_net_structure: np.ndarray = np.concatenate(
            cast(Any, ([sd], run_conf["structure"], [self.num_output_net]))
        )

        af = run_conf["act_fn"]
        self.network = MultilayerPerceptron(
            mc, hidden_neurons=policy_net_structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"]
        )
        self.add_module("nn", self.network)

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
            return self.network.forward(states)

        elif self.cov_type == "fixed" or self.cov_type == "single":
            mu = self.network.forward(states)
            assert self.cov_variable is not None
            fw = torch.exp(self.cov_variable)
            sigma = fw.expand_as(mu)
            return torch.cat([mu, sigma], 1)

        raise NotImplementedError(self.cov_type)

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
        ll = std_normal.log_prob(actions.t())
        return ll.t().contiguous()

    def mode(self, states: torch.Tensor | None = None, suff_stats: torch.Tensor | None = None) -> torch.Tensor:

        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        return suff_stats[:, 0:1]

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
        samples = std_normal.rsample([num_actions])
        return samples.t().contiguous()

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
        self.network.sync(v.network)
        return v

    def entropy(self, batch: Batch) -> torch.Tensor:
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
