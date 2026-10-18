#  Copyright 2018 Markus Semmler
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#  Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import torch
import torch.nn as nn

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy
from abstract_rl.value_functions.distributional import (
    mixture_mean,
    mog_sample_nll,
    project_mixture_samples,
    sample_mixture,
    split_mixture,
)


class FullyQNeuralNet(nn.Module, Syncable, VanillaGradient):
    """Fully connected neural network."""

    def __init__(self, mc: ModelConfiguration) -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""
        nn.Module.__init__(self)

        conf = mc.get("conf")

        # required settings
        struct = conf["struct"]
        self.batch_size = conf["batch_size"]

        # neural nets
        layer_norm = conf.get("layer_norm", False)
        layernorm_tanh = bool(conf.get("layernorm_tanh", False))
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]
        # critic_type "mog" is the optional mixture-of-Gaussians head; "none" is the scalar critic
        self.critic_type = str(conf.get("critic_type", "none"))
        self.n_components = int(conf.get("n_components", 5))
        self.mog_init_scale = float(conf.get("mog_init_scale", 1e-3))
        self.value_samples = int(conf.get("mog_value_samples", 128))
        raw_conf = mc["conf"].to_dict()
        self.bootstrap_actions = int(raw_conf.get("res_add_acts", 20))
        grad_clip = conf.get("grad_clip", None)

        # shortcut for state and action dim
        sd = mc.env.observation_dim
        ad = mc.env.action_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = torch.nn.MSELoss()
        match self.critic_type:
            case "none":
                out_dim = 1
            case "mog":
                out_dim = 3 * self.n_components
            case _:
                raise ValueError(f"unknown critic type {self.critic_type}")

        # neural nets
        self.fc_net = MultilayerPerceptron(
            mc,
            np.concatenate(([sd + ad], struct, [out_dim])),
            def_fn,
            layer_norm,
            init,
            layernorm_tanh=layernorm_tanh,
        )
        self.add_module("neural_net", self.fc_net)

        # set up trust regions
        VanillaGradient.__init__(
            self,
            self.fc_net,
            conf.get("optimizer", "lbfgs"),
            conf.get("lr", 1e-3),
            None if grad_clip is None else float(grad_clip),
        )

    def clone(self) -> "FullyQNeuralNet":
        """Clones the Neural Network"""

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = FullyQNeuralNet(self.mc)
        self.sync(v)
        return v

    def sync(self, v: "FullyQNeuralNet") -> "FullyQNeuralNet":
        """Clones the Neural Network"""

        self.fc_net.sync(v.fc_net)
        return v

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def v_val(
        self,
        states: torch.Tensor,
        exp_actions: int = 10,
        policy: DiagonalGaussianPolicy | UnivariateGaussianPolicy | None = None,
    ) -> torch.Tensor:
        """Estimate v values by averaging q values of actions sampled from the target policy.

        :param states: A matrix [D, Ds] of states.
        :param exp_actions: The number of sampled actions K per state.
        :param policy: The policy to evaluate, defaults to the target policy.
        :return: A matrix [D, 1] of v values.
        """

        policy = policy if policy is not None else self.mc.get("policy", True)
        samp_actions = policy.sample_actions(states, num_actions=exp_actions)
        q_vals = self.q_val_multi(states, samp_actions)
        return torch.mean(q_vals, dim=1, keepdim=True)

    def q_val_multi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate K actions per state.

        :param states: A matrix [D, Ds] of states.
        :param actions: A matrix [D, K] of (univariate) actions or a tensor [D, K, A] for A action dimensions.
        :return: A matrix [D, K] of q values.
        """
        d, k = actions.shape[:2]
        rep_states = states.repeat_interleave(k, dim=0)
        q_vals = self.q_val(rep_states, actions.reshape([d * k, -1]))
        return q_vals.view([d, k])

    def _features(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.fc_net.forward(torch.cat([states, actions], dim=1))

    def _mixture(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Means, softplus scales and logits of the mixture at each state-action pair."""

        return split_mixture(self._features(states, actions), self.n_components, self.mog_init_scale)

    def q_val(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Scalar Q. The mixture head returns the mean of its components."""

        match self.critic_type:
            case "none":
                return self._features(states, actions)
            case "mog":
                means, _, logits = self._mixture(states, actions)
                return mixture_mean(means, logits)
            case _:
                raise ValueError(f"unknown critic type {self.critic_type}")

    def tr_obj(self, batch: Batch) -> torch.Tensor:
        """Mean squared error, or the mixture negative log likelihood when that head is enabled."""

        match self.critic_type:
            case "none":
                t_states = torch.as_tensor(batch.states, dtype=torch.float32)
                t_actions = torch.as_tensor(batch.actions, dtype=torch.float32)
                q_targets = torch.as_tensor(batch["tq"], dtype=torch.float32).view(-1)
                return self.loss(self.q_val(t_states, t_actions).view(-1), q_targets)
            case "mog":
                return self._mog_loss(batch)
            case _:
                raise ValueError(f"unknown critic type {self.critic_type}")

    def _mog_loss(self, batch: Batch) -> torch.Tensor:
        """Sample-based cross-entropy of the reward-and-discount projection of the target mixture.

        Actions at the next state are drawn from the target policy. Samples from the target critic at those actions
        are projected by ``reward + discount * (1 - done) * sample`` and scored by the online mixture. Neither the
        target critic nor the target policy receives a gradient.
        """

        states = torch.as_tensor(batch.states, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32)
        means, scales, logits = self._mixture(states, actions)

        with torch.no_grad():
            target_policy = self.mc.get("policy", True)
            target_q = self.mc.get("q_network", True)
            next_actions = target_policy.sample_actions(next_states, num_actions=self.bootstrap_actions)
            count = int(next_actions.shape[1])
            flat_next = next_states.repeat_interleave(count, dim=0)
            flat_actions = next_actions.reshape(next_states.shape[0] * count, -1)
            target_means, target_scales, target_logits = target_q._mixture(flat_next, flat_actions)
            draws = sample_mixture(target_means, target_scales, target_logits, self.value_samples)
            draws = draws.reshape(self.value_samples, next_states.shape[0], count)
            draws = draws.permute(2, 0, 1).reshape(count * self.value_samples, next_states.shape[0])

        projected = project_mixture_samples(draws, rewards, self.env.discount(), dones)
        return mog_sample_nll(means, scales, logits, projected)
