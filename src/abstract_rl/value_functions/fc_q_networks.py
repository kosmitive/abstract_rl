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
from abstract_rl.value_functions.distributional import c51_expectation, c51_loss, mixture_mean, mog_nll


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
        self.distributional = str(conf.get("distributional", "none"))
        self.n_atoms = int(conf.get("n_atoms", 51))
        self.n_components = int(conf.get("n_components", 5))
        grad_clip = conf.get("grad_clip", None)

        # shortcut for state and action dim
        sd = mc.env.observation_dim
        ad = mc.env.action_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = torch.nn.MSELoss()
        match self.distributional:
            case "none":
                out_dim = 1
            case "c51":
                out_dim = self.n_atoms
                atoms = torch.linspace(float(conf.get("v_min", -100.0)), float(conf.get("v_max", 100.0)), self.n_atoms)
                self.atoms: torch.Tensor
                self.register_buffer("atoms", atoms)
            case "mog":
                out_dim = 3 * self.n_components
            case _:
                raise ValueError(f"unknown distributional critic {self.distributional}")

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

    def q_val(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Scalar Q. A distributional head returns the mean of its distribution."""

        raw = self._features(states, actions)
        match self.distributional:
            case "none":
                return raw
            case "c51":
                return c51_expectation(raw, self.atoms)
            case "mog":
                return mixture_mean(raw, self.n_components)
            case _:
                raise ValueError(f"unknown distributional critic {self.distributional}")

    def tr_obj(self, batch: Batch) -> torch.Tensor:
        """Mean squared error, or the distributional loss when that head is enabled."""

        t_states = torch.as_tensor(batch.states, dtype=torch.float32)
        t_actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        q_targets = torch.as_tensor(batch["tq"], dtype=torch.float32).view(-1)
        match self.distributional:
            case "none":
                return self.loss(self.q_val(t_states, t_actions).view(-1), q_targets)
            case "c51":
                return c51_loss(self._features(t_states, t_actions), q_targets, self.atoms)
            case "mog":
                return mog_nll(self._features(t_states, t_actions), q_targets, self.n_components)
            case _:
                raise ValueError(f"unknown distributional critic {self.distributional}")
