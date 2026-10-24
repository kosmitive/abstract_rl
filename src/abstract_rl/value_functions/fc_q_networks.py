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

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.mixed import to_mx
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


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
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]

        # shortcut for state and action dim
        sd = mc.env.observation_dim
        ad = mc.env.action_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = nn.losses.mse_loss

        # neural nets
        self.fc_net = MultilayerPerceptron(mc, np.concatenate(([sd + ad], struct, [1])), def_fn, layer_norm, init)

        # set up trust regions
        VanillaGradient.__init__(self, self.fc_net, lr=conf.get("lr", 1e-3))

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

    def __call__(self, states: mx.array, actions: mx.array) -> mx.array:
        return self.forward(states, actions)

    def forward(self, states: mx.array, actions: mx.array) -> mx.array:
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def v_val(
        self,
        states: mx.array,
        exp_actions: int = 10,
        policy: DiagonalGaussianPolicy | UnivariateGaussianPolicy | None = None,
        noise: mx.array | None = None,
    ) -> mx.array:
        """Estimate v values by averaging q values of actions sampled from the target policy.

        :param states: A matrix [D, Ds] of states.
        :param exp_actions: The number of sampled actions K per state.
        :param policy: The policy to evaluate, defaults to the target policy.
        :param noise: The noise of the sampled actions, see sample_noise of the policy. Drawn if None.
        :return: A matrix [D, 1] of v values.
        """

        policy = policy if policy is not None else self.mc.get("policy", True)
        samp_actions = policy.sample_actions(states, num_actions=exp_actions, noise=noise)
        q_vals = self.q_val_multi(states, samp_actions)
        return mx.mean(q_vals, axis=1, keepdims=True)

    def q_val_multi(self, states: mx.array, actions: mx.array) -> mx.array:
        """Evaluate K actions per state.

        :param states: A matrix [D, Ds] of states.
        :param actions: A matrix [D, K] of (univariate) actions or a tensor [D, K, A] for A action dimensions.
        :return: A matrix [D, K] of q values.
        """
        d, k = actions.shape[:2]
        rep_states = mx.repeat(states, k, axis=0)
        q_vals = self.q_val(rep_states, actions.reshape([d * k, -1]))
        return q_vals.reshape([d, k])

    def q_val(self, states: mx.array, actions: mx.array) -> mx.array:

        comb = mx.concatenate([states, actions], axis=1)
        q_vals = self.fc_net.forward(comb)
        return q_vals

    def tr_obj(self, batch: Batch) -> mx.array:

        # build up mse loss for q function
        t_states = to_mx(batch.states)
        t_actions = to_mx(batch.actions)
        q_vals = self.q_val(t_states, t_actions).reshape(-1)
        q_targets = to_mx(batch["tq"]).reshape(-1)
        return self.loss(q_vals, q_targets)
