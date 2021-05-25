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
import torch.nn as nn
import torch

from abstract_rl.src.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.src.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.src.neural_modules.neural_network import MultilayerPerceptron


class FullyQNeuralNet(nn.Module, Syncable, VanillaGradient):

    """Fully connected neural network."""
    def __init__(self, mc):
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""
        nn.Module.__init__(self)

        conf = mc.get('conf')

        # required settings
        struct = conf['struct']
        self.batch_size = conf['batch_size']

        # neural nets
        layer_norm = conf.get('layer_norm', False)
        init = conf.get('init', 'orthogonal')
        def_fn = conf['act_fn']

        # shortcut for state and action dim
        sd = mc.env.observation_dim
        ad = mc.env.action_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = torch.nn.MSELoss()

        # neural nets
        self.fc_net = MultilayerPerceptron(mc, np.concatenate(([sd+ad], struct, [1])), def_fn, layer_norm, init)
        self.add_module('neural_net', self.fc_net)

        # set up trust regions
        VanillaGradient.__init__(self, self.fc_net)

    def clone(self):
        """Clones the Neural Network"""

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = FullyQNeuralNet(self.mc)
        self.sync(v)
        return v

    def sync(self, v):
        """Clones the Neural Network"""

        self.fc_net.sync(v.fc_net)
        return v

    def count_parameters(self):
        """Counts the parameters of the module itself."""

        return self.fc_net.num_params()

    def forward(self, states, actions):
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def v_val(self, states, exp_actions=10):

        # use policy to estimate v values
        policy = self.mc.get('policy', True)
        rep_states = states.repeat([exp_actions, 1])
        samp_actions = policy.sample_actions(states, num_actions=exp_actions)
        samp_ll = policy.log_prob(samp_actions, states=states)
        lin_actions = samp_actions.view([-1, 1])

        # calculate q values and
        q_vals = self.q_val(rep_states, lin_actions)
        q_vals = q_vals.view([-1, exp_actions])
        v_vals = torch.mean(q_vals, dim=1)
        v_vals = v_vals.view([-1, 1])
        return v_vals

    def q_val(self, states, actions):

        comb = torch.cat([states, actions], dim=1)
        q_vals = self.fc_net.forward(comb)
        return q_vals

    def tr_obj(self, batch):

        # build up mse loss for q function
        t_states = torch.Tensor(batch.states)
        t_actions = torch.Tensor(batch.actions)
        q_vals = self.q_val(t_states, t_actions)
        q_targets = torch.Tensor(batch['tq']).view(-1)
        return self.loss(q_vals, q_targets)
