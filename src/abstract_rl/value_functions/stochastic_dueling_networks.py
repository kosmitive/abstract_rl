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
import torch.optim as opt
import torch

from abstract_rl.src.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.src.misc.cli_printer import cli
from abstract_rl.src.misc.mixed import select_activation
from abstract_rl.src.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable


class StochasticDuelingNeuralNet(nn.Module, Syncable, TrustRegionGradient):

    """Fully connected neural network."""

    def __init__(self, mc):
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""
        nn.Module.__init__(self)

        conf = mc.get('conf')
        shared_struct = conf['struct']
        assert len(shared_struct) == 2

        # set settings with default values
        layer_norm = conf['layer_norm']
        init = conf.get('init', 'xavier_uniform')

        # shortcut for state and action dim
        sd = mc.env.observation_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = torch.nn.MSELoss()

        # neural nets
        self.shared_net = MultilayerPerceptron(
            hidden_neurons=np.stack([sd, shared_struct[0]]),
            act_fn=['tanh'],
            layer_norm=layer_norm,
            init=init
        )
        self.v_net = MultilayerPerceptron(
            hidden_neurons=np.stack((shared_struct[1], 1)),
            act_fn=['none'],
            layer_norm=False,
            init=init
        )
        self.adv_net = MultilayerPerceptron(
            np.stack([shared_struct[1], 1]),
            act_fn=['none'],
            layer_norm=False,
            init=init
        )

        # create filling layers
        self.v_connect = nn.Linear(shared_struct[1], shared_struct[0])
        self.a_connect = nn.Linear(shared_struct[1] + 1, shared_struct[0])
        self.tanh_act = nn.Tanh()

        self.shared_struct = shared_struct
        self.batch_size = conf['batch_size']
        self.opt_steps = conf.get('steps_per_epoch', 1)

        self.add_module('shared_net', self.shared_net)
        self.add_module('v_net', self.v_net)
        self.add_module('adv_net', self.adv_net)
        self.add_module("v_connect", self.v_connect)
        self.add_module("adv_connect", self.a_connect)

        tr_eps = conf['tr_eps']
        cg_max_k = conf.get('cg_max_k', 10)
        cg_damping = conf.get('dvp_damping', 1e-4)
        cg_residual_tol = conf.get('cg_res_tol', 1e-10)

        # set up trust regions
        TrustRegionGradient.__init__(self, self, tr_eps, cg_max_k, cg_residual_tol, cg_damping)

    def clone(self):
        """Clones the Neural Network"""

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = StochasticDuelingNeuralNet(self.mc)
        self.sync(v)

        return v

    def sync(self, v):
        """Clones the Neural Network"""

        self.shared_net.sync(v.shared_net)
        self.adv_net.sync(v.adv_net)
        self.v_net.sync(v.v_net)
        return v

    def count_parameters(self):
        """Counts the parameters of the module itself."""

        return self.shared_net.num_params() \
               + self.adv_net.num_params() \
                + self.v_net.num_params()

    def forward(self, states, actions):
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def q_val(self, states, actions):

        shared_out = self.shared_net.forward(states)
        a_val = self._a_val(states, actions, shared_out)
        v_val = self.v_val(states, shared_out)
        return a_val + v_val

    def tr_distance(self, batch):

        loss = nn.MSELoss()
        var = self.tr_obj(batch).detach()

        # build up mse loss for v function
        t_states = torch.Tensor(batch.states)
        t_actions = torch.Tensor(batch.actions)
        q_vals = self.q_val(t_states, t_actions).view(-1)
        q_targets = torch.Tensor(batch['q']).view(-1)
        obj = loss(q_vals, q_targets)
        return obj / (2 * var)

    def reset_gradients(self):

        # zero grads
        for p in self.parameters():
            p.grad.detach_()
            p.grad.zero_()

    def set_gradients(self, grads):
        """
        Set gradient itself from the outside.
        :param grads the gradient vector to set.
        """

        with torch.no_grad():
            # zero grads
            i = 0
            for p in self.parameters():
                # extract relevant gradients
                num_els = p.numel()
                ext_grads = grads[i:i + num_els]

                # update grad detach and increase i
                p.grad.copy_(ext_grads.view(p.data.size()))
                i += num_els

            assert len(grads) == i

    def get_gradients(self):
        """
        Gets gradients loaded internally.
        :return: A vector containing the gradients.
        """

        with torch.no_grad():

            lst_all = []
            lst_all.append(self.shared_net)

    def apply_gradients(self, lr):
        """
        Apply loaded gradients by using inverse learning rate.
        :param lr: The learning rate to use.
        """

        # zero grads
        for p in self.parameters():

            # update grad detach and increase i
            p.data.add_(lr * p.grad)

    def tr_obj(self, batch):

        # build up mse loss for q function
        t_states = torch.Tensor(batch.states)
        t_actions = torch.Tensor(batch.actions)
        q_vals = self.q_val(t_states, t_actions)
        q_targets = torch.Tensor(batch['tq']).view(-1)
        return self.loss(q_vals, q_targets)

    def v_val(self, states, shared_out=None):

        if shared_out is None: shared_out = self.shared_net.forward(states)

        v_in = self.v_connect(shared_out)
        v_in = self.tanh_act(v_in)

        # values
        return self.v_net.forward(v_in)

    def _a_val(self, states, actions, shared_out=None):
        if shared_out is None: shared_out = self.shared_net.forward(states)

        policy = self.mc.get('policy', True)
        u_vals = policy.sample_actions(states, num_actions=20)

        num_add_actions = u_vals.size()[1]
        u_val_vec = u_vals.view([-1, 1])
        u_val_vec = u_val_vec.detach()

        # process middle
        rep_shared_out = shared_out.repeat([num_add_actions, 1])
        adv_shared_in = torch.cat([rep_shared_out, u_val_vec], dim=1)
        adv_in = self.a_connect(adv_shared_in)
        adv_in = self.tanh_act(adv_in)

        # repeat the shared out part
        outp = self.adv_net.forward(adv_in)
        outp = outp.view([-1, num_add_actions])
        a_mean_vals = torch.sum(outp, dim=1) / num_add_actions
        a_mean_vals = a_mean_vals.view([-1, 1])

        adv_shared_in = torch.cat([shared_out, actions], dim=1)
        adv_in = self.a_connect(adv_shared_in)
        adv_in = self.tanh_act(adv_in)
        a_vals = self.adv_net.forward(adv_in)
        return a_vals - a_mean_vals
