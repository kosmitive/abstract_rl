import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from abstract_rl.src.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.src.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable


class FullyVNeuralNet(nn.Module, Syncable, VanillaGradient):

    """Fully connected neural network."""

    def __init__(self, mc):
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""
        nn.Module.__init__(self)

        conf = mc.get('conf')
        struct = conf['struct']

        # shortcut for state and action dim
        sd = mc.env.observation_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = torch.nn.MSELoss()

        # neural nets
        layer_norm = conf.get('layer_norm', False)
        init = conf.get('init', 'orthogonal')
        def_fn = conf['act_fn']
        self.l2 = conf['l2_reg']

        # create network
        self.net = MultilayerPerceptron(mc, np.concatenate(([sd], struct, [1])), def_fn, layer_norm, init)
        self.add_module('neural_net', self.net)

        # set up trust regions
        VanillaGradient.__init__(self, self.net)

    def clone(self):
        """Clones the Neural Network"""

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = FullyVNeuralNet(self.mc)
        self.sync(v)
        return v

    def sync(self, v):
        """Clones the Neural Network"""

        self.net.sync(v.net)
        return v

    def count_parameters(self):
        """Counts the parameters of the module itself."""

        return self.net.num_params()

    def forward(self, states):
        """Propagate the x in a forward fashion through the network."""
        return self.v_val(states)

    def v_val(self, states):
        v_vals = self.net.forward(states)
        return v_vals

    def tr_obj(self, batch):

        loss = nn.MSELoss()

        # build up mse loss for v function
        t_states = torch.Tensor(batch.states)
        states = Variable(t_states)
        v_vals = self.v_val(states).view(-1)
        v_targets = torch.Tensor(batch['tv']).view(-1)
        v_targets = Variable(v_targets)
        w = torch.cat([p.view(-1) for p in self.parameters()])
        obj = loss(v_vals, v_targets) + self.l2 * torch.norm(w)
        return obj
