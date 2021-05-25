import math as m
import torch.nn as nn
import torch

from abstract_rl.src.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.src.misc.mixed import select_activation


class MultilayerPerceptron(nn.Module, Syncable):
    """
    Fully connected neural network
    """

    def __init__(self, mc, hidden_neurons, act_fn='tanh', layer_norm=False, init='orthogonal'):
        """
        Create a new neural network, whereas the structure is passed as a list of numbers, representing the size of
        each fully connected layer including the input as well as the output layer.

        :param hidden_neurons: List of integers containing the size of the hidden layers
        :param act_fn: A string encoding the activation function.
        :param layer_norm: Activate layer normalization in between.
        :param init: A string representing the initialization method.

        """
        nn.Module.__init__(self)

        assert len(act_fn) == len(hidden_neurons) - 1 or len(act_fn) == 1

        # save settings
        self.mc = mc
        self.structure = hidden_neurons
        self.act_fn = act_fn
        self.layer_norm = layer_norm
        self.init = init

        # select the correct activation functions.
        self.act_fn = [[select_activation(ae) for ae in a] if isinstance(a, list) else select_activation(a) for a in act_fn]

        # create the layers, add internally and initialize
        self.layers = [nn.Linear(self.structure[i], self.structure[i + 1]) for i in range(len(self.structure) - 1)]
        for i in range(len(self.layers)): self.add_module(f"layer_{i}", self.layers[i])
        self.initialize_layers(init)

    def initialize_layers(self, init):
        """
        This method can be used to initialize a layer
        """

        for layer in self.layers: nn.init.zeros_(layer.bias)
        if init == 'xavier_gaussian':
            for layer in self.layers: nn.init.xavier_normal_(layer.weight)

        elif init == 'xavier_uniform':
            for layer in self.layers: nn.init.xavier_uniform_(layer.weight)

        elif init == 'orthogonal':
            for layer in self.layers: nn.init.orthogonal_(layer.weight, m.sqrt(2))

        elif init == 'gaussian':
            for layer in self.layers:
                nn.init.normal_(layer.weight, 0, 0.1)

        elif init == 'uniform':
            for layer in self.layers:
                nn.init.uniform_(layer.weight, -0.1, 0.1)

        elif init == 'high':
            for layer in self.layers:
                nn.init.uniform_(layer.weight, 0.05, 0.1)

        elif init == 'none': return
        else:
            raise NotImplementedError("initialization method not defined.")

    def forward(self, x):
        """
        Apply represented function to x.
        :param x: The input to the network.
        :return: The outputted values of the network.
        """

        # apply layer norm if wanted
        out = x
        out = self.layers[0](out)
        if self.layer_norm:
            layer_norm = nn.LayerNorm(out.shape[-1])
            out = layer_norm(out)

        out = self.act_fn[0](out)

        # iteratively complete the graph in the middle
        num_layers = len(self.layers)
        for l in range(1, num_layers):

            # assemble linear part, layer norm and activation for each layer
            linear_f = self.layers[l](out)
            act = self.act_fn[l] if len(self.act_fn) > 0 else self.act_fn[0]
            if isinstance(act, list):
                splitted_ts = list(torch.split(linear_f, 1, dim=1))
                for i in range(len(splitted_ts)):
                    splitted_ts[i] = act[i](splitted_ts[i])
                out = torch.cat(splitted_ts, dim=1)

            else:
                out = act(linear_f)

        return out

    def num_params(self):
        """
        Count parameters of module.
        """

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sync(self, v):
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """

        # check if it can be copied
        v.load_state_dict(self.state_dict())
        return v