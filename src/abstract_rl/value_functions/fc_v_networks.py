import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.mixed import to_mx
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


class FullyVNeuralNet(nn.Module, Syncable, VanillaGradient):
    """Fully connected neural network."""

    def __init__(self, mc: ModelConfiguration) -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""
        nn.Module.__init__(self)

        conf = mc.get("conf")
        struct = conf["struct"]

        # shortcut for state and action dim
        sd = mc.env.observation_dim

        # set the standard act function
        self.env = mc.env
        self.mc = mc
        self.loss = nn.losses.mse_loss

        # neural nets
        layer_norm = conf.get("layer_norm", False)
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]
        self.l2 = conf["l2_reg"]

        # create network
        self.net = MultilayerPerceptron(mc, np.concatenate(([sd], struct, [1])), def_fn, layer_norm, init)

        # set up trust regions
        VanillaGradient.__init__(self, self.net, lr=conf.get("lr", 1e-3))

    def clone(self) -> "FullyVNeuralNet":
        """Clones the Neural Network"""

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = FullyVNeuralNet(self.mc)
        self.sync(v)
        return v

    def sync(self, v: "FullyVNeuralNet") -> "FullyVNeuralNet":
        """Clones the Neural Network"""

        self.net.sync(v.net)
        return v

    def __call__(self, states: mx.array) -> mx.array:
        return self.forward(states)

    def forward(self, states: mx.array) -> mx.array:
        """Propagate the x in a forward fashion through the network."""
        return self.v_val(states)

    def v_val(self, states: mx.array) -> mx.array:
        v_vals = self.net.forward(states)
        return v_vals

    def tr_obj(self, batch: Batch) -> mx.array:

        # build up mse loss for v function
        states = to_mx(batch.states)
        v_vals = self.v_val(states).reshape(-1)
        v_targets = to_mx(batch["tv"]).reshape(-1)
        w = get_flat_params_from(self.net)
        obj = self.loss(v_vals, v_targets) + self.l2 * mx.linalg.norm(w)
        return obj
