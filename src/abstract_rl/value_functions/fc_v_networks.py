import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


class FullyVNeuralNet(VanillaGradient, Syncable):
    """Fully connected neural network."""

    l2: float = eqx.field(static=True)
    net: MultilayerPerceptron

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray, mc_key: str = "v_network") -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""

        conf = mc.get("conf")
        struct = conf["struct"]

        # shortcut for state and action dim
        sd = mc.env.observation_dim

        # neural nets
        layer_norm = conf.get("layer_norm", False)
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]
        self.l2 = conf["l2_reg"]

        # create network
        self.net = MultilayerPerceptron(mc, np.concatenate(([sd], struct, [1])), def_fn, layer_norm, init, key=key)

        # set up the adam fit
        self.lr = conf.get("lr", 1e-3)
        self.mc_key = mc_key

    def clone(self) -> "FullyVNeuralNet":
        """Clones the Neural Network, modules are immutable so the clone is the instance itself."""
        return self

    def sync(self, v: "FullyVNeuralNet") -> "FullyVNeuralNet":
        """Clones the Neural Network"""
        return self

    def forward(self, states: Float[Array, "D Ds"]) -> Float[Array, "D 1"]:
        """Propagate the x in a forward fashion through the network."""
        return self.v_val(states)

    def v_val(self, states: Float[Array, "D Ds"]) -> Float[Array, "D 1"]:
        v_vals = self.net.forward(jnp.asarray(states, DTYPE))
        return v_vals

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:

        # build up mse loss for v function
        states = jnp.asarray(batch.states, DTYPE)
        v_vals = self.v_val(states).reshape(-1)
        v_targets = jnp.asarray(batch["tv"], DTYPE).reshape(-1)
        w = get_flat_params_from(self)
        obj = masked_mean((v_vals - v_targets) ** 2, batch_mask(batch)) + self.l2 * jnp.linalg.norm(w)
        return obj
