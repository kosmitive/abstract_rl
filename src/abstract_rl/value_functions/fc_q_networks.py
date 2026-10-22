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

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


class FullyQNeuralNet(VanillaGradient, Syncable):
    """Fully connected neural network."""

    batch_size: int = eqx.field(static=True)
    fc_net: MultilayerPerceptron

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray, mc_key: str = "q_network") -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""

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

        # neural nets
        structure = np.concatenate(([sd + ad], struct, [1]))
        self.fc_net = MultilayerPerceptron(mc, structure, def_fn, layer_norm, init, key=key)

        # set up the adam fit
        self.lr = conf.get("lr", 1e-3)
        self.mc_key = mc_key

    def clone(self) -> "FullyQNeuralNet":
        """Clones the Neural Network, modules are immutable so the clone is the instance itself."""
        return self

    def sync(self, v: "FullyQNeuralNet") -> "FullyQNeuralNet":
        """Clones the Neural Network"""
        return self

    def forward(self, states: Float[Array, "D Ds"], actions: Float[Array, "D A"]) -> Float[Array, "D 1"]:
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def v_val(
        self,
        state: ModelState,
        key: PRNGKeyArray,
        states: Float[Array, "D Ds"],
        exp_actions: int = 10,
        policy: Any = None,
    ) -> Float[Array, "D 1"]:
        """Estimate v values by averaging q values of actions sampled from the target policy.

        :param state: The traced model state (gives the target policy).
        :param key: The random key of the action samples.
        :param states: A matrix [D, Ds] of states.
        :param exp_actions: The number of sampled actions K per state.
        :param policy: The policy to evaluate, defaults to the target policy.
        :return: A matrix [D, 1] of v values.
        """

        policy = policy if policy is not None else state.get("policy", True)
        samp_actions = policy.sample_actions(key, states, num_actions=exp_actions)
        q_vals = self.q_val_multi(states, samp_actions)
        return jnp.mean(q_vals, axis=1, keepdims=True)

    def q_val_multi(self, states: Float[Array, "D Ds"], actions: Any) -> Float[Array, "D K"]:
        """Evaluate K actions per state.

        :param states: A matrix [D, Ds] of states.
        :param actions: A matrix [D, K] of (univariate) actions or a tensor [D, K, A] for A action dimensions.
        :return: A matrix [D, K] of q values.
        """
        d, k = actions.shape[:2]
        rep_states = jnp.repeat(jnp.asarray(states, DTYPE), k, axis=0)
        q_vals = self.q_val(rep_states, actions.reshape([d * k, -1]))
        return q_vals.reshape([d, k])

    def q_val(self, states: Float[Array, "D Ds"], actions: Float[Array, "D A"]) -> Float[Array, "D 1"]:

        comb = jnp.concatenate([jnp.asarray(states, DTYPE), jnp.asarray(actions, DTYPE)], axis=1)
        q_vals = self.fc_net.forward(comb)
        return q_vals

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:

        # build up mse loss for q function
        t_states = jnp.asarray(batch.states, DTYPE)
        t_actions = jnp.asarray(batch.actions, DTYPE)
        q_vals = self.q_val(t_states, t_actions).reshape(-1)
        q_targets = jnp.asarray(batch["tq"], DTYPE).reshape(-1)
        return masked_mean((q_vals - q_targets) ** 2, batch_mask(batch))
