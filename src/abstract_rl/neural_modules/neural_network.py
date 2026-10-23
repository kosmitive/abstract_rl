import math as m
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from numpy import ndarray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.misc.mixed import select_activation
from abstract_rl.misc.typing import DTYPE


class MultilayerPerceptron(eqx.Module, Syncable):
    """
    Fully connected neural network
    """

    structure: tuple[int, ...] = eqx.field(static=True)
    layer_norm: bool = eqx.field(static=True)
    init: str = eqx.field(static=True)
    act_fn: tuple[Any, ...] = eqx.field(static=True)
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        mc: ModelConfiguration,
        hidden_neurons: ndarray,
        act_fn: str | Sequence[str | list[str]] = "tanh",
        layer_norm: bool = False,
        init: str = "orthogonal",
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Create a new neural network, whereas the structure is passed as a list of numbers, representing the size of
        each fully connected layer including the input as well as the output layer.

        :param hidden_neurons: List of integers containing the size of the hidden layers
        :param act_fn: A string encoding the activation function.
        :param layer_norm: Activate layer normalization in between.
        :param init: A string representing the initialization method.
        :param key: The random key for the initialization.

        """
        assert len(act_fn) == len(hidden_neurons) - 1 or len(act_fn) == 1

        # save settings
        self.structure = tuple(int(n) for n in hidden_neurons)
        self.layer_norm = layer_norm
        self.init = init

        # select the correct activation functions (tuples, the static structure has to be hashable).
        self.act_fn = tuple(
            tuple(select_activation(ae) for ae in a) if isinstance(a, list | tuple) else select_activation(a)
            for a in act_fn
        )

        # create the layers and initialize
        num_layers = len(self.structure) - 1
        keys = jax.random.split(key, 2 * num_layers)
        layers = [
            eqx.nn.Linear(self.structure[i], self.structure[i + 1], key=keys[i], dtype=DTYPE) for i in range(num_layers)
        ]
        self.layers = self.initialize_layers(layers, init, keys[num_layers:])

    @staticmethod
    def initialize_layers(layers: list[eqx.nn.Linear], init: str, keys: Any) -> list[eqx.nn.Linear]:
        """
        This method can be used to initialize a layer
        """

        layers = [eqx.tree_at(lambda la: la.bias, layer, jnp.zeros(layer.out_features, DTYPE)) for layer in layers]
        normal, uniform = jax.random.normal, jax.random.uniform
        w_init: Callable[..., Array]
        if init == "xavier_gaussian":
            w_init = jax.nn.initializers.variance_scaling(1.0, "fan_avg", "normal", in_axis=-1, out_axis=-2)

        elif init == "xavier_uniform":
            w_init = jax.nn.initializers.variance_scaling(1.0, "fan_avg", "uniform", in_axis=-1, out_axis=-2)

        elif init == "orthogonal":
            w_init = jax.nn.initializers.orthogonal(m.sqrt(2))

        elif init == "gaussian":

            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return 0.1 * normal(k, shape, dtype)

        elif init == "uniform":

            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return uniform(k, shape, dtype, -0.1, 0.1)

        elif init == "high":

            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return uniform(k, shape, dtype, 0.05, 0.1)

        elif init == "none":
            return layers
        else:
            raise NotImplementedError("initialization method not defined.")

        return [
            eqx.tree_at(lambda la: la.weight, layer, w_init(k, layer.weight.shape, DTYPE))
            for layer, k in zip(layers, keys, strict=True)
        ]

    def forward(self, x: Float[Array, "*D n_in"]) -> Float[Array, "*D n_out"]:
        """
        Apply represented function to x.
        :param x: The input to the network.
        :return: The outputted values of the network.
        """

        # apply layer norm if wanted (torch nn.LayerNorm created per call: unit weight, zero bias, eps 1e-5)
        out = x
        bias0 = self.layers[0].bias
        assert bias0 is not None
        out = out @ self.layers[0].weight.T + bias0
        if self.layer_norm:
            mean = jnp.mean(out, axis=-1, keepdims=True)
            var = jnp.var(out, axis=-1, keepdims=True)
            out = (out - mean) / jnp.sqrt(var + 1e-5)

        out = self.act_fn[0](out)

        # iteratively complete the graph in the middle
        num_layers = len(self.layers)
        for layer_idx in range(1, num_layers):
            # assemble linear part, layer norm and activation for each layer
            bias = self.layers[layer_idx].bias
            assert bias is not None
            linear_f = out @ self.layers[layer_idx].weight.T + bias
            act = self.act_fn[layer_idx] if len(self.act_fn) > 1 else self.act_fn[0]
            if isinstance(act, tuple):
                splitted_ts = [linear_f[..., i : i + 1] for i in range(linear_f.shape[-1])]
                for i in range(len(splitted_ts)):
                    splitted_ts[i] = act[i](splitted_ts[i])
                out = jnp.concatenate(splitted_ts, axis=-1)

            else:
                out = act(linear_f)

        return out

    def num_params(self) -> int:
        """
        Count parameters of module.
        """

        return sum(p.size for p in jax.tree.leaves(eqx.filter(self, eqx.is_inexact_array)))

    def sync(self, v: "MultilayerPerceptron") -> "MultilayerPerceptron":
        """
        Copies the content of the current instance to the supplied parameter instance. Modules are immutable, so the
        synced instance is the current one.
        :param v: The other instance, where the current values should be copied to.
        """
        return self
