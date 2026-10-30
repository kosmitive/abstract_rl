import math as m
from collections.abc import Callable, Sequence
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from numpy import ndarray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.misc.mixed import select_activation


class MultilayerPerceptron(nn.Module, Syncable):
    """
    Fully connected neural network
    """

    def __init__(
        self,
        mc: ModelConfiguration | None,
        hidden_neurons: ndarray,
        act_fn: str | Sequence[str | list[str]] = "tanh",
        layer_norm: bool = False,
        init: str = "orthogonal",
    ) -> None:
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
        self.layer_norm = layer_norm
        self.init = init

        # select the correct activation functions. They are not parameters of the network (an underscore keeps them
        # out of the parameter tree), as in the torch version where they were not registered as sub modules.
        self._act_fn: list[Any] = [
            [select_activation(ae) for ae in a] if isinstance(a, list) else select_activation(a) for a in act_fn
        ]

        # create the layers (registered as parameters through the list) and initialize
        self.layers = [
            nn.Linear(int(self.structure[i]), int(self.structure[i + 1])) for i in range(len(self.structure) - 1)
        ]
        self.initialize_layers(init)

    @property
    def act_fn(self) -> list[Any]:
        return self._act_fn

    def initialize_layers(self, init: str) -> None:
        """
        This method can be used to initialize a layer. Weights are ``[out, in]`` as in torch, so the fan in and fan
        out of the glorot initializations match ``torch.nn.init``.
        """

        for layer in self.layers:
            layer.bias = mx.zeros_like(layer.bias)
        if init == "none":
            return
        initializers: dict[str, Callable[[], Callable[..., mx.array]]] = {
            "xavier_gaussian": lambda: nn.init.glorot_normal(),
            "xavier_uniform": lambda: nn.init.glorot_uniform(),
            "orthogonal": lambda: nn.init.orthogonal(m.sqrt(2)),
            "gaussian": lambda: nn.init.normal(0.0, 0.1),
            "uniform": lambda: nn.init.uniform(-0.1, 0.1),
            "high": lambda: nn.init.uniform(0.05, 0.1),
        }
        if init not in initializers:
            raise NotImplementedError("initialization method not defined.")
        fn = initializers[init]()
        for layer in self.layers:
            layer.weight = fn(layer.weight)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    def forward(self, x: mx.array) -> mx.array:
        """
        Apply represented function to x.
        :param x: The input to the network.
        :return: The outputted values of the network.
        """

        # apply layer norm if wanted (no affine parameters, like the freshly created torch.nn.LayerNorm)
        out = x
        out = self.layers[0](out)
        if self.layer_norm:
            out = mx.fast.layer_norm(out, None, None, 1e-5)

        out = self._act_fn[0](out)

        # iteratively complete the graph in the middle
        num_layers = len(self.layers)
        for layer_idx in range(1, num_layers):
            # assemble linear part, layer norm and activation for each layer
            linear_f = self.layers[layer_idx](out)
            act = self._act_fn[layer_idx] if len(self._act_fn) > 1 else self._act_fn[0]
            if isinstance(act, list):
                splitted_ts = [linear_f[:, i : i + 1] for i in range(linear_f.shape[1])]
                for i in range(len(splitted_ts)):
                    splitted_ts[i] = act[i](splitted_ts[i])
                out = mx.concatenate(splitted_ts, axis=1)

            else:
                out = act(linear_f)

        return out

    def num_params(self) -> int:
        """
        Count parameters of module.
        """

        from mlx.utils import tree_flatten

        params = cast(list[tuple[str, mx.array]], tree_flatten(self.trainable_parameters()))
        return sum(p.size for _, p in params)

    def sync(self, v: "MultilayerPerceptron") -> "MultilayerPerceptron":
        """
        Copies the content of the current instance to the supplied parameter instance. MLX arrays are immutable, so
        sharing them is a copy: every later update of either module replaces its arrays.
        :param v: The other instance, where the current values should be copied to.
        """

        v.update(self.parameters())
        return v
