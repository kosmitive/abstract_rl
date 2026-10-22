from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch


def default_settings(num_threads: int = 1) -> None:
    """
    Apply default settings to all.
    :param num_threads: Number of intra-op threads for torch. The networks are tiny, so a single thread is fastest.
    """
    torch.set_num_threads(num_threads)

    def float_formatter(x: Any) -> str:
        return f"{x: 10.4f}"

    np.set_printoptions(formatter={"float_kind": float_formatter})
    # torch.set_printoptions(profile='full')


def _identity(x: jax.Array) -> jax.Array:
    return x


def _prelu(x: jax.Array) -> jax.Array:
    # torch nn.PReLU at its initial slope 0.25: the module was never registered on the network, so it was never trained
    return jnp.where(x >= 0, x, 0.25 * x)


def _lrelu(x: jax.Array) -> jax.Array:
    return jax.nn.leaky_relu(x, 0.01)


def select_activation(name: str | None) -> Any:
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: The activation function (a module level function, so two networks compare equal).
    """
    if name == "none" or name is None:
        return _identity

    elif name.lower() == "softplus":
        return jax.nn.softplus

    elif name.lower() == "elu":
        return jax.nn.elu

    elif name.lower() == "prelu":
        return _prelu

    elif name.lower() == "relu":
        return jax.nn.relu

    elif name.lower() == "lrelu":
        return _lrelu

    elif name.lower() == "tanh":
        return jnp.tanh

    elif name.lower() == "sigmoid":
        return jax.nn.sigmoid

    raise NotImplementedError(f"unknown activation {name}")
