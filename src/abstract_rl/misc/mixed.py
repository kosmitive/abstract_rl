from typing import Any

import numpy as np
import torch
import torch.nn as nn


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


def select_activation(name: str | None) -> nn.Module:
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: A object of the activation.
    """
    if name == "none" or name is None:
        return nn.Sequential()

    elif name.lower() == "softplus":
        return nn.Softplus()

    elif name.lower() == "elu":
        return nn.ELU()

    elif name.lower() == "prelu":
        return nn.PReLU()

    elif name.lower() == "relu":
        return nn.ReLU()

    elif name.lower() == "lrelu":
        return nn.LeakyReLU()

    elif name.lower() == "tanh":
        return nn.Tanh()

    elif name.lower() == "sigmoid":
        return nn.Sigmoid()

    raise NotImplementedError(f"unknown activation {name}")
