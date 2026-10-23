from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def default_settings(device: str = "cpu", prefer_gpu: bool = False) -> None:
    """
    Apply default settings to all.
    :param device: The MLX device all arrays are created and evaluated on, ``cpu``, ``gpu`` (Metal) or ``auto``.
    :param prefer_gpu: Whether ``auto`` selects the GPU, if Metal is available.
    """
    set_device(resolve_device(device, prefer_gpu))

    def float_formatter(x: Any) -> str:
        return f"{x: 10.4f}"

    np.set_printoptions(formatter={"float_kind": float_formatter})


def resolve_device(device: str, prefer_gpu: bool) -> str:
    """Resolve ``auto`` to ``gpu`` if the algorithm prefers it and Metal is available and to ``cpu`` otherwise."""
    if device != "auto":
        return device
    return "gpu" if prefer_gpu and mx.metal.is_available() else "cpu"


def set_device(device: str) -> None:
    """Select the default MLX device, ``cpu`` or ``gpu``."""
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        if not mx.metal.is_available():
            raise RuntimeError("device=gpu needs MLX with Metal support (Apple silicon), use device=cpu")
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"unknown device {device}, use cpu or gpu")


def on_gpu() -> bool:
    """Whether the default MLX device is the GPU."""
    return mx.default_device() == mx.gpu


def to_mx(x: Any) -> mx.array:
    """Convert a numpy array (or a scalar) into a float32 MLX array. MLX arrays are passed through."""
    if isinstance(x, mx.array):
        return x
    return mx.array(np.asarray(x, dtype=np.float32))


def to_np(x: mx.array) -> np.ndarray:
    """Evaluate an MLX array and return it as a numpy array."""
    return np.array(x)


class ELU(nn.Module):
    """ELU whose gradient stays finite for large inputs.

    ``mlx.nn.elu`` evaluates ``exp(x)`` on both branches of a ``where``, for inputs above about 88 the masked branch
    overflows in the backward pass and turns the gradient into NaN (0 * inf), torch does not have this problem.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._alpha = alpha

    def __call__(self, x: mx.array) -> mx.array:
        return mx.where(x > 0, x, self._alpha * (mx.exp(mx.minimum(x, 0)) - 1))


class Identity(nn.Module):
    """Identity activation."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def select_activation(name: str | None) -> nn.Module:
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: A object of the activation.
    """
    if name == "none" or name is None:
        return Identity()

    elif name.lower() == "softplus":
        return nn.Softplus()

    elif name.lower() == "elu":
        return ELU()

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
