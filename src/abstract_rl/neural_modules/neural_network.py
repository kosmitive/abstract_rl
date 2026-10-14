"""Fully connected network used by policies and value functions."""

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.misc.mixed import apply_activation


def _linear(key: PRNGKeyArray, in_size: int, out_size: int, init: str) -> eqx.nn.Linear:
    """Create a linear layer with the requested weight initialisation and zero bias.

    Args:
        key: Random key.
        in_size: Input features.
        out_size: Output features.
        init: Name of the initialisation, matching the historical PyTorch helpers.

    Returns:
        An Equinox linear layer.
    """
    key_w, key_layer = jax.random.split(key)
    weight_key, _ = jax.random.split(key_w)
    match init:
        case "orthogonal":
            weight = jax.nn.initializers.orthogonal(scale=math.sqrt(2.0))(weight_key, (out_size, in_size))
        case "xavier_gaussian":
            weight = jax.nn.initializers.glorot_normal()(weight_key, (out_size, in_size))
        case "xavier_uniform":
            weight = jax.nn.initializers.glorot_uniform()(weight_key, (out_size, in_size))
        case "gaussian":
            weight = 0.1 * jax.random.normal(weight_key, (out_size, in_size))
        case "uniform":
            weight = jax.random.uniform(weight_key, (out_size, in_size), minval=-0.1, maxval=0.1)
        case "high":
            weight = jax.random.uniform(weight_key, (out_size, in_size), minval=0.05, maxval=0.1)
        case "none":
            # PyTorch's Linear default is kaiming uniform with a = sqrt(5). That is uniform
            # with variance scale 1/3 on the fan-in axis. The weight matrix is (out, in).
            kaiming = jax.nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform", in_axis=-1, out_axis=-2)
            weight = kaiming(weight_key, (out_size, in_size))
        case _:
            raise NotImplementedError("initialization method not defined.")
    layer = eqx.nn.Linear(in_size, out_size, key=key_layer)
    layer = eqx.tree_at(lambda lin: lin.weight, layer, weight.astype(jnp.float32))
    layer = eqx.tree_at(lambda lin: lin.bias, layer, jnp.zeros((out_size,), dtype=jnp.float32))
    return layer


def _normalize_activations(act_fn: Sequence[str | Sequence[str]], n_layers: int) -> tuple[tuple[str, ...], ...]:
    """Broadcast the config activation list to one tuple of names per layer."""
    if len(act_fn) == 1:
        raw: Sequence[str | Sequence[str]] = list(act_fn) * n_layers
    elif len(act_fn) == n_layers:
        raw = act_fn
    else:
        raise AssertionError("activation list must have length 1 or one entry per layer")
    normalized: list[tuple[str, ...]] = []
    for entry in raw:
        if isinstance(entry, str):
            normalized.append((entry,))
        else:
            normalized.append(tuple(str(name) for name in entry))
    return tuple(normalized)


def _dense(layer: eqx.nn.Linear, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
    """Apply a linear layer to every leading batch axis.

    Equinox linear modules act on a single vector. The weight is stored as ``(out, in)``, so a
    matmul on the last axis matches that layout and keeps the batch together.
    """
    y = x @ layer.weight.T
    if layer.bias is not None:
        y = y + layer.bias
    return y


def _activate(names: tuple[str, ...], x: Float[Array, "... features"]) -> Float[Array, "... features"]:
    """Apply one activation to every feature, or one activation per output feature."""
    if len(names) == 1:
        return apply_activation(names[0], x)
    if len(names) != x.shape[-1]:
        raise ValueError(f"expected {x.shape[-1]} activations, got {len(names)}")
    parts = [apply_activation(names[i], x[..., i : i + 1]) for i in range(x.shape[-1])]
    return jnp.concatenate(parts, axis=-1)


class MultilayerPerceptron(eqx.Module):
    """Fully connected network. Leading axes are batch dimensions; the last axis is features."""

    layers: tuple[eqx.nn.Linear, ...]
    activations: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    layer_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        structure: Sequence[int],
        act_fn: Sequence[str | Sequence[str]],
        layer_norm: bool,
        init: str,
    ) -> None:
        """Build a network whose layer widths are ``structure`` (input through output).

        Args:
            key: Random key for the initialisation.
            structure: Width of every layer, including the input and the output.
            act_fn: One activation name per layer, or a single name reused for every layer.
                A layer entry may itself be a list with one name per output.
            layer_norm: If set, standardise the first linear output before its activation.
            init: Weight initialisation name.
        """
        widths = [int(size) for size in structure]
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(_linear(keys[i], widths[i], widths[i + 1], init) for i in range(len(widths) - 1))
        self.activations = _normalize_activations(act_fn, len(self.layers))
        self.layer_norm = layer_norm

    def forward(self, x: Float[Array, "... features"]) -> Float[Array, "... out"]:
        """Apply the network to ``x``.

        Args:
            x: Inputs whose last axis is the network input width.

        Returns:
            Outputs whose last axis is the network output width.
        """
        hidden = _dense(self.layers[0], x)
        if self.layer_norm:
            hidden = jax.nn.standardize(hidden, axis=-1, epsilon=1e-5)
        hidden = _activate(self.activations[0], hidden)
        for index, layer in enumerate(self.layers[1:], start=1):
            hidden = _activate(self.activations[index], _dense(layer, hidden))
        return hidden
