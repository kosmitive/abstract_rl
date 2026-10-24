from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


def _flat_trainable(model: nn.Module) -> list[tuple[str, mx.array]]:
    return cast(list[tuple[str, mx.array]], tree_flatten(model.trainable_parameters()))


def get_flat_params_from(model: nn.Module) -> mx.array:
    """Extract the trainable parameters of a model as one flat vector."""

    return mx.concatenate([p.reshape(-1) for _, p in _flat_trainable(model)])


def unflatten_params(model: nn.Module, flat_params: mx.array) -> Any:
    """Split a flat vector into a parameter tree shaped like the trainable parameters of the model."""
    out = []
    prev_ind = 0
    for name, param in _flat_trainable(model):
        flat_size = param.size
        out.append((name, flat_params[prev_ind : prev_ind + flat_size].reshape(param.shape)))
        prev_ind += flat_size
    return tree_unflatten(out)


def set_flat_params_to(model: nn.Module, flat_params: mx.array) -> None:
    """Set the trainable parameters of a model from a flat vector."""
    model.update(unflatten_params(model, flat_params))
