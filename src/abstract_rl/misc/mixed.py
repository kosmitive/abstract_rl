"""Small helpers shared by the training loops."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def default_settings(num_threads: int = 1) -> None:
    """Apply process-wide numeric defaults.

    The thread count is accepted so existing configs keep working. JAX schedules its own
    CPU pool; the networks are small enough that the gain is in jitting whole updates
    rather than in pinning intra-op threads.

    Args:
        num_threads: Unused legacy torch intra-op thread count.
    """
    del num_threads
    jax.config.update("jax_enable_x64", True)

    def float_formatter(x: Any) -> str:
        return f"{x: 10.4f}"

    np.set_printoptions(formatter={"float_kind": float_formatter})


def apply_activation(name: str | None, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Apply a named pointwise activation.

    Args:
        name: Activation name, or ``none`` for the identity.
        x: Input array.

    Returns:
        The activated array, with the same shape as ``x``.
    """
    key = "none" if name is None else name.lower()
    match key:
        case "none":
            return x
        case "softplus":
            return jax.nn.softplus(x)
        case "elu":
            return jax.nn.elu(x)
        case "prelu":
            # Stateless stand-in for the default PReLU slope. The benchmark configs do not use it.
            return jnp.where(x >= 0, x, 0.25 * x)
        case "relu":
            return jax.nn.relu(x)
        case "lrelu":
            return jax.nn.leaky_relu(x, negative_slope=0.01)
        case "tanh":
            return jnp.tanh(x)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case _:
            raise NotImplementedError(f"unknown activation {name}")
