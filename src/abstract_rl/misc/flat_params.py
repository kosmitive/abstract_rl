from typing import Any

import equinox as eqx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float


def get_flat_params_from(model: Any) -> Float[Array, "n"]:
    """Extract flat parameters form an equinox model (every inexact array leaf is a parameter)."""

    flat_params, _ = ravel_pytree(eqx.filter(model, eqx.is_inexact_array))
    return flat_params


def set_flat_params_to(model: Any, flat_params: Float[Array, "n"]) -> Any:
    """Set flat parameters to an equinox model. Models are immutable, so the updated model is returned."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    _, unravel = ravel_pytree(params)
    return eqx.combine(unravel(flat_params), static)
