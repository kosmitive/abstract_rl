"""Flatten and restore Equinox parameter trees."""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float


def ravel_module[M: eqx.Module](
    model: M,
) -> tuple[Float[Array, " params"], Callable[[Float[Array, " params"]], M]]:
    """Flatten the floating-point leaves of an Equinox module.

    Args:
        model: Module whose inexact arrays are trainable weights.

    Returns:
        The flat parameter vector and a function that writes a vector of the same size
        back into a module of the same structure.
    """
    diff, static = eqx.partition(model, eqx.is_inexact_array)
    flat, unravel_diff = jax.flatten_util.ravel_pytree(diff)

    def unravel(vector: Float[Array, " params"]) -> M:
        return cast(M, eqx.combine(unravel_diff(vector), static))

    return flat, unravel


def l2_norm(model: eqx.Module) -> Float[Array, ""]:
    """L2 norm of every floating-point leaf, matching ``torch.norm`` on the flat vector."""
    leaves = [leaf.reshape(-1) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))]
    if not leaves:
        return jnp.zeros(())
    return jnp.linalg.norm(jnp.concatenate(leaves))


def copy_module[M: eqx.Module](model: M) -> M:
    """Copy array leaves so a target network does not share storage with the live one."""

    def _copy(value: object) -> object:
        if eqx.is_array(value):
            return jnp.array(value)
        return value

    return cast(M, jax.tree.map(_copy, model))
