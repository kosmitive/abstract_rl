"""Running mean / standard deviation filter for states and rewards.

Adapted from the running statistics in John Schulman's modular_rl (https://github.com/joschu/modular_rl) and Ilya
Kostrikov's pytorch-trpo (https://github.com/ikostrikov/pytorch-trpo/blob/master/running_state.py), which use the
update of https://www.johndcook.com/blog/standard_deviation/.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from abstract_rl.misc.typing import DTYPE


class RunningCenteredFilter(eqx.Module):
    _n: Int[Array, ""]
    _m: Float[Array, " dim"]
    _s: Float[Array, " dim"]
    _name: str = eqx.field(static=True)
    _dim: int = eqx.field(static=True)
    _clip: float = eqx.field(static=True)
    _std_dev_noise: float = eqx.field(static=True)

    def __init__(self, name: str, dim: int, std_dev_noise: float = 1e-8, clip: int = 5) -> None:

        # build running variables
        self._n = jnp.asarray(0, jnp.int32)
        self._dim = dim
        self._m = jnp.zeros(dim, DTYPE)
        self._s = jnp.zeros(dim, DTYPE)
        self._name = name
        self._clip = float(clip)
        self._std_dev_noise = std_dev_noise

    def register(
        self, xs: Float[np.ndarray | Array, "..."], mask: Bool[Array, " N"] | None = None
    ) -> "RunningCenteredFilter":
        """Welford update with every (valid) row of ``xs``, in order. Returns the updated filter."""
        xs = jnp.asarray(xs, DTYPE).reshape([-1, self._dim])
        mask = jnp.ones(xs.shape[0], bool) if mask is None else mask

        def one(carry: tuple[Any, Any, Any], row: tuple[Any, Any]) -> tuple[tuple[Any, Any, Any], None]:
            n, m, s = carry
            x, valid = row
            n_new = n + 1
            # trivial
            first = n_new == 1
            # update
            old_mean = m
            m_new = jnp.where(first, x, old_mean + (x - old_mean) / n_new)
            s_new = jnp.where(first, s, s + (x - old_mean) * (x - m_new))
            return (jnp.where(valid, n_new, n), jnp.where(valid, m_new, m), jnp.where(valid, s_new, s)), None

        (n, m, s), _ = jax.lax.scan(one, (self._n, self._m, self._s), (xs, mask))
        return eqx.tree_at(lambda f: (f._n, f._m, f._s), self, (n, m, s))

    def __call__(self, x: Float[np.ndarray | Array, "..."], update: bool = False) -> Float[Array, "..."]:
        if update:
            raise ValueError("register returns the updated filter, call it before normalizing")
        x = (jnp.asarray(x, DTYPE) - self.mean) / (self.std + self._std_dev_noise)
        x = jnp.clip(x, -self._clip, self._clip)
        return x

    @property
    def mean(self) -> Float[Array, " dim"]:
        return self._m

    @property
    def var(self) -> Float[Array, " dim"]:
        return jnp.where(self._n > 1, self._s / jnp.maximum(self._n - 1, 1), self._m * self._m)

    @property
    def std(self) -> Float[Array, " dim"]:
        return jnp.sqrt(self.var)
