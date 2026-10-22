"""Observation normalisation statistics of Gymnasium's ``NormalizeObservation`` (``RunningMeanStd``) as a pytree."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from abstract_rl.misc.typing import DTYPE


class RunningMeanStd(eqx.Module):
    mean: Float[Array, " dim"]
    var: Float[Array, " dim"]
    count: Float[Array, ""]

    @classmethod
    def create(cls, dim: int, epsilon: float = 1e-4) -> "RunningMeanStd":
        return cls(jnp.zeros(dim, DTYPE), jnp.ones(dim, DTYPE), jnp.asarray(epsilon, DTYPE))

    def update(self, xs: Float[np.ndarray | Array, "N dim"], mask: Bool[Array, " N"] | None = None) -> "RunningMeanStd":
        """Parallel mean / variance update with the valid rows (Gymnasium ``update_mean_var_count_from_moments``)."""
        xs = jnp.asarray(xs, DTYPE)
        mask = jnp.ones(xs.shape[0], bool) if mask is None else mask
        w = mask.astype(DTYPE)[:, None]
        batch_count = jnp.sum(mask).astype(DTYPE)
        safe = jnp.maximum(batch_count, 1.0)
        batch_mean = jnp.sum(w * jnp.where(mask[:, None], xs, 0.0), axis=0) / safe
        batch_var = jnp.sum(w * jnp.where(mask[:, None], (xs - batch_mean) ** 2, 0.0), axis=0) / safe

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        any_valid = batch_count > 0
        return RunningMeanStd(
            jnp.where(any_valid, new_mean, self.mean), jnp.where(any_valid, new_var, self.var), tot_count
        )

    def normalize(self, x: Float[np.ndarray | Array, "..."], epsilon: float = 1e-8) -> Float[Array, "..."]:
        return (jnp.asarray(x, DTYPE) - self.mean) / jnp.sqrt(self.var + epsilon)
