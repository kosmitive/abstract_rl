"""Running observation statistics applied when the policy acts and when it trains.

``filter_states`` used to rewrite stored trajectories after the action had been chosen on the raw
observation. These statistics are the replacement. They are a Welford aggregate (count, mean, second
moment), updated from the observations just collected and held fixed for the following policy and
value update. The same affine is used in the rollout, including the jitted classic-control scan, in
TRPO and MPO, and in greedy evaluation.

The MPO replay stores the unnormalised observations. Each update normalises the arrays it samples
with the statistics frozen for that update, so a row collected under older statistics is not left
stale. The aggregate itself is stored next to ``policy.data``.
"""

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

_EPS = 1e-8


class ObsNorm(eqx.Module):
    """Population moments of the observations the policy sees.

    Variance is the unbiased second moment ``m2 / (count - 1)``. Fewer than two samples, and a
    disabled filter, both leave the observation unchanged. ``clip`` and ``enabled`` are static, so
    a new batch of moments does not recompile the rollout.
    """

    count: Float[Array, ""]
    mean: Float[Array, " obs"]
    m2: Float[Array, " obs"]
    clip: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    enabled: bool = eqx.field(static=True)

    @classmethod
    def create(cls, dim: int, enabled: bool, clip: float, eps: float = _EPS) -> "ObsNorm":
        """Zero moments for an observation of size ``dim``.

        Args:
            dim: Observation dimension.
            enabled: Apply the affine. A disabled filter is the identity.
            clip: Symmetric clip of the normalised observation.
            eps: Added to the standard deviation.

        Returns:
            A filter with ``count == 0``.
        """
        return cls(
            count=jnp.zeros((), dtype=jnp.float32),
            mean=jnp.zeros((dim,), dtype=jnp.float32),
            m2=jnp.zeros((dim,), dtype=jnp.float32),
            clip=float(clip),
            eps=float(eps),
            enabled=bool(enabled),
        )

    def update(self, observations: Float[Array, "... obs"] | np.ndarray) -> "ObsNorm":
        """Merge a batch with Chan's parallel Welford formula.

        Args:
            observations: Rows of observations, shape ``[batch, obs]`` or ``[obs]``.

        Returns:
            A new filter. A disabled filter is returned unchanged.
        """
        if not self.enabled:
            return self
        batch = jnp.asarray(observations, dtype=self.mean.dtype)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if int(batch.shape[0]) == 0:
            return self
        return _merge(self, batch)

    def numpy_affine(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Host mean, inverse standard deviation and clip, or ``None`` when the affine is off.

        The rollout caches this for an epoch. ``None`` means the observation is passed through.
        """
        if not self.enabled:
            return None
        count = float(np.asarray(self.count))
        if count <= 1.0:
            return None
        mean = np.asarray(self.mean, dtype=np.float32)
        variance = np.asarray(self.m2, dtype=np.float32) / np.float32(count - 1.0)
        inverse = np.float32(1.0) / (np.sqrt(variance) + np.float32(self.eps))
        return mean, inverse.astype(np.float32), float(self.clip)

    def rollout_params(
        self,
    ) -> tuple[Float[Array, " obs"], Float[Array, " obs"], Float[Array, ""], Float[Array, ""]]:
        """Arrays for the jitted rollout: mean, inverse std, clip, and a 0/1 ready flag.

        The flag is an array, so the first batch that crosses two samples does not recompile.
        A disabled filter still returns well-shaped arrays; the rollout ignores them.
        """
        mean = self.mean
        inverse = jnp.reciprocal(_std(self))
        clip = jnp.asarray(self.clip, dtype=mean.dtype)
        ready = (self.count > 1).astype(mean.dtype)
        return mean, inverse, clip, ready


def normalize_obs(norm: ObsNorm, observations: Float[Array, "... obs"]) -> Float[Array, "... obs"]:
    """Apply ``norm`` without writing into ``observations``.

    A disabled filter returns the same array. The moments are stopped so an update that traces
    this function cannot move them.

    Args:
        norm: Statistics frozen for this update.
        observations: Observations in the same space the moments were estimated on.

    Returns:
        Normalised observations, clipped to ``[-clip, clip]`` once two samples have been seen.
    """
    if not norm.enabled:
        return observations
    return _normalize(norm, observations)


def normalize_mc(mc: Any, states: Any) -> Float[Array, "... obs"]:
    """Normalise with the live policy stored on ``mc``. The source array is not modified."""
    observed = jnp.asarray(states, dtype=jnp.float32)
    return normalize_obs(mc.get("policy", False).observation_norm(), observed)


def absorb_observations(norm: ObsNorm, trajectories: Sequence[Any]) -> ObsNorm:
    """Fold every state of ``trajectories``, including the bootstrap state, into ``norm``.

    Args:
        norm: Statistics that were held fixed while these trajectories were collected and fitted.
        trajectories: Episodes whose ``all_states`` are the unnormalised observations.

    Returns:
        The updated filter. Called after the policy and value update so that step saw a constant.
    """
    if not norm.enabled or not trajectories:
        return norm
    rows = [np.asarray(trajectory.all_states, dtype=np.float32) for trajectory in trajectories]
    if not rows:
        return norm
    return norm.update(np.concatenate(rows, axis=0))


def obs_norm_path(path: str) -> str:
    """Checkpoint path of the moments stored beside a ``policy.data`` file."""
    return f"{path}.obs_norm"


def _std(norm: ObsNorm) -> Float[Array, " obs"]:
    variance = norm.m2 / jnp.maximum(norm.count - 1.0, 1.0)
    return jnp.sqrt(jnp.maximum(variance, 0.0)) + jnp.asarray(norm.eps, dtype=norm.m2.dtype)


@eqx.filter_jit
def _merge(norm: ObsNorm, batch: Float[Array, "batch obs"]) -> ObsNorm:
    """Chan merge of ``norm`` with the moments of ``batch``. The leading axis is static."""
    batch_count = jnp.asarray(batch.shape[0], dtype=norm.count.dtype)
    batch_mean = jnp.mean(batch, axis=0)
    centered = batch - batch_mean
    batch_m2 = jnp.sum(centered * centered, axis=0)
    new_count = norm.count + batch_count
    delta = batch_mean - norm.mean
    new_mean = norm.mean + delta * (batch_count / new_count)
    new_m2 = norm.m2 + batch_m2 + delta * delta * norm.count * batch_count / new_count
    return eqx.tree_at(lambda tree: (tree.count, tree.mean, tree.m2), norm, (new_count, new_mean, new_m2))


@eqx.filter_jit
def _normalize(norm: ObsNorm, observations: Float[Array, "... obs"]) -> Float[Array, "... obs"]:
    """``(x - mean) / std``, clipped. Fewer than two samples returns ``observations``."""
    ready = norm.count > 1
    mean = jax.lax.stop_gradient(norm.mean)
    inverse = jax.lax.stop_gradient(jnp.reciprocal(_std(norm)))
    clipped = jnp.clip((observations - mean) * inverse, -norm.clip, norm.clip)
    return jnp.where(ready, clipped, observations)
