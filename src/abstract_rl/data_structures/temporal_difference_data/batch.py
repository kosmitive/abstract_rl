from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def _broadcast_mask(x: Array, mask: Bool[Array, "N"]) -> Array:
    return jnp.broadcast_to(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), x.shape)


def masked_mean(x: Array, mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    """Mean over all entries whose leading index is valid. Padded entries never reach the value or the gradient."""
    if mask is None:
        return jnp.mean(x)
    m = _broadcast_mask(x, mask)
    return jnp.sum(jnp.where(m, x, 0.0)) / jnp.sum(m)


def masked_std(x: Array, mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    """Unbiased standard deviation (torch ``Tensor.std()``) over the valid entries."""
    if mask is None:
        return jnp.std(x, ddof=1)
    m = _broadcast_mask(x, mask)
    xs = jnp.where(m, x, 0.0)  # mask first, so padded NaN/inf never enter the arithmetic or its backward pass
    mean = jnp.sum(xs) / jnp.sum(m)
    return jnp.sqrt(jnp.sum(jnp.where(m, (xs - mean) ** 2, 0.0)) / (jnp.sum(m) - 1))


def batch_mask(batch: Any) -> Any:
    """The valid rows of a batch; objects without a mask (light stubs with only ``states``) count as fully valid."""
    return getattr(batch, "mask", None)


def masked_reduce(x: Array, mask: Bool[Array, "N"] | None, est: str) -> Float[Array, ""]:
    """``est`` = "sum" or "mean" over the valid entries."""
    if est == "sum":
        return jnp.sum(x) if mask is None else jnp.sum(jnp.where(_broadcast_mask(x, mask), x, 0.0))
    elif est == "mean":
        return masked_mean(x, mask)
    raise NotImplementedError(est)


class Batch(eqx.Module):
    """
    Represents a batch of state, action, reward data tuples. ``mask`` marks the valid rows of a padded batch, None if
    every row is valid.
    """

    _states: Any
    _actions: Any
    _log_likelihoods: Any
    _rewards: Any
    _next_states: Any
    _dones: Any
    _annotations: dict[str, Any]
    _mask: Any

    def __init__(
        self,
        states: Any,
        actions: Any,
        log_likelihood: Any,
        rewards: Any,
        next_states: Any,
        ds: Any,
        annotations: dict[str, Any] | None = None,
        mask: Any = None,
    ) -> None:
        """
        Creates a new batch built upon the passed values.
        :param states: The states this batch contains.
        :param actions: The actions this batch contains.
        :param log_likelihood: The log likelihoods this batch contains.
        :param rewards: The rewards this batch contains.
        :param next_states: The next states this batch contains.
        :param annotations: The annotations this batch contains.
        :param mask: The valid rows, None if all rows are valid.
        """
        self._annotations = dict(annotations) if annotations is not None else {}
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._next_states = next_states
        self._dones = ds
        self._log_likelihoods = log_likelihood
        self._mask = mask

    # --- some syntactic sugar

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key: str) -> Any:
        return self._annotations[key]

    @property
    def mask(self) -> Any:
        """
        Valid rows of the batch.
        :return: A boolean vector [N] or None if every row is valid.
        """
        return self._mask

    @property
    def states(self) -> Any:
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states

    @property
    def actions(self) -> Any:
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def log_likelihoods(self) -> Any:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

    @property
    def rewards(self) -> Any:
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def next_states(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._next_states

    @property
    def dones(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._dones
