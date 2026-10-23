"""Traced counterpart of ModelConfiguration: the models, optimiser states and buffers of one seed."""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


class ModelState(eqx.Module):
    """
    A pytree with the accessors of ModelConfiguration for everything jitted code needs. ``data`` holds the main elements
    (optimiser states, buffers),
    ``syncables`` the (online, target) pair of every sync able. ``get`` and ``set`` take the
    same keys and the same ``target`` flag as ``ModelConfiguration.get``, so ``self.mc.get(...)`` becomes
    ``state.get(...)``.
    """

    data: dict[str, Any]
    syncables: dict[str, tuple[Any, Any]]
    sync_steps: tuple[tuple[str, int], ...] = eqx.field(static=True)
    # a scalar per seed, "*seeds" so a host copy batched over seeds is a valid state too
    overall_step: Int[Array, "*seeds"]

    @classmethod
    def create(
        cls, data: dict[str, Any], syncables: dict[str, tuple[Any, Any]], sync_steps: dict[str, int] | None = None
    ) -> "ModelState":
        steps = sync_steps if sync_steps is not None else dict.fromkeys(syncables, 1)
        return cls(dict(data), dict(syncables), tuple((k, int(steps[k])) for k in syncables), jnp.asarray(0, jnp.int32))

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def get(self, key: str, target: bool = False) -> Any:
        """
        Access a specific key, the target instance of a sync able if ``target`` is True.
        """
        ti = 0 if not target else 1
        return self.data[key] if key in self.data else self.syncables[key][ti]

    def set(self, key: str, value: Any, target: bool = False) -> "ModelState":
        """A new state where ``key`` (its target if ``target`` is True) is replaced by ``value``."""
        if key in self.data:
            return dataclasses.replace(self, data={**self.data, key: value})
        online, t_value = self.syncables[key]
        pair = (online, value) if target else (value, t_value)
        return dataclasses.replace(self, syncables={**self.syncables, key: pair})

    def sync(self) -> "ModelState":
        """
        ModelConfiguration.sync: the target of every sync able is replaced by the online instance if
        ``overall_step % sync_steps == 0``, afterwards ``overall_step`` is incremented.
        """
        syncables = dict(self.syncables)
        for k, s in self.sync_steps:
            o, to = syncables[k]
            synced = o.sync(to)
            do_sync = self.overall_step % s == 0
            syncables[k] = (o, jax.tree.map(lambda a, b, do=do_sync: jnp.where(do, a, b), synced, to))
        return dataclasses.replace(self, syncables=syncables, overall_step=self.overall_step + 1)
