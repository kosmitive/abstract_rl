"""Several seeds of one run configuration, trained as one vmapped program (spec P6)."""

import copy
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.misc.data_logger import DataLogger

T = TypeVar("T")


class SeedRuns:
    """One run directory, model configuration and logger per seed; models are built and trained batched over seeds."""

    def __init__(self, conf: dict[str, Any], alg: str) -> None:
        seeds = conf.get("seeds") or [conf["seed"]]
        self.seeds = [int(s) for s in seeds]
        self.mcs: list[ModelConfiguration] = []
        for s in self.seeds:
            c = copy.deepcopy(conf)
            c["seed"] = s
            c.setdefault("alg", alg)
            mc = ModelConfiguration(c)
            mc.add_main("log", DataLogger(mc.data_dir))
            self.mcs.append(mc)
        self.mc = self.mcs[0]
        self.keys = jax.vmap(jax.random.key)(jnp.asarray(self.seeds, jnp.uint32))
        # checkpoints are written by one background thread, in submission order (spec P5)
        self._writer = ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future] = []

    def build(self, fn: Callable[[Any], T]) -> T:
        """``fn(key)`` for every seed, batched along a leading seed axis."""
        return eqx.filter_vmap(fn)(self.keys)

    def broadcast(self, tree: T) -> T:
        """The same value for every seed, with a leading seed axis."""
        n = len(self.seeds)
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), tree)

    def state(self) -> ModelState:
        """The batched traced state of the wiring configuration, with target arrays in their own buffers."""
        st = self.mc.state()
        syncables = {k: (o, jax.tree.map(jnp.copy, t)) for k, (o, t) in st.syncables.items()}
        st = eqx.tree_at(lambda s: s.syncables, st, syncables)
        return eqx.tree_at(lambda s: s.overall_step, st, jnp.zeros(len(self.seeds), jnp.int32))

    @staticmethod
    def seed_slice(tree: T, i: int) -> T:
        """The unbatched value of seed ``i``."""
        return jax.tree.map(lambda x: x[i], tree)

    def _submit(self, fn: Callable[..., Any], *args: Any) -> None:
        """Queue a write, first re-raising the error of any write which already failed."""
        for f in [f for f in self._pending if f.done()]:
            self._pending.remove(f)
            f.result()
        self._pending.append(self._writer.submit(fn, *args))

    def store(self, name: str, models_per_seed: dict[str, Any]) -> None:
        """Store the batched models of every seed into its own run directory (asynchronously)."""
        host = jax.device_get(models_per_seed)
        for i, mc in enumerate(self.mcs):
            models = {k: self.seed_slice(v, i) for k, v in host.items()}
            self._submit(mc.store, name, models)

    def store_best(self, i: int, ret: float, epoch: int) -> None:
        """Keep the last stored models of seed ``i`` as its best ones if ``ret`` is a new best return."""
        self._submit(self.mcs[i].store_best, ret, epoch)

    def close(self) -> None:
        """Wait for all pending checkpoint writes, re-raising the first failed one."""
        self._writer.shutdown(wait=True)
        pending, self._pending = self._pending, []
        for f in pending:
            f.result()
