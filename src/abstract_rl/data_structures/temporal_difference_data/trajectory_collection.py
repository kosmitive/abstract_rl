#  MIT License
#
#  Copyright (c) 2019 Markus Semmler
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


def _flat(x: Any) -> Any:
    return x.reshape((-1, *x.shape[2:]))


class TrajectoryCollection(eqx.Module):
    """Represents a simple trajectory collection: the lanes [E] of one rollout, each a padded Trajectory of L steps."""

    _trajectories: Trajectory
    _ptr: Any = None
    _inserted: Any = None
    _high_water: Any = None

    def __init__(
        self,
        env: Any,
        trajectories: Trajectory,
        *,
        ptr: Any = None,
        inserted: Any = None,
        high_water: Any = None,
    ) -> None:
        self._trajectories = trajectories
        self._ptr = ptr
        self._inserted = inserted
        self._high_water = high_water

    def __len__(self) -> int:
        return int(self._trajectories.mask.shape[0])

    def trajectories(self) -> Trajectory:
        return self._trajectories

    def occupied_rows(self) -> Int[Array, ""]:
        """The number of leading memory rows that were ever written: the insert pointer until the ring first wraps,
        afterwards the highest pointer reached. All valid rows lie below it. A collection that is not a replay memory
        is occupied throughout: all rows of a lane."""
        if self._high_water is None:
            return jnp.asarray(self._trajectories.mask.shape[1], jnp.int32)
        return self._high_water

    def with_trajectories(self, trajectories: Trajectory) -> "TrajectoryCollection":
        return eqx.tree_at(lambda c: c._trajectories, self, trajectories)

    def to_batch(self, copy_fields: list[str] | None = None) -> Batch:
        """Convert the trajectory collection to a batch.

        :param copy_fields: The fields to copy into the batch.
        :return: The final created batch, padded steps are masked.
        """
        t = self._trajectories
        fields = copy_fields if copy_fields is not None else []
        annots = {a: _flat(t[a]) for a in fields}
        return Batch(
            _flat(t.states),
            _flat(t.actions),
            _flat(t.log_likelihoods),
            _flat(t.rewards),
            _flat(t.next_states),
            _flat(t.dones),
            annots,
            mask=_flat(t.mask),
        )

    def sample(self, key: PRNGKeyArray, batch_size: int, copy_fields: list[str] | None = None) -> Batch:
        """Uniformly sample ``batch_size`` valid transitions with replacement."""
        t = self._trajectories
        fields = copy_fields if copy_fields is not None else []
        mask = _flat(t.mask)
        p = mask.astype(jnp.float32) / jnp.sum(mask)
        tindices = jax.random.choice(key, mask.shape[0], (batch_size,), replace=True, p=p)

        def take(x: Any) -> Any:
            return _flat(x)[tindices]

        annots = {a: take(t[a]) for a in fields}
        return Batch(
            take(t.states),
            take(t.actions),
            take(t.log_likelihoods),
            take(t.rewards),
            take(t.next_states),
            take(t.dones),
            annots,
        )

    def episode_stats(self) -> dict[str, Array]:
        """Raw return and length of every episode: one slot per possible episode, ``valid`` marks the real ones."""
        t = self._trajectories
        e, length = t.mask.shape
        ends = t.ends & t.mask
        episode = jnp.cumsum(ends, axis=1) - ends + jnp.arange(e)[:, None] * length
        seg = episode.reshape(-1)
        raw = jnp.where(t.mask, t.raw_rewards, 0.0).reshape(-1)
        returns = jax.ops.segment_sum(raw, seg, num_segments=e * length)
        lengths = jax.ops.segment_sum(t.mask.reshape(-1).astype(jnp.int32), seg, num_segments=e * length)
        valid = jax.ops.segment_sum(ends.reshape(-1).astype(jnp.int32), seg, num_segments=e * length) > 0
        return {"returns": returns, "lengths": lengths, "valid": valid}

    @classmethod
    def memory(cls, env: Any, capacity: int, obs_dim: int, act_dim: int) -> "TrajectoryCollection":
        """An empty replay memory: one lane of ``capacity`` invalid rows (MPO, spec section 6.2)."""
        z = jnp.zeros((1, capacity, 1), jnp.float32)
        trajectories = Trajectory(
            jnp.zeros((1, capacity, obs_dim)),
            jnp.zeros((1, capacity, obs_dim)),
            jnp.zeros((1, capacity, act_dim)),
            z,
            jnp.zeros((1, capacity)),
            z,
            z,
            jnp.zeros((1, capacity), bool),
            jnp.zeros((1, capacity), bool),
            {"insert_id": jnp.full((1, capacity), -1, jnp.int32), "time": jnp.zeros((1, capacity), jnp.int32)},
        )
        zero = jnp.asarray(0, jnp.int32)
        return cls(env, trajectories, ptr=zero, inserted=zero, high_water=zero)

    def with_annotations(self, template: dict[str, Any]) -> "TrajectoryCollection":
        """Zero-filled annotations with the given per-row shapes, e.g. the output structure of the MPO operators."""
        t = self._trajectories
        lanes, length = t.mask.shape
        for k, spec in template.items():
            t = t.set(k, jnp.zeros((lanes, length, *spec.shape), spec.dtype))
        return self.with_trajectories(t)

    def merge(self, collection: "TrajectoryCollection", max_l: int) -> "TrajectoryCollection":
        """Merges the valid rows of a rollout collection into this replay memory and drops the oldest episodes until
        at most ``max_l`` transitions remain (the newest episode is always kept).

        The annotations of the merged rows are stale (the rollout's, or the overwritten rows' own): the operators
        recompute every annotation each epoch, so nothing reads them before that.

        :param collection: The rollout to merge, lanes of whole episodes with at most ``capacity`` valid rows.
        :param max_l: The maximum number of transitions.
        """
        mem = jax.tree.map(lambda x: x[0], self._trajectories)
        new = jax.tree.map(_flat, collection.trajectories())
        capacity = mem.mask.shape[0]
        n_new = new.mask.shape[0]
        slots = jnp.arange(capacity)

        # compact the valid rows of the rollout, lane by lane, so episodes stay contiguous
        order = jnp.argsort(~new.mask, stable=True)
        count = jnp.sum(new.mask)
        ptr = jnp.where(self._ptr + count > capacity, 0, self._ptr)
        write = jnp.arange(n_new) < count
        rows = jnp.where(write, ptr + jnp.arange(n_new), capacity)  # out of bounds rows are dropped

        # a kept episode that the insert partly overwrites is dropped as a whole; it only drops episodes that capping
        # would drop anyway, because the insert is newer than every kept episode and the ring holds tc_size + 2R rows
        written = (slots >= ptr) & (slots < ptr + count) & mem.mask
        old_episode = jnp.cumsum(mem.ends) - mem.ends
        hit = jax.ops.segment_max(written.astype(jnp.int32), old_episode, num_segments=capacity) > 0
        mem = mem.replace(mask=mem.mask & ~hit[old_episode])

        def insert(old: Any, value: Any) -> Any:
            return old.at[rows].set(value[order], mode="drop")

        fields = {
            "states": new.states,
            "next_states": new.next_states,
            "actions": new.actions,
            "rewards": new.rewards,
            "raw_rewards": new.raw_rewards,
            "dones": new.dones,
            "log_likelihoods": new.log_likelihoods,
            "ends": new.ends,
            "mask": new.mask,
        }
        mem = mem.replace(**{k: insert(getattr(mem, k), v) for k, v in fields.items()})
        mem = mem.set("time", insert(mem["time"], new["time"]))
        mem = mem.set("insert_id", insert(mem["insert_id"], jnp.full(n_new, self._inserted, jnp.int32)))

        # capping: newest episodes first (insert id, then the time of their last step, then the segment index, i.e. the
        # slot order of the episodes), longest prefix within max_l
        ends = mem.ends & mem.mask
        episode = jnp.cumsum(mem.ends) - mem.ends
        length = jax.ops.segment_sum(mem.mask.astype(jnp.int32), episode, num_segments=capacity)
        valid = jax.ops.segment_sum(ends.astype(jnp.int32), episode, num_segments=capacity) > 0
        inserted = jax.ops.segment_max(jnp.where(ends, mem["insert_id"], -1), episode, num_segments=capacity)
        time = jax.ops.segment_max(jnp.where(ends, mem["time"], -1), episode, num_segments=capacity)
        inserted = jnp.where(valid, inserted, -1)
        newest_first = jnp.lexsort((slots, time, inserted))[::-1]
        cum = jnp.cumsum(length[newest_first] * valid[newest_first])
        keep_sorted = valid[newest_first] & ((cum <= max_l) | (slots == 0))
        keep = jnp.zeros(capacity, bool).at[newest_first].set(keep_sorted)
        mem = mem.replace(mask=mem.mask & keep[episode])

        out = self.with_trajectories(jax.tree.map(lambda x: x[None], mem))
        high_water = jnp.maximum(self._high_water, ptr + count).astype(jnp.int32)
        return eqx.tree_at(
            lambda c: (c._ptr, c._inserted, c._high_water), out, (ptr + count, self._inserted + 1, high_water)
        )
