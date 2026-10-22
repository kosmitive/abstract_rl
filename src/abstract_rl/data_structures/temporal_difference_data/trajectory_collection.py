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
from jaxtyping import Array, PRNGKeyArray

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


def _flat(x: Any) -> Any:
    return x.reshape((-1, *x.shape[2:]))


class TrajectoryCollection(eqx.Module):
    """Represents a simple trajectory collection: the lanes [E] of one rollout, each a padded Trajectory of L steps."""

    _trajectories: Trajectory

    def __init__(self, env: Any, trajectories: Trajectory) -> None:
        self._trajectories = trajectories

    def __len__(self) -> int:
        return int(self._trajectories.mask.shape[0])

    def trajectories(self) -> Trajectory:
        return self._trajectories

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
