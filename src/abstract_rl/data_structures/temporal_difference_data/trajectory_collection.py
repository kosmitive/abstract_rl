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

from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.cli_printer import CliPrinter


def draw_indices(num_data_points: int, batch_size: int) -> np.ndarray:
    """Draw the sorted transition indices of one batch from numpy's global RNG, at most every transition once."""
    batch_size = min(batch_size, num_data_points)
    return np.sort(np.random.randint(0, num_data_points, size=batch_size))  # noqa: NPY002


class FlatReplay:
    """
    The transitions of a trajectory collection concatenated once into float32 MLX arrays. Drawing a batch is then one
    gather on the device instead of a walk over the trajectories and a copy from numpy. For the same state of numpy's
    RNG, sample returns the same arrays as TrajectoryCollection.sample(...).to_tree(...).
    """

    def __init__(self, tc: "TrajectoryCollection", fields: list[str]) -> None:
        self.tree = tc.to_batch([]).to_tree([])
        self.tree["annotations"] = {f: tc.annotation(f) for f in fields}
        self.num_data_points = int(self.tree["rewards"].shape[0])
        mx.eval(self.tree)

    def draw(self, batch_size: int) -> np.ndarray:
        return draw_indices(self.num_data_points, batch_size)

    def draw_batches(self, num_batches: int, batch_size: int) -> np.ndarray:
        """
        The indices [num_batches, batch_size] of num_batches calls of draw, also the same state of numpy's RNG after
        it: randint draws its values one after another, so one call for all batches gives the same values.
        """
        batch_size = min(batch_size, self.num_data_points)
        indices = np.random.randint(0, self.num_data_points, size=(num_batches, batch_size))  # noqa: NPY002
        # the same order, sorting int32 is about three times faster
        return np.sort(indices.astype(np.int32), axis=1) if self.num_data_points <= 2**31 else np.sort(indices, axis=1)

    def gather(self, indices: Any) -> dict[str, Any]:
        """The rows at the indices of every array of the tree."""
        indices = mx.array(indices, dtype=mx.int32) if isinstance(indices, np.ndarray) else indices
        return tree_map(lambda v: mx.take(v, indices, axis=0), self.tree)

    def sample(self, batch_size: int) -> dict[str, Any]:
        return self.gather(self.draw(batch_size))


class TrajectoryCollection:
    """Represents a simple trajectory collection."""

    def __init__(self, env: Any, max_l: int | None = None) -> None:
        self.max_l = max_l
        self.__num_data_points = 0
        self.__trajectories: list[Trajectory] = []
        self._env = env

        self._sd = self._env.observation_dim
        self._ad = self._env.action_dim
        self.cli = CliPrinter().instance
        # per annotation: the per-trajectory values and their concatenation on the device
        self._flat: dict[str, tuple[tuple[Any, ...], mx.array]] = {}

    def __len__(self) -> int:
        return len(self.__trajectories)

    def merge(self, collection: "TrajectoryCollection") -> None:
        """Merges an existing collection into this object.

        :param collection: The collection to merge into
        """
        self.extend(collection.trajectories())

    def print(self) -> None:
        """
        print information
        """

        titstr = "%16s | %10s | %28s | %29s | %10s %10s %10s"
        fmtstr = "%10i [%3i] | %10.5f | a ~ %10.5f +- %10.5f | %10.5f <= r <= %10.5f | %10.5f %10.5f %10.5f"

        self.cli.empty().print(
            titstr % ("id [len]", "entropy", "action_stats", "reward_stats", "avg_rew", "tot_rew", "disc_rew")
        ).line()
        for traj in self.__trajectories:
            print_str = fmtstr % (
                traj.traj_id,
                len(traj),
                traj.entropy,
                np.mean(traj.actions),
                np.std(traj.actions),
                np.min(traj.rewards),
                np.max(traj.rewards),
                traj.average_reward,
                traj.total_reward,
                traj.discounted_reward(traj.env.discount()),
            )
            self.cli.print(print_str)

    def extend(self, trajectories: list[Trajectory]) -> None:
        """
        Add an existing list of trajectories to the internal list. If a maximum number of transitions is set, the
        oldest trajectories are dropped until the collection fits again.
        :param trajectories: The trajectories to add.
        """
        self.__trajectories.extend(trajectories)

        if self.max_l is not None:
            k = 0
            i = len(self.__trajectories)
            while i > 0 and k + len(self.__trajectories[i - 1]) <= self.max_l:
                i -= 1
                k += len(self.__trajectories[i])

            # always keep at least the newest trajectory
            i = min(i, len(self.__trajectories) - 1)
            self.__trajectories = self.__trajectories[i:]

        self.__num_data_points = sum([len(t) for t in self.__trajectories])

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.__trajectories)

    def trajectories(self) -> list[Trajectory]:
        return self.__trajectories

    def annotation(self, field: str) -> mx.array:
        """
        An annotation of all trajectories concatenated, as float32 MLX array. If a batched operator stored it with
        store_annotation and no trajectory changed it since, this is the array the operator computed, no copy.
        """
        # the cache keeps the per-trajectory arrays it was built from, so their identities cannot be reused
        rows = tuple(t[field] for t in self.__trajectories)
        cached = self._flat.get(field)
        same = cached is not None and len(cached[0]) == len(rows)
        if cached is not None and same and all(a is b for a, b in zip(cached[0], rows, strict=True)):
            return cached[1]
        from abstract_rl.misc.mixed import to_mx

        flat = mx.concatenate([to_mx(t[field]) for t in self.__trajectories])
        self._flat[field] = (rows, flat)
        return flat

    def store_annotation(self, field: str, flat: mx.array) -> None:
        """Store the rows of an annotation of all trajectories: per trajectory a view of flat, and flat itself."""
        offset = 0
        for t in self.__trajectories:
            t[field] = flat[offset : offset + len(t)]
            offset += len(t)
        assert offset == flat.shape[0]
        self._flat[field] = (tuple(t[field] for t in self.__trajectories), flat)

    def to_batch(self, copy_fields: list[str] | None = None) -> Batch:
        """Convert the trajectory collection to a batch.

        :param copy_fields: The fields to copy into the batch.
        :return: The final created batch.
        """

        # define basic lists
        states: Any = []
        nxt_states: Any = []
        actions: Any = []
        rewards: Any = []
        lls: Any = []
        ds: Any = []

        # make space for annotations
        fields = copy_fields if copy_fields is not None else []
        annots: dict[str, Any] = {a: [] for a in fields}

        # if it should be continued
        for t in range(len(self.__trajectories)):
            # obtain possible length
            traj = self.__trajectories[t]

            # append infos
            states.append(traj.states)
            nxt_states.append(traj.next_states)
            actions.append(traj.actions)
            rewards.append(traj.rewards)
            lls.append(traj.log_likelihoods)
            ds.append(traj.dones)

            # append rest
            for a in fields:
                annots[a].append(traj[a])

        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        lls = np.concatenate(lls, 0)
        nxt_states = np.concatenate(nxt_states, 0)
        rewards = np.concatenate(rewards, 0)
        ds = np.concatenate(ds, 0)
        for a in fields:
            annots[a] = np.concatenate(annots[a], 0)

        batch = Batch(states, actions, lls, rewards, nxt_states, ds, annots)
        assert len(batch) > 0
        return batch

    def flatten(self, copy_fields: list[str] | None = None) -> FlatReplay:
        """A flat copy of the collection on the device, for drawing many batches, see FlatReplay."""
        return FlatReplay(self, copy_fields if copy_fields is not None else [])

    def sample(self, batch_size: int, copy_fields: list[str] | None = None) -> Batch:

        fields = copy_fields if copy_fields is not None else []
        tindices = draw_indices(self.__num_data_points, batch_size)

        # define basic lists
        states: Any = []
        nxt_states: Any = []
        actions: Any = []
        rewards: Any = []
        lls: Any = []
        ds: Any = []

        # make space for annotations
        annots: dict[str, Any] = {a: [] for a in fields}

        # map the global indices onto the trajectories
        traj_offset = 0
        for traj in self.__trajectories:
            traj_len = len(traj)
            lo, hi = np.searchsorted(tindices, [traj_offset, traj_offset + traj_len])
            m_indices = tindices[lo:hi] - traj_offset
            traj_offset += traj_len
            if len(m_indices) == 0:
                continue

            # append infos
            states.append(traj.states[m_indices])
            nxt_states.append(traj.next_states[m_indices])
            actions.append(traj.actions[m_indices])
            rewards.append(traj.rewards[m_indices])
            lls.append(traj.log_likelihoods[m_indices])
            ds.append(traj.dones[m_indices])

            # append rest
            for a in fields:
                annots[a].append(traj[a][m_indices])

        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        lls = np.concatenate(lls, 0)
        nxt_states = np.concatenate(nxt_states, 0)
        rewards = np.concatenate(rewards, 0)
        ds = np.concatenate(ds, 0)
        for a in fields:
            annots[a] = np.concatenate(annots[a], 0)

        batch = Batch(states, actions, lls, rewards, nxt_states, ds, annots)
        assert len(batch) > 0
        return batch
