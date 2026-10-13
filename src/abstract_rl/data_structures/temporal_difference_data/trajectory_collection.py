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

import numpy as np

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.cli_printer import CliPrinter


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

    def to_batch(self, copy_fields: list[str] | None = None) -> Batch:
        """Convert the trajectory collection to a batch.

        :param copy_fields: The fields to copy into the batch.
        :return: The final created batch.
        """

        # define basic lists
        states: list[np.ndarray] = []
        nxt_states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        lls: list[np.ndarray] = []
        ds: list[np.ndarray] = []

        # make space for annotations
        copy_fields = copy_fields if copy_fields is not None else []
        annots: dict[str, list[np.ndarray]] = {a: [] for a in copy_fields}

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
            for a in copy_fields:
                annots[a].append(np.asarray(traj[a]))

        states_all = np.concatenate(states, 0)
        actions_all = np.concatenate(actions, 0)
        lls_all = np.concatenate(lls, 0)
        nxt_states_all = np.concatenate(nxt_states, 0)
        rewards_all = np.concatenate(rewards, 0)
        ds_all = np.concatenate(ds, 0)
        annots_all = {a: np.concatenate(annots[a], 0) for a in copy_fields}
        batch = Batch(states_all, actions_all, lls_all, rewards_all, nxt_states_all, ds_all, annots_all)
        assert len(batch) > 0
        return batch

    def sample(self, batch_size: int, copy_fields: list[str] | None = None) -> Batch:
        """Sample a batch of transitions uniformly (with replacement) from all stored trajectories.

        :param batch_size: The number of transitions to sample.
        :param copy_fields: The annotation fields to copy into the batch.
        :return: The sampled batch.
        """

        copy_fields = copy_fields if copy_fields is not None else []
        batch_size = min(batch_size, self.__num_data_points)
        tindices = np.sort(np.random.randint(0, self.__num_data_points, size=batch_size))

        # define basic lists
        states: list[np.ndarray] = []
        nxt_states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        lls: list[np.ndarray] = []
        ds: list[np.ndarray] = []

        # make space for annotations
        annots: dict[str, list[np.ndarray]] = {a: [] for a in copy_fields}

        # map the global indices onto the trajectories
        traj_offset = 0
        for traj in self.__trajectories:
            traj_len = len(traj)
            lo, hi = np.searchsorted(tindices, [traj_offset, traj_offset + traj_len])
            m_indices = tindices[lo:hi] - traj_offset
            traj_offset += traj_len
            if len(m_indices) == 0:
                continue

            # append infos (states and next states share the same underlying array)
            all_states = traj.all_states
            states.append(all_states[m_indices])
            nxt_states.append(all_states[m_indices + 1])
            actions.append(traj.actions[m_indices])
            rewards.append(traj.rewards[m_indices])
            lls.append(traj.log_likelihoods[m_indices])
            ds.append(traj.dones[m_indices])

            # append rest
            for a in copy_fields:
                annots[a].append(traj[a][m_indices])

        states_all = np.concatenate(states, 0)
        actions_all = np.concatenate(actions, 0)
        lls_all = np.concatenate(lls, 0)
        nxt_states_all = np.concatenate(nxt_states, 0)
        rewards_all = np.concatenate(rewards, 0)
        ds_all = np.concatenate(ds, 0)
        annots_all = {a: np.concatenate(annots[a], 0) for a in copy_fields}
        batch = Batch(states_all, actions_all, lls_all, rewards_all, nxt_states_all, ds_all, annots_all)
        assert len(batch) > 0
        return batch
