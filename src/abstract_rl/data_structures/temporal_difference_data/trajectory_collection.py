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

from abstract_rl.src.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.src.data_structures.temporal_difference_data.trajectory import Trajectory
import numpy as np

from abstract_rl.src.misc.cli_printer import CliPrinter


class TrajectoryCollection:
    """Represents a simple trajectory collection."""

    def __init__(self, env, max_l=None):
        self.max_l = max_l
        self.__num_data_points = 0
        self.__trajectories = []
        self._env = env

        self._sd = self._env.observation_dim
        self._ad = self._env.action_dim
        self.cli = CliPrinter().instance

    def __len__(self):
        return len(self.__trajectories)

    def merge(self, collection):
        """Merges an existing collection into this object.

        :param collection: The collection to merge into
        """
        self.extend(collection.trajectories())

    def is_full(self):
        """
        :return: True if is full.
        """
        return len(self.__trajectories) == self.max_l

    def print(self):
        """
        print information
        """

        titstr = "%16s | %10s | %28s | %29s | %10s %10s %10s"
        fmtstr = "%10i [%3i] | %10.5f | a ~ %10.5f +- %10.5f | %10.5f <= r <= %10.5f | %10.5f %10.5f %10.5f"

        self.cli.empty().print(titstr % ("id [len]", "entropy", "action_stats", "reward_stats", "avg_rew", "tot_rew", "disc_rew")).line()
        for traj in self.__trajectories:

            print_str = fmtstr % (traj.traj_id, len(traj), traj.entropy, np.mean(traj.actions), np.std(traj.actions),
                      np.min(traj.rewards), np.max(traj.rewards), traj.average_reward, traj.total_reward, traj.discounted_reward(traj.env.discount()))
            self.cli.print(print_str)

    def extend(self, trajectories):
        """
        Add an existing list of trajectories to the internal list.
        :param trajectories: The trajectories to add.
        """
        self.__trajectories.extend(trajectories)

        k = 0
        i = len(trajectories)
        while k < self.max_l and i > 0:
            i -= 1
            k += len(trajectories[i])

        self.__trajectories = self.__trajectories[i:]
        self.__num_data_points = sum([len(t) for t in self.__trajectories])

    def __iter__(self): return iter(self.__trajectories)

    def trajectories(self): return self.__trajectories

    def to_batch(self, copy_fields=None):
        """Convert the trajectory collection to a batch.

        :param copy_fields: The fields to copy into the batch.
        :return: The final created batch.
        """

        # define basic lists
        states = []
        nxt_states = []
        actions = []
        rewards = []
        lls = []
        ds = []

        # make space for annotations
        annots = {a: [] for a in copy_fields}

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
                annots[a].append(traj[a])

        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        lls = np.concatenate(lls, 0)
        nxt_states = np.concatenate(nxt_states, 0)
        rewards = np.concatenate(rewards, 0)
        ds = np.concatenate(ds, 0)
        for a in copy_fields: annots[a] = np.concatenate(annots[a], 0)

        batch = Batch(states, actions, lls, rewards, nxt_states, ds, annots)
        assert len(batch) > 0
        return batch

    def sample(self, batch_size, copy_fields=None):

        batch_size = min(batch_size, self.__num_data_points)
        tindices = np.random.randint(0, batch_size, size=batch_size)
        tindices.sort()

        # define basic lists
        states = []
        nxt_states = []
        actions = []
        rewards = []
        lls = []
        ds = []

        # make space for annotations
        annots = {a: [] for a in copy_fields}

        # count independently for trajectories and indices
        i = 0
        traj_offset = 0

        # if it should be continued
        for t in range(len(self.__trajectories)):

            # obtain possible length
            traj = self.__trajectories[t]
            traj_len = len(traj) - 7

            # get all possible indices
            j = 0
            while j < traj_len and i+j < len(tindices) and tindices[i+j] - traj_offset < traj_len: j += 1

            # map indices to trajectory
            m_indices = tindices[i:i+j] - traj_offset

            # append infos
            states.append(traj.states[m_indices])
            nxt_states.append(traj.next_states[m_indices])
            actions.append(traj.actions[m_indices])
            rewards.append(traj.rewards[m_indices])
            lls.append(traj.log_likelihoods[m_indices])
            ds.append(traj.dones[m_indices])
            i += j

            # append rest
            for a in copy_fields:
                annots[a].append(traj[a][m_indices])

        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        lls = np.concatenate(lls, 0)
        nxt_states = np.concatenate(nxt_states, 0)
        rewards = np.concatenate(rewards, 0)
        ds = np.concatenate(ds, 0)
        for a in copy_fields: annots[a] = np.concatenate(annots[a], 0)

        batch = Batch(states, actions, lls, rewards, nxt_states, ds, annots)
        assert len(batch) > 0
        return batch
