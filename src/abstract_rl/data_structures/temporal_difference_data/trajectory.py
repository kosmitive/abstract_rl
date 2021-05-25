import numpy as np

from abstract_rl.src.misc.cli_printer import CliPrinter


class Trajectory:
    """
    Represents a simple Trajectory, which can be used by the remaining software.
    """

    def __init__(self, id, env, pandas_frame, annotations=None):
        """Initialize a new trajectory, therefore a initial state s0
        has to be given.

        :param pandas_frame: The pandas __data frame to use.
        """
        # the list for collecting
        self.traj_id = id
        self._annotations = annotations if annotations is not None else {}
        self._trajectory = pandas_frame
        self.env = env
        self._sd = self.env.observation_dim
        self._ad = self.env.action_dim
        self.cli = CliPrinter().instance
        self._states = self.__data_frame.loc[:, "s_0":f"s_{self._sd-1}"].values
        self._rewards = self.__data_frame.loc[:, "r":"r"].values[:-1]

    def __len__(self):
        return len(self._trajectory.index) - 1

    def __setitem__(self, key, item): self._annotations[key] = item

    def __getitem__(self, key): return self._annotations[key]

    def __contains__(self, value): return value in self._annotations

    # --- standard reinforcement learning fields

    def list_annotations(self): return self._annotations.keys()

    @property
    def state_dim(self): return self._sd

    @property
    def act_dim(self): return self._ad

    @property
    def states(self):
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states[:-1]

    @property
    def actions(self):
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self.__data_frame.loc[:, "a_0":f"a_{self._ad-1}"].values[:-1]

    @property
    def dones(self):
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self.__data_frame.loc[:, "d":"d"].values[:-1]
    @property
    def log_likelihoods(self):
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self.__data_frame.loc[:, "ll":"ll"].values[:-1]

    @property
    def rewards(self):
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def next_states(self):
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._states[1:]

    @property
    def all_states(self):
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._states

    @property
    def __data_frame(self):
        """
        Get internal data frame of trajectory.
        :return: Returns the internal saved pandas frame.
        """
        return self._trajectory

    # --- evaluate rewards

    @property
    def average_reward(self):
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        T = len(self)
        return np.sum(self.rewards / T)

    @property
    def total_reward(self):
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        return np.sum(self.rewards)

    def discounted_reward(self, discount):
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """

        tl = len(self)
        return (1 - discount) * np.sum(discount ** np.arange(tl) * self.rewards)

    @property
    def entropy(self):
        """Calculate entropy of policy.
        :return: The calculated entropy.
        """
        return -np.sum(self.log_likelihoods * np.exp(self.log_likelihoods))