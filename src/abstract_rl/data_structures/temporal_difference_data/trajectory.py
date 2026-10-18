from typing import Any

import jax
import numpy as np
import pandas as pd

from abstract_rl.misc.cli_printer import CliPrinter


class Trajectory:
    """
    Represents a simple Trajectory, which can be used by the remaining software.
    """

    def __init__(
        self, id: int, env: Any, pandas_frame: pd.DataFrame, annotations: dict[str, Any] | None = None
    ) -> None:
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
        self._states = self.__data_frame.loc[:, "s_0" : f"s_{self._sd - 1}"].to_numpy(copy=True)
        self._rewards = self.__data_frame.loc[:, "r":"r"].to_numpy(copy=True)[:-1]
        # return of the environment, differs from total_reward if the training reward is scaled or shifted
        self.raw_return = float(np.sum(self._rewards))

        # cache the remaining columns as numpy arrays, the pandas frame is only the storage format and slicing it on
        # every access is very slow for off policy sampling
        self._actions = self.__data_frame.loc[:, "a_0" : f"a_{self._ad - 1}"].to_numpy(copy=True)[:-1]
        self._dones = self.__data_frame.loc[:, "d":"d"].to_numpy(copy=True)[:-1]
        self._log_likelihoods = self.__data_frame.loc[:, "ll":"ll"].to_numpy(copy=True)[:-1]

    def __len__(self) -> int:
        return len(self._trajectory.index) - 1

    def __setitem__(self, key: str, item: Any) -> None:
        # annotations are stored as numpy arrays, so sampling from many trajectories stays cheap
        if isinstance(item, jax.Array):
            item = np.asarray(item)
        self._annotations[key] = item

    def __getitem__(self, key: str) -> Any:
        return self._annotations[key]

    def __contains__(self, value: str) -> bool:
        return value in self._annotations

    # --- standard reinforcement learning fields

    @property
    def states(self) -> np.ndarray:
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states[:-1]

    @property
    def actions(self) -> np.ndarray:
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def dones(self) -> np.ndarray:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._dones

    @property
    def log_likelihoods(self) -> np.ndarray:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

    @property
    def rewards(self) -> np.ndarray:
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def next_states(self) -> np.ndarray:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._states[1:]

    @property
    def all_states(self) -> np.ndarray:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._states

    @property
    def __data_frame(self) -> pd.DataFrame:
        """
        Get internal data frame of trajectory.
        :return: Returns the internal saved pandas frame.
        """
        return self._trajectory

    # --- evaluate rewards

    @property
    def average_reward(self) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        T = len(self)
        return np.sum(self.rewards / T)

    @property
    def total_reward(self) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        return np.sum(self.rewards)

    def discounted_reward(self, discount: float) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """

        tl = len(self)
        return (1 - discount) * np.sum(discount ** np.arange(tl) * self.rewards)

    @property
    def entropy(self) -> float:
        """Monte Carlo entropy of the behaviour policy along this episode.

        ``-mean_t log pi(a_t | s_t)``. The stored log-density already includes the tanh
        Jacobian when the action is squashed.
        :return: Per-step entropy estimate.
        """
        return float(-np.mean(self.log_likelihoods))
