from typing import Any

import numpy as np
import pandas as pd

from abstract_rl.misc.cli_printer import CliPrinter


class Trajectory:
    """
    Represents a simple Trajectory, which can be used by the remaining software.
    """

    _trajectory: pd.DataFrame | None

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
        self.raw_return = np.sum(self._rewards)
        # pandas .values is a non-writable view, copy so the arrays can be written in place
        self._actions = self.__data_frame.loc[:, "a_0" : f"a_{self._ad - 1}"].to_numpy(copy=True)[:-1]
        self._dones = self.__data_frame.loc[:, "d":"d"].to_numpy(copy=True)[:-1]
        self._log_likelihoods = self.__data_frame.loc[:, "ll":"ll"].to_numpy(copy=True)[:-1]

    @classmethod
    def from_arrays(
        cls,
        traj_id: int,
        env: Any,
        states: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        log_likelihoods: Any,
        raw_return: float,
    ) -> "Trajectory":
        """A trajectory that already lives in arrays, with no pandas frame.

        ``states`` includes the final next state. ``rewards`` has one row per
        transition, which is also what ``__len__`` reports. The Gymnasium path
        builds that same layout from its frame.

        Args:
            traj_id: Id of this episode.
            env: The environment the episode came from.
            states: Shape ``(T + 1, obs)``.
            actions: Shape ``(T, act)``.
            rewards: Shape ``(T,)`` or ``(T, 1)``.
            dones: Terminations, not truncations. Shape ``(T,)`` or ``(T, 1)``.
            log_likelihoods: Shape ``(T,)`` or ``(T, 1)``.
            raw_return: The unscaled sum of environment rewards.

        Returns:
            The trajectory. The arrays are stored as given, so an MLX view stays a view.
        """
        trajectory = cls.__new__(cls)
        trajectory.traj_id = traj_id
        trajectory._annotations = {}
        trajectory._trajectory = None
        trajectory.env = env
        trajectory._sd = env.observation_dim
        trajectory._ad = env.action_dim
        trajectory.cli = CliPrinter().instance
        trajectory._states = states
        trajectory._actions = actions
        trajectory._rewards = rewards
        trajectory._dones = dones
        trajectory._log_likelihoods = log_likelihoods
        trajectory.raw_return = float(raw_return)
        return trajectory

    def __len__(self) -> int:
        return int(self._rewards.shape[0])

    def __setitem__(self, key: str, item: Any) -> None:
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
        if self._trajectory is None:
            raise RuntimeError("this trajectory was built from arrays and has no pandas frame")
        return self._trajectory

    # --- evaluate rewards

    @property
    def average_reward(self) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        length = len(self)
        return float(np.sum(np.asarray(self.rewards) / length))

    @property
    def total_reward(self) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """
        return float(np.sum(np.asarray(self.rewards)))

    def discounted_reward(self, discount: float) -> float:
        """
        Cumulative reward of the trajectory.
        :return: Simply sum of all rewards.
        """

        length = len(self)
        return float((1 - discount) * np.sum(discount ** np.arange(length) * np.asarray(self.rewards).reshape(-1)))

    @property
    def entropy(self) -> float:
        """Calculate entropy of policy.
        :return: The calculated entropy.
        """
        return float(-np.mean(np.asarray(self.log_likelihoods)))
