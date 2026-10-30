from typing import Any

from numpy import ndarray


class Batch:
    """
    Represents a batch of state, action, reward data tuples.
    """

    def __init__(
        self,
        states: ndarray,
        actions: ndarray,
        log_likelihood: ndarray,
        rewards: ndarray,
        next_states: ndarray,
        ds: ndarray,
        annotations: dict[str, ndarray] | None = None,
    ) -> None:
        """
        Creates a new batch built upon the passed values.
        :param states: The states this batch contains.
        :param actions: The actions this batch contains.
        :param log_likelihood: The log likelihoods this batch contains.
        :param rewards: The rewards this batch contains.
        :param next_states: The next states this batch contains.
        :param annotations: The annotations this batch contains.
        """
        self._annotations = annotations if annotations is not None else {}
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._next_states = next_states
        self._dones = ds
        self._log_likelihoods = log_likelihood

    # --- conversion into a tree of MLX arrays, used as the input of compiled functions

    def to_tree(self, fields: list[str] | None = None) -> dict[str, Any]:
        """
        Convert the batch (with the selected annotations) into a dictionary of float32 MLX arrays. This is the one
        copy from numpy per batch.
        :param fields: The annotations to convert, all of them if None.
        :return: A tree of arrays, see from_tree.
        """
        from abstract_rl.misc.mixed import to_mx

        keys = list(self._annotations) if fields is None else fields
        return {
            "states": to_mx(self._states),
            "actions": to_mx(self._actions),
            "log_likelihoods": to_mx(self._log_likelihoods),
            "rewards": to_mx(self._rewards),
            "next_states": to_mx(self._next_states),
            "dones": to_mx(self._dones),
            "annotations": {k: to_mx(self._annotations[k]) for k in keys},
        }

    @staticmethod
    def from_tree(tree: dict[str, Any]) -> "Batch":
        """Build a batch, whose fields are MLX arrays, from a tree created by to_tree."""
        return Batch(
            tree["states"],
            tree["actions"],
            tree["log_likelihoods"],
            tree["rewards"],
            tree["next_states"],
            tree["dones"],
            dict(tree["annotations"]),
        )

    # --- some syntactic sugar

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key: str) -> ndarray:
        return self._annotations[key]

    def __setitem__(self, key: str, data: ndarray) -> None:
        self._annotations[key] = data

    @property
    def states(self) -> ndarray:
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states

    @property
    def actions(self) -> ndarray:
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def log_likelihoods(self) -> ndarray:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

    @property
    def rewards(self) -> ndarray:
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def next_states(self) -> ndarray:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._next_states

    @property
    def dones(self) -> ndarray:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._dones
