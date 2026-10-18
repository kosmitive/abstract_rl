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
