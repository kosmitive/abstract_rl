class Batch:
    """
    Represents a batch of state, action, reward data tuples.
    """

    def __init__(self, states, actions, log_likelihood, rewards, next_states, ds, annotations=None):
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

    def __len__(self): return len(self.states)

    def __getitem__(self, key): return self._annotations[key]

    def __setitem__(self, key, data): self._annotations[key] = data

    @property
    def states(self):
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states

    @property
    def actions(self):
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def log_likelihoods(self):
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

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
        return self._next_states

    @property
    def dones(self):
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._dones
