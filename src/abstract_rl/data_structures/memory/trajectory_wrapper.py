import numpy as np


class TrajectoryWrapper:
    """
    Simple trajectory based access wrapper to the replay memory.
    """

    def __init__(self, memory, s, e, tail_batch):
        """
        Initializes a new trajectory wrapper, with the given position and passed tail.
        :param memory: The memory to access.
        :param s: The start index in the ring buffer.
        :param e: THe end index in the ring buffer.
        :param tail_batch: The tail batch not directly saved in the memory.
        """
        self.memory = memory
        self.s = s
        self.e = e
        self.tail_batch = tail_batch

    # --- some syntactic sugar

    def __len__(self):
        """
        Create the length of the trajectory wrapper.
        :return: Depending on the start and end index calculate the right length.
        """
        if self.e > self.s: return self.e - self.s + self.len_tail()
        else: return self.memory.size - self.s + self.e + self.len_tail()

        # overwrite build in functions to add annotations

    def len_tail(self):
        """Return length of tail.
        :return: Simply call len on the tail batch.
        """
        return len(self.tail_batch)

    # --- standard reinforcement learning fields

    def __setitem__(self, key, item):
        """
        Set a key to a specifc value, e.g. for annotation.
        :param key: The key to set a item.
        :param item: The values to save
        """
        self.set_field(key, item)

    def __getitem__(self, key):
        """Get a key, e.g. for annotation.
        :param key: The key to set a item.
        """
        return self.get_field(key)

    def init_field(self, key, dim):
        self.tail_batch.init(key, dim)

    def set_field(self, key, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set(key, self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set(key, self.s, self.e, item[:-bl])
            self.tail_batch[key] = item[-bl:]

    def get_field(self, key, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            return self.memory.get(key, self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get(key, self.s, self.e),
                    self.tail_batch[key]
                ), axis=0
            )

    def set_actions(self, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set('actions', self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set('actions', self.s, self.e, item[:-bl])
            self.tail_batch['actions'] = item[-bl:]

    def get_actions(self, short=False):
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            return self.memory.get('actions', self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get('actions', self.s, self.e),
                    self.tail_batch.actions
                ), axis=0
            )

    def set_log_likelihoods(self, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set('log_likelihoods', self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set('log_likelihoods', self.s, self.e, item[:-bl])
            self.tail_batch['log_likelihoods'] = item[-bl:]

    def get_log_likelihoods(self, short=False):
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        :param short: Get fields of short trajectory.
        """
        if short:
            return self.memory.get('log_likelihoods', self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get('log_likelihoods', self.s, self.e),
                    self.tail_batch.log_likelihoods
                ), axis=0
            )

    def set_rewards(self, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set('rewards', self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set('rewards', self.s, self.e, item[:-bl])
            self.tail_batch['rewards'] = item[-bl:]

    def get_rewards(self, short=False):
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            return self.memory.get('rewards', self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get('rewards', self.s, self.e),
                    self.tail_batch.rewards
                ), axis=0
            )

    def set_next_states(self, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set('next_states', self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set('next_states', self.s, self.e, item[:-bl])
            self.tail_batch['next_states'] = item[-bl:]


    def get_next_states(self, short=False):
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            return self.memory.get('next_states', self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get('next_states', self.s, self.e),
                    self.tail_batch.next_states
                ), axis=0
            )

    @property
    def states(self):
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self.get_states()

    @property
    def actions(self):
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self.get_actions()

    @property
    def log_likelihoods(self):
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self.get_log_likelihoods()

    @property
    def rewards(self):
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self.get_rewards()

    @property
    def next_states(self):
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self.get_next_states()

    def set_states(self, item, short=False):
        """
        Get a key, e.g. for annotation.
        :param key: The key to set a item.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            self.memory.set('states', self.s, self.e, item)
        else:
            bl = len(self.tail_batch)
            self.memory.set('states', self.s, self.e, item[:-bl])
            self.tail_batch['states'] = item[-bl:]

    def get_states(self, short=False):
        """
        States of trajectory.
        :return: All states of the trajectory.
        :param short: Get fields of short trajectory.
        """
        if short or self.tail_batch is None:
            return self.memory.get('states', self.s, self.e)
        else:
            return np.concatenate(
                (
                    self.memory.get('states', self.s, self.e),
                    self.tail_batch.states
                ), axis=0
            )
