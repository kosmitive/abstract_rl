import numpy as np
import pandas as pd

from abstract_rl.src.data_structures.temporal_difference_data.trajectory import Trajectory


class TrajectoryBuilder:
    """
    Represents a simple TrajectoryBuilder. It gets initialized with a starting state. Then
    the callee can add subsequently new observation tuples. Afterwards the trajectory is
    finalized and a instance of Trajectory is returned.
    """

    def __init__(self, id, env, s0):
        """
        This method initializes the TrajectoryBuilder using a starting state s0.
        :param s0: Simply the starting state.
        """

        # one field plus reset to initial state.
        self.id = id
        self._trajectory = None
        self.env = env
        self.reset(s0)

    def reset(self, s0):
        """
        This method resets the TrajectoryBuilder using a starting state s0.

        :param s0: Simply the starting state.
        """
        self._trajectory = [s0]

    def observe(self, act, rew, nxt_state, ll, done):
        """
        Register internally a (state, action, reward, likelihood) tuple.
        :param act: the taken action a_t.
        :param rew: the obtained reward r_t.
        :param nxt_state: the next state observed s_(t+1)
        :param ll: the likelihood p(a_t|s_t)
        """

        # extend the trajectory itself
        self._trajectory.extend([act, ll, rew, done, nxt_state])

    def finalize(self):
        """
        Finalize the trajectory, e.g. create pandas __data frames from it.
        :return The built trajectory.
        """

        # obtain number of steps
        num_steps = int(len(self._trajectory) / 5) + 1

        # obtain state and action dimension
        ad = len(self._trajectory[1]) if isinstance(self._trajectory[1], np.ndarray) else 1
        sd = len(self._trajectory[0]) if isinstance(self._trajectory[0], np.ndarray) else 1

        # reserve space for the desired elements.
        actions = np.empty([num_steps, ad])
        states = np.empty([num_steps, sd])
        rewards = np.empty([num_steps, 1])
        likelihood = np.empty([num_steps, 1])
        dones = np.empty([num_steps, 1])

        # obtain reward and q values
        l_actions = self._trajectory[1::5]
        l_states = self._trajectory[0::5]
        np_actions = np.stack(l_actions)
        np_states = np.stack(l_states)
        if np_states.ndim == 1: np_states = np.expand_dims(np_states, 1)
        if np_actions.ndim == 1: np_actions = np.expand_dims(np_actions, 1)

        # fill in the values
        states[:] = np_states
        actions[:-1] = np_actions
        rewards[:-1, 0] = self._trajectory[3::5]
        likelihood[:-1, 0] = self._trajectory[2::5]
        dones[:-1, 0] = self._trajectory[4::5]

        # get the __data frame
        data = np.concatenate((states, actions, rewards, likelihood, dones), 1)

        # assign the trajectory
        col_names = [f"s_{i}" for i in range(sd)]
        col_names.extend([f"a_{i}" for i in range(ad)])
        col_names.extend(["r", "ll", "d"])

        # build a trajectory using the specified __data frame.
        df = pd.DataFrame(data, columns=col_names)
        return Trajectory(self.id, self.env, df)
