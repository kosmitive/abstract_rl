import shutil
from os import mkdir
from os.path import exists, join, basename

import numpy as np

from abstract_rl.src.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.src.data_structures.memory.trajectory_wrapper import TrajectoryWrapper


class CircularTailMemory:
    """
    Represents a circular tail memory. It is a mixture of a trajectory based memory and a circular memory. So each
    trajectory has a tail, whereas the length of the tail defines also the minimum length of a trajectory. Sampling and
    converting to a batch can be made very efficiently. Additionally a wrapper class is available to access the data
    in a trajectory based fashion, e.g. for a evaluation operator.
    """

    def __init__(self, mc):
        """
        Initializes a new circular tail memory.
        :param mc: The model configuration containing all important elements like env and so on.
        """

        # save the internal variables savely
        conf = mc['conf']
        env = mc['env']
        size = conf['size']
        self.min_l = conf['bootstrap_steps']

        # obtain sizes
        observation_dim = env.observation_dim
        action_dim = env.action_dim
        act_type = np.float64

        # setup all necessary fields
        self._fields = {
            'log_likelihoods': np.zeros([size, 1]),
            'states': np.empty([size, observation_dim]),
            'next_states': np.empty([size, observation_dim]),
            'actions': np.empty([size, action_dim], dtype=act_type),
            'rewards': np.empty([size, 1])
        }

        # maintain a list of added fields
        self._added_fields = set()
        self._dims = {}

        # some stuff for bookkeeping
        self.i = 0
        self.real_size = 0
        self.overwrite_traj = 0
        self.size = size
        self.trajectories = []
        self.last_inserted = -1

    def clear(self):

        # some stuff for bookkeeping
        self.i = 0
        self.real_size = 0
        self.overwrite_traj = 0
        self.trajectories = []
        self.last_inserted = -1

    def __len__(self): return self.real_size

    def add_trajectory(self, trajectory):
        """
        Adds a single trajectory to the memory.
        :param trajectory: A trajectory recorded by the abstract environment.
        """

        # check if length of trajectory fits in
        assert len(trajectory) <= self.size
        if self.min_l > len(trajectory): return # skip too short trajectories

        # get pointer to length and update real size
        traj_len = len(trajectory) - self.min_l

        # setup first window
        f_window_start = self.i
        f_window_end = f_window_start + traj_len

        # obtain index
        overflow = f_window_end > self.size
        if overflow: f_window_end -= self.size

        # obtain the states actions and rewards
        states = trajectory.states
        next_states = trajectory.next_states
        actions = trajectory.actions
        rewards = trajectory.rewards
        ll = trajectory.log_likelihoods

        # set it internally
        self.set('states', f_window_start, f_window_end, states[:-self.min_l])
        self.set('next_states', f_window_start, f_window_end, next_states[:-self.min_l])
        self.set('actions', f_window_start, f_window_end, actions[:-self.min_l])
        self.set('rewards', f_window_start, f_window_end, rewards[:-self.min_l])
        self.set('log_likelihoods', f_window_start, f_window_end, ll[:-self.min_l])

        # iterate over the trajectories
        ml = len(self.trajectories)
        k = 0
        while k < ml:
            wrapper = self.trajectories[k]
            if not overflow:
                if f_window_end == wrapper.e:
                    k += 1
                    break
                if wrapper.s < f_window_end < wrapper.e:
                    wrapper.s = f_window_end
                    break

            k += 1

        # cut off trajectories at beginning and append new wrapper
        self.trajectories = self.trajectories[:k]
        batch_add_data = {}
        for f in self._added_fields:
            batch_add_data[f] = np.empty([self.min_l, self._dims[f]])

        self.trajectories.append(
            TrajectoryWrapper(
                self, f_window_start, f_window_end,
                Batch(
                    states[-self.min_l:], actions[-self.min_l:], ll[-self.min_l:],
                    rewards[-self.min_l:], next_states[-self.min_l:], batch_add_data
                )
            ) if self.min_l > 0 else None
        )

        # update i
        self.real_size = np.minimum(self.size, self.real_size + traj_len)
        self.i = (self.i + traj_len) % self.size

    def store(self, mem):
        if exists(mem): shutil.rmtree(mem)
        mkdir(mem)
        for k in range(len(self.trajectories)):
            traj_folder = join(mem, str(k))
            mkdir(traj_folder)
            np.savetxt(join(traj_folder, 'states.txt'), self.trajectories[k].states)
            np.savetxt(join(traj_folder, 'actions.txt'), self.trajectories[k].actions)
            np.savetxt(join(traj_folder, 'rewards.txt'), self.trajectories[k].rewards)
            np.savetxt(join(traj_folder, 'log_likelihoods.txt'), self.trajectories[k].log_likelihoods)
            np.savetxt(join(traj_folder, 'next_states.txt'), self.trajectories[k].next_states)

    # def load(self, mem):
    #     subfolders = [f.path for f in scandir(mem) if f.is_dir()]
    #
    #     i = 0
    #     for traj_folder in subfolders:
    #         k = int(basename(traj_folder))
    #
    #         states = np.loadtxt(join(traj_folder, 'states.txt'))
    #         actions = np.expand_dims(np.loadtxt(join(traj_folder, 'actions.txt')), 1)
    #         rewards = np.expand_dims(np.loadtxt(join(traj_folder, 'rewards.txt')), 1)
    #         ll = np.expand_dims(np.loadtxt(join(traj_folder, 'log_likelihoods.txt')), 1)
    #         next_states = np.loadtxt(join(traj_folder, 'next_states.txt'))
    #
    #         ni = i + len(states)
    #         self.trajectories = self.trajectories[:k]
    #         batch_add_data = {}
    #         for f in self._added_fields:
    #             batch_add_data[f] = np.empty([self.min_l, self._dims[f]])
    #         self.trajectories.append(TrajectoryWrapper(
    #             self, i, ni-self.min_l,
    #             Batch(
    #                 states[-self.min_l:], actions[-self.min_l:], ll[-self.min_l:],
    #                 rewards[-self.min_l:], next_states[-self.min_l:], batch_add_data
    #             )
    #         ))
    #         self.trajectories[-1].set_states(states[:-self.min_l], short=True)
    #         self.trajectories[-1].set_actions(actions[:-self.min_l], short=True)
    #         self.trajectories[-1].set_rewards(rewards[:-self.min_l], short=True)
    #         self.trajectories[-1].set_log_likelihoods(ll[:-self.min_l], short=True)
    #         self.trajectories[-1].set_next_states(next_states[:-self.min_l], short=True)
    #         i = ni

    def add_trajectories(self, trajectories):
        """
        Add a list of trajectories to the memory.
        :param trajectories: A list of trajectories.
        :return: mean cumulative reward +_standard deviation cumulative reward
        """
        for traj in trajectories: self.add_trajectory(traj)

    def start_traj(self):
        self.trajectories.append(TrajectoryWrapper(self, self.i, self.i, None))

    def stop_traj(self):
        self.trajectories[-1].e = self.i

    def add(self, cs, a, r, ns, adds=None):

        f_window_start = self.i
        f_window_end = self.i + 1
        self.set('states', f_window_start, f_window_end, cs)
        self.set('actions', f_window_start, f_window_end, a)
        self.set('rewards', f_window_start, f_window_end, r)
        self.set('next_states', f_window_start, f_window_end, ns)

        if adds is not None:
            for k, v in adds.items():
                self.set(k, f_window_start, f_window_end, v)

        # update i
        self.real_size = np.minimum(self.size, self.real_size + 1)
        self.i = (self.i + 1) % self.size

    def get(self, name, s, e):
        """
        Get the data specified at the positions of the wrapper. Wraps are handled automatically.
        :param name: The name of the data.
        :param s: The start of the data.
        :param e: The end of the data.
        :return: The extracted data.
        """

        if s < e: return self._fields[name][s:e]
        else:
            return np.concatenate(
                (
                    self._fields[name][s:],
                    self._fields[name][:e]
                ), axis=0)

    def set(self, name, s, e, vals):
        """
        Set the data specified at the position of the wrapper with the data supplied. Wraps are handled automatically.
        :param name: The name of the data.
        :param s: The start of the data.
        :param e: The end of the data.
        :param vals: The values to add to the memory.
        :return: The extracted data.
        """

        if name not in self._fields:
            d = vals.shape[1]
            self._fields[name] = np.empty([self.size, d])

        if s + 1 == e: self._fields[name][s] = vals
        elif s < e: self._fields[name][s:e] = vals
        else:
            self._fields[name][s:] = vals[:-e]
            self._fields[name][:e] = vals[-e:]

    def apply_operator(self, operator):
        """
        Apply a evaluation operator to each trajectory in the memory.
        :param operator: The operator to apply, e.g. retrace or gae.
        """
        # apply to each trajectory
        for traj in self.trajectories: operator.transform(traj)

    def apply_batch_operator(self, operator):
        """
        Apply a evaluation operator to each trajectory in the memory.
        :param operator: The operator to apply, e.g. retrace or gae.
        """
        # apply to each trajectory
        batch = self.to_batch()
        operator.transform(batch)
        self._fields['add_act'] = batch['add_act']

    def apply_operators(self, operators):
        """
        Applies a list of evaluation operators to each trajectory. Simply apply each operator
        independently.
        :param operators: A list of operators to apply in the order supplied.
        """
        [self.apply_operator(op) for op in operators]

    def register_fields(self, name, dim=1):
        """
        Use this to register a new field, e.g. space is aquired and initialized randomly.
        :param name: The name of the new field.
        :param dim: The dimension, (default: 1)
        """
        if name in self._added_fields: return
        self._added_fields.add(name)
        self._fields[name] = np.empty([self.size, dim])
        self._dims[name] = dim
        for traj in self.trajectories:
            traj.init_field(name, dim)

    def sample(self, batch_size):
        """
        Sample a batch of data from the memory. Note the minimum from the real size and the batch size is used.
        :param batch_size: The number of samples which should be sampled.
        :return: A batch with the sampled data.
        """

        # get some space
        states = self._fields['states']
        actions = self._fields['actions']
        rewards = self._fields['rewards']
        nxt_states = self._fields['next_states']
        log_likelihoods = self._fields['log_likelihoods']

        # obtain the wanted trajectory
        batch_size = np.minimum(self.real_size, batch_size)

        # define the batch indices
        batch_indices = np.random.permutation(self.real_size)[:batch_size]

        # reserve space for the actions
        states = states[batch_indices]
        actions = actions[batch_indices]
        rewards = rewards[batch_indices]
        nxt_states = nxt_states[batch_indices]
        log_likelihoods = log_likelihoods[batch_indices]

        add_fields = {
            k: self._fields[k][batch_indices]
            for k in self._added_fields
        }

        return Batch(states, actions, log_likelihoods, rewards, nxt_states, add_fields)

    def to_batch(self):
        """
        Converts the complete memory into a batch. This operation usually should interface the data directly instead
        of doing costly copying.
        :return: A batch containing all valid data from the memory.
        """

        states = self._fields['states'][:self.real_size]
        nxt_states = self._fields['next_states'][:self.real_size]
        actions = self._fields['actions'][:self.real_size]
        log_likelihoods = self._fields['log_likelihoods'][:self.real_size]
        rewards = self._fields['rewards'][:self.real_size]
        add_data = {k: self._fields[k][:self.real_size] for k in self._added_fields}
        return Batch(states, actions, log_likelihoods, rewards, nxt_states, add_data)
