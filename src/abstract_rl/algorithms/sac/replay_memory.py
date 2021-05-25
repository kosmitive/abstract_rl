import numpy as np
import torch
import random

from src.sac.rl_util import tt


class ReplayMemory:

    def __init__(self, env, size):

        # obtain state and action dimension
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # control variables
        self.size = size
        self.filled = 0
        self.i = 0

        # create space for the variables
        self.overall_dim = 2 * self.state_dim + self.action_dim + 2
        self.data = np.empty([size, self.overall_dim])

    def insert(self, state, action, reward, done, next_state):
        if np.isscalar(reward): reward = np.ones(1) * reward
        if np.isscalar(done): done = np.ones(1) * done

        conc_ent = np.concatenate((state, action, reward, done, next_state))
        self.data[self.i] = tt(conc_ent)
        self.filled = min(self.size, self.filled + 1)
        self.i = (self.i + 1) % self.size

    def sample(self, num):

        # get number of samples
        num = min(self.filled, num)
        inds = random.sample(range(self.filled), num)

        l = 0
        r = self.state_dim
        states = self.data[inds, l:r]

        l = r
        r += self.action_dim
        actions = self.data[inds, l:r]

        l = r
        r = l + 1
        rewards = self.data[inds, l:r]

        l = r
        r += 1
        dones = self.data[inds, l:r]

        l = r
        r += self.state_dim
        nxt_states = self.data[inds, l:r]

        return tt(states), tt(actions), tt(rewards), tt(dones), tt(nxt_states)
