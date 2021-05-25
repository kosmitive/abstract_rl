from quanser_robots import GentlyTerminating

import threading

import gym
import torch
import numpy as np

from abstract_rl.src.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.src.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.src.misc.cli_printer import CliPrinter
from abstract_rl.src.misc.running_centered_filter import RunningCenteredFilter


class MCMCEnvWrapper:
    """
    Environment wrapper for gym environments. Adds support for executing a whole trajectory based on a policy,
    instead of only giving a step based interface to the outside.
    """

    def namespace(self):
        return self._name_sp

    def __init__(self, mc):
        """
        Initialize a new environment.
        :param mc: The model configuration with everything important.
        """
        conf = mc['conf']
        self.mc = mc
        self.num_threads = conf['num_threads']
        self.render = conf['render']
        self._name_sp = conf.get_namespace()

        # save config
        self.conf = conf
        self.num_epochs = conf.get_root('num_epochs')
        self.env_name = conf['name']
        self.env = GentlyTerminating(gym.make(self.env_name))

        self._normalize = conf['normalize']
        self._discount = conf['discount']

        self.epoch = 0

        # set best measured reward to lowest possible reward
        self.best_reward = np.finfo(np.float64).min
        self.last_reward = None
        self.max_traj_len = 0
        self.min_reward = None
        self.cli = CliPrinter().instance
        self.created_trajectories = 0
        self.obs_sp = self.env.observation_space
        self.act_sp = self.env.action_space
        self.thread_lock = threading.Lock()

        self.state_filter = RunningCenteredFilter('states', self.observation_dim)
        self.reward_filter = RunningCenteredFilter('rewards', 1)

    def last_ucb_reward(self):
        assert self.last_reward is not None
        return self.last_reward

    def discount(self):
        """
        Discount factor of the environment.
        :return: The discount factor used.
        """
        return self._discount

    def reset(self):
        """
        Resets the environment.
        :return: The state after the reset.
        """
        cs = self.env.reset()
        return cs

    def execute_policy(self, policy, max_steps, batch_size, exploration=True, render=False, rew_field_name=None):
        """
        Executes a policy for a maximum number of steps multiple times. This work can be split onto multiple threads as
        well.
        :param policy: The policy to evaluate.
        :param max_steps: The maximum number of steps.
        :return: A list of trajectories.
        """

        with self.conf.ns('policy'):
            t = 0
            k = 0
            trajectories = []
            while t < batch_size:
                tr = self.execute_policy_once(np.minimum(batch_size - t, max_steps), policy, render and k == 0, opt=not exploration)
                trajectories.append(tr)
                t += len(tr)
                k += 1

        if rew_field_name is not None:
            disc_rewards = [traj.discounted_reward(self.discount()) for traj in trajectories]
            self.mc['log'].log({rew_field_name: [np.mean(disc_rewards), np.std(disc_rewards)]})

        self.epoch += 1

        tj = TrajectoryCollection(self, sum([len(tra) for tra in trajectories]))
        tj.extend(trajectories)
        tj.print()
        return tj

    def execute_policy_once(self, max_steps, policy, render=False, opt=False):
        """
        Execute a policy once for the maximum number of steps or the environment sends done.
        :param max_steps: The maximum number of steps, if done not set.
        :param policy: The policy to use.
        :param render: Render the environment.
        :param seed: Set a seed if wanted.
        :return: The finalized and built trajectory.
        """

        # reset environment and create empty trajectory
        env = GentlyTerminating(gym.make(self.env_name)) if not self.render else self.env
        cs = env.reset()
        if self._normalize: cs /= env.observation_space.high
        self.state_filter.register(cs)

        # create new trajectory builder
        with self.thread_lock:
            new_id = self.created_trajectories
            self.created_trajectories += 1

        traj_builder = TrajectoryBuilder(new_id, self, cs)
        t = 0

        # repeat for the number of steps
        while max_steps is None or t < max_steps:

            # sample distribution based on state
            tcs = torch.Tensor(cs)
            tcs = tcs.view([1, -1])

            # sample action and calc log likelihood
            suff_stats = policy.forward(tcs)
            a = policy.mode(suff_stats=suff_stats) \
                if opt else policy.sample_actions(suff_stats=suff_stats)

            ll = policy.log_prob(a, suff_stats=suff_stats)

            # prepare for usage
            ll = ll.detach().numpy()[0]
            a = a.detach().numpy()[0]
            cs, r, done, info = env.step(a)
            self.state_filter.register(cs)
            self.reward_filter.register(cs)

            # bug fix for quanser
            cs /= env.observation_space.high

            t += 1
            # only punish if episode aborted
            traj_builder.observe(a, r, cs, ll, int(done))

            # render if needed
            if render: env.render()

            # break if necessary
            if done: break

        # compile using the discount factor
        traj = traj_builder.finalize()
        self.max_traj_len = max(self.max_traj_len, t)
        env.close()
        return traj

    @property
    def observation_space(self):
        """
        Bounds for the observation space.
        :return: Bound of the observation space.
        """

        return self.obs_sp

    @property
    def action_space(self):
        """
        Bounds for the action space.
        :return: Bound of the action space.
        """
        return self.act_sp

    @property
    def observation_dim(self):
        """
        Dimension of observation space.
        :return: Dimension of the observation space.
        """
        return int(np.prod(self.observation_space.high.shape))

    @property
    def action_dim(self):
        """
        Dimension of action space.
        :return: Dimension of the action space.
        """
        return int(np.prod(self.action_space.high.shape))
