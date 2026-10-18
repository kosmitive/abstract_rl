import threading
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter

if TYPE_CHECKING:
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
    from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


class MCMCEnvWrapper:
    """
    Environment wrapper for gym environments. Adds support for executing a whole trajectory based on a policy,
    instead of only giving a step based interface to the outside.
    """

    def __init__(self, mc: "ModelConfiguration") -> None:
        """
        Initialize a new environment.
        :param mc: The model configuration with everything important.
        """
        conf = mc["conf"]
        self.mc = mc
        self.num_threads = conf["num_threads"]
        self.render = conf["render"]

        # save config
        self.conf = conf
        self.num_epochs = conf.get_root("num_epochs")
        self.env_name = conf["name"]
        self.env = gym.make(self.env_name, render_mode="human" if self.render else None)

        self._normalize = conf["normalize"]
        self._discount = conf["discount"]

        self.epoch = 0

        # set best measured reward to lowest possible reward
        self.best_reward = np.finfo(np.float64).min
        self.max_traj_len = 0
        self.cli = CliPrinter().instance
        self.created_trajectories = 0
        self.obs_sp = cast(Box, self.env.observation_space)
        self.act_sp = cast(Box, self.env.action_space)
        self.thread_lock = threading.Lock()

        # observations are scaled by the (finite) upper bound of the observation space if normalize is set
        high = np.asarray(self.obs_sp.high, dtype=np.float64)
        self._obs_scale = np.where(np.isfinite(high) & (high > 0), high, 1.0)

        self.state_filter = RunningCenteredFilter("states", self.observation_dim)
        self.reward_filter = RunningCenteredFilter("rewards", 1)

    def discount(self) -> float:
        """
        Discount factor of the environment.
        :return: The discount factor used.
        """
        return self._discount

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return obs / self._obs_scale if self._normalize else obs

    def reset(self) -> np.ndarray:
        """
        Resets the environment.
        :return: The state after the reset.
        """
        cs, _info = self.env.reset()
        return self._preprocess(cs)

    def execute_policy(
        self,
        policy: Any,
        max_steps: int | None,
        batch_size: int,
        exploration: bool = True,
        render: bool = False,
        rew_field_name: str | None = None,
    ) -> TrajectoryCollection:
        """
        Executes a policy for a maximum number of steps multiple times. This work can be split onto multiple threads as
        well.
        :param policy: The policy to evaluate.
        :param max_steps: The maximum number of steps.
        :return: A list of trajectories.
        """

        with self.conf.ns("policy"):
            t = 0
            k = 0
            trajectories = []
            while t < batch_size:
                assert max_steps is not None
                tr = self.execute_policy_once(
                    np.minimum(batch_size - t, max_steps), policy, render and k == 0, opt=not exploration
                )
                trajectories.append(tr)
                t += len(tr)
                k += 1

        if rew_field_name is not None:
            disc_rewards = [traj.discounted_reward(self.discount()) for traj in trajectories]
            self.mc["log"].log({rew_field_name: [np.mean(disc_rewards), np.std(disc_rewards)]})

        self.epoch += 1

        tj = TrajectoryCollection(self, sum([len(tra) for tra in trajectories]))
        tj.extend(trajectories)
        tj.print()
        return tj

    def execute_policy_once(
        self, max_steps: int | None, policy: Any, render: bool = False, opt: bool = False
    ) -> "Trajectory":
        """
        Execute a policy once for the maximum number of steps or the environment sends done.
        :param max_steps: The maximum number of steps, if done not set.
        :param policy: The policy to use.
        :param render: Render the environment.
        :param seed: Set a seed if wanted.
        :return: The finalized and built trajectory.
        """

        # one environment for the run, so its RNG advances across episodes
        env = self.env
        cs, _info = env.reset()
        cs = self._preprocess(cs)
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
            a = policy.mode(suff_stats=suff_stats) if opt else policy.sample_actions(suff_stats=suff_stats)

            ll = policy.log_prob(a, suff_stats=suff_stats)

            # prepare for usage
            ll = float(ll.detach().reshape(-1)[0])
            a = a.detach().numpy()[0]
            cs, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated
            cs = self._preprocess(cs)
            self.state_filter.register(cs)
            self.reward_filter.register(np.asarray([r], dtype=np.float64))

            t += 1
            # only punish if episode aborted
            traj_builder.observe(a, float(r), cs, ll, int(done))

            # render if needed
            if render:
                env.render()

            # break if necessary
            if done:
                break

        # compile using the discount factor
        traj = traj_builder.finalize()
        self.max_traj_len = max(self.max_traj_len, t)
        return traj

    def close(self) -> None:
        self.env.close()

    @property
    def observation_space(self) -> Box:
        """
        Bounds for the observation space.
        :return: Bound of the observation space.
        """

        return self.obs_sp

    @property
    def action_space(self) -> Box:
        """
        Bounds for the action space.
        :return: Bound of the action space.
        """
        return self.act_sp

    @property
    def observation_dim(self) -> int:
        """
        Dimension of observation space.
        :return: Dimension of the observation space.
        """
        return int(np.prod(self.observation_space.high.shape))

    @property
    def action_dim(self) -> int:
        """
        Dimension of action space.
        :return: Dimension of the action space.
        """
        return int(np.prod(self.action_space.high.shape))
