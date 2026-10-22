import threading
from os.path import exists, join
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation

from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter

if TYPE_CHECKING:
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
    from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


class RolloutEnvWrapper:
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
        self.num_threads = conf.get("num_threads", 1)
        self.render = conf.get("render", False)

        # save config
        self.conf = conf
        self.num_epochs = conf.get_root("num_epochs")
        self.env_name = conf["name"]
        self.env = gym.make(self.env_name, render_mode="human" if self.render else None)
        # one env, so the running stats live on this wrapper; a saved run reloads them and freezes them
        self._norm: NormalizeObservation | None = None
        if conf.to_dict().get("obs_norm", False):
            self.env = self._norm = NormalizeObservation(self.env)
            path = join(self.mc.run_root, "model", "obs_rms.npz")
            if exists(path):
                self.load_obs_rms(path)
        self._seed = conf.get_root("seed")

        self._normalize = conf.get("normalize", False)
        self._discount = conf["discount"]

        self.epoch = 0

        # set best measured reward to lowest possible reward
        self.best_reward = float(np.finfo(np.float64).min)
        self.max_traj_len = 0
        self.cli = CliPrinter().instance
        self.created_trajectories = 0
        self.obs_sp = cast(Box, self.env.observation_space)
        self.act_sp = cast(Box, self.env.action_space)
        self.thread_lock = threading.Lock()

        # observations are scaled by the (finite) upper bound of the observation space if normalize is set
        high = np.asarray(self.obs_sp.high, dtype=np.float64)
        self._obs_scale = np.where(np.isfinite(high) & (high > 0), high, 1.0)

        # affine reward transform r' = reward_scale * r + reward_offset used for training only (evaluation scripts
        # report the raw environment return)
        self._reward_scale = conf.get("reward_scale", 1.0)
        self._reward_offset = conf.get("reward_offset", 0.0)

        self.state_filter = RunningCenteredFilter("states", self.observation_dim)
        self.reward_filter = RunningCenteredFilter("rewards", 1)

    def load_obs_rms(self, path: str) -> None:
        """Load stored observation statistics and freeze them."""
        assert self._norm is not None
        stats = np.load(path)
        self._norm.obs_rms.mean = stats["mean"].copy()
        self._norm.obs_rms.var = stats["var"].copy()
        self._norm.obs_rms.count = float(stats["count"])
        self._norm.update_running_mean = False

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
        Resets the environment. The very first reset is seeded with the configured seed.
        :return: The state after the reset.
        """
        if self._seed is not None:
            cs, _info = self.env.reset(seed=int(self._seed))
            self.act_sp.seed(int(self._seed))
            self._seed = None
        else:
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
        Executes the policy for complete episodes (each at most max_steps long) until at least batch_size
        transitions were collected.
        :param policy: The policy to evaluate.
        :param max_steps: The maximum number of steps per episode.
        :param batch_size: The minimum number of transitions to collect.
        :param exploration: Sample from the policy if True, otherwise use its mode.
        :param render: Render the first episode.
        :param rew_field_name: If given, log the mean and std of the episodic return under this name.
        :return: A trajectory collection.
        """

        t = 0
        k = 0
        trajectories = []
        while t < batch_size:
            tr = self.execute_policy_once(max_steps, policy, render and k == 0, opt=not exploration)
            trajectories.append(tr)
            t += len(tr)
            k += 1

        if rew_field_name is not None:
            # the logged return is the one of the environment, not of the (optionally scaled) training reward
            returns = [traj.raw_return for traj in trajectories]
            mean_return = float(np.mean(returns))
            self.best_reward = max(self.best_reward, mean_return)
            log = self.mc["log"]
            log.log({rew_field_name: [np.mean(returns), np.std(returns)]})
            if log.has_field("steps"):
                log.log({"steps": [t]})
            self.total_steps = getattr(self, "total_steps", 0) + t

        self.epoch += 1
        if self._norm is not None and self._norm.update_running_mean:
            rms = self._norm.obs_rms
            np.savez(
                join(self.mc.run_root, "model", "obs_rms.npz"),
                mean=rms.mean,
                var=rms.var,
                count=np.float64(rms.count),
            )

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

        # reset environment and create empty trajectory
        env = self.env
        cs = self.reset()
        self.state_filter.register(cs)

        # create new trajectory builder
        with self.thread_lock:
            new_id = self.created_trajectories
            self.created_trajectories += 1

        traj_builder = TrajectoryBuilder(new_id, self, cs)
        t = 0
        raw_return = 0.0

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
            a = a.detach().reshape(-1).numpy().astype(np.float64)
            cs, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated
            cs = self._preprocess(cs)
            self.state_filter.register(cs)
            raw_return += float(r)
            r = self._reward_scale * r + self._reward_offset
            self.reward_filter.register(np.asarray([r], dtype=np.float64))

            t += 1
            # only true terminations are stored as done, so truncated episodes can be bootstrapped
            traj_builder.observe(a, float(r), cs, ll, int(terminated))

            # render if needed
            if render:
                env.render()

            # break if necessary
            if done:
                break

        # compile using the discount factor
        traj = traj_builder.finalize()
        traj.raw_return = raw_return
        self.max_traj_len = max(self.max_traj_len, t)
        return traj

    def close(self) -> None:
        self.env.close()

    def frozen_eval_env(self, render_mode: str) -> gym.Env:
        """New env for greedy eval. Observation stats are the training ones and do not update."""
        env: gym.Env = gym.make(self.env_name, render_mode=render_mode)
        if self._norm is None:
            return env
        wrapped = NormalizeObservation(env)
        wrapped.obs_rms.mean = np.array(self._norm.obs_rms.mean, copy=True)
        wrapped.obs_rms.var = np.array(self._norm.obs_rms.var, copy=True)
        wrapped.obs_rms.count = self._norm.obs_rms.count
        wrapped.update_running_mean = False
        return wrapped

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
