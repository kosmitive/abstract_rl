import threading
from os.path import exists, join
from typing import cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import NormalizeObservation
from numpy import ndarray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy


class MCMCEnvWrapper:
    """
    Environment wrapper for gymnasium environments. Adds support for executing a whole trajectory based on a policy,
    instead of only giving a step based interface to the outside.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
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
        if bool(conf.get_root("obs_norm")):
            self.env = self._norm = NormalizeObservation(self.env)
            path = join(self.mc.run_root, "model", "obs_rms.npz")
            if exists(path):
                stats = np.load(path)
                self._norm.obs_rms.mean = stats["mean"].copy()
                self._norm.obs_rms.var = stats["var"].copy()
                self._norm.obs_rms.count = float(stats["count"])
                self._norm.update_running_mean = False
        self._seed = conf.get_root("seed")

        self._normalize = conf.get("normalize", False)
        self._discount = conf["discount"]

        self.epoch = 0

        # set best measured reward to lowest possible reward
        self.best_reward: float = float(np.finfo(np.float64).min)
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
        # report the raw environment return). reward_clip, when set, clips that transformed reward to [-c, c].
        # num_envs above 1 collects with a gymnasium SyncVectorEnv; the default of one keeps the serial rollout.
        # membership, not conf.get: a missing key must not be written into a TRPO config
        self.num_envs = int(conf["num_envs"]) if "num_envs" in conf else 1
        self._reward_scale = float(conf.get("reward_scale", 1.0))
        self._reward_offset = float(conf.get("reward_offset", 0.0))
        clip = conf["reward_clip"] if "reward_clip" in conf else None  # noqa: SIM401
        self._reward_clip = None if clip is None else float(clip)
        self._vec: SyncVectorEnv | None = None
        self._open_obs: ndarray | None = None
        self._open_builders: list[TrajectoryBuilder] | None = None
        self._open_taken: ndarray | None = None
        self._open_return: ndarray | None = None

        self.state_filter = RunningCenteredFilter("states", self.observation_dim)
        self.reward_filter = RunningCenteredFilter("rewards", 1)

    def discount(self) -> float:
        """
        Discount factor of the environment.
        :return: The discount factor used.
        """
        return self._discount

    def _preprocess(self, obs: ndarray) -> ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return obs / self._obs_scale if self._normalize else obs

    def _shape_reward(self, reward: float) -> float:
        """Training reward: scale, shift, then optional symmetric clip."""

        shaped = self._reward_scale * float(reward) + self._reward_offset
        if self._reward_clip is not None:
            shaped = float(np.clip(shaped, -self._reward_clip, self._reward_clip))
        return shaped

    def reset(self) -> ndarray:
        """
        Resets the environment. The very first reset is seeded with the configured seed.
        :return: The state after the reset.
        """
        if self._seed is not None:
            cs, _ = self.env.reset(seed=int(self._seed))
            self.act_sp.seed(int(self._seed))
            self._seed = None
        else:
            cs, _ = self.env.reset()
        return self._preprocess(cs)

    def execute_policy(
        self,
        policy: ContinuousPolicy,
        max_steps: int,
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

        if self.num_envs > 1:
            trajectories = self._collect_vector(policy, max_steps, batch_size, exploration)
            t = sum(len(traj) for traj in trajectories)
        else:
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
        self,
        max_steps: int | None,
        policy: ContinuousPolicy,
        render: bool = False,
        opt: bool = False,
    ) -> Trajectory:
        """
        Execute a policy once for the maximum number of steps or until the episode terminates or is truncated.
        :param max_steps: The maximum number of steps.
        :param policy: The policy to use.
        :param render: Render the environment.
        :param opt: Use the mode of the policy instead of sampling.
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
        with torch.no_grad():
            while max_steps is None or t < max_steps:
                # sample distribution based on state
                tcs = torch.as_tensor(cs, dtype=torch.float32).view([1, -1])

                # sample action and calc log likelihood
                suff_stats = policy.forward(tcs)
                a = policy.mode(suff_stats=suff_stats) if opt else policy.sample_actions(suff_stats=suff_stats)
                ll_t = policy.log_prob(a, suff_stats=suff_stats)

                # prepare for usage
                ll = float(ll_t.reshape(-1)[0])
                a_np = a.reshape(-1).numpy().astype(np.float64)
                cs, r, terminated, truncated, _ = env.step(a_np)
                cs = self._preprocess(cs)
                self.state_filter.register(cs)
                raw_return += float(r)
                r = self._shape_reward(float(r))
                self.reward_filter.register(np.asarray([r], dtype=np.float64))

                t += 1

                # only true terminations are stored as done, so truncated episodes can be bootstrapped
                traj_builder.observe(a_np, float(r), cs, ll, int(terminated))

                # render if needed
                if render:
                    env.render()

                # break if necessary
                if terminated or truncated:
                    break

        # compile using the discount factor
        traj = traj_builder.finalize()
        traj.raw_return = raw_return
        self.max_traj_len = max(self.max_traj_len, t)
        return traj

    def close(self) -> None:
        self.env.close()
        if self._vec is not None:
            self._vec.close()

    def _make_sub_env(self) -> gym.Env:
        """One copy inside the vector env. Observation stats are Gymnasium's, shared with the serial wrapper."""

        env = gym.make(self.env_name)
        if self._norm is None:
            return env
        wrapped = NormalizeObservation(env)
        wrapped.obs_rms = self._norm.obs_rms
        wrapped.update_running_mean = self._norm.update_running_mean
        return wrapped

    def _vector_env(self) -> SyncVectorEnv:
        """Lazy vector env. Created once and left running so later calls continue open episodes.

        Gymnasium's vector ``NormalizeObservation`` requires next-step autoreset and rejects a partial reset, so it
        cannot keep one copy mid-episode while another copy ends. Each sub-environment is wrapped with the same
        scalar ``NormalizeObservation`` the serial rollout uses, and those wrappers share one running statistic.
        """

        if self._vec is None:
            self._vec = SyncVectorEnv(
                [self._make_sub_env for _ in range(self.num_envs)], autoreset_mode=AutoresetMode.DISABLED
            )
        return self._vec

    def _collect_vector(
        self, policy: ContinuousPolicy, max_steps: int, batch_size: int, exploration: bool
    ) -> list[Trajectory]:
        """Collect transitions from ``num_envs`` copies stepped together until ``batch_size`` is reached.

        The vector env is reset only on the first call. An episode that is still running is stored and continued
        the next time this method is called.
        """

        vec = self._vector_env()
        if (
            self._open_obs is None
            or self._open_builders is None
            or self._open_taken is None
            or self._open_return is None
        ):
            if self._seed is not None:
                obs, _ = vec.reset(seed=int(self._seed))
                self._seed = None
            else:
                obs, _ = vec.reset()
            obs = self._preprocess(np.asarray(obs))
            builders = [self._new_builder(row) for row in obs]
            taken = np.zeros(self.num_envs, dtype=np.int64)
            raw_return = np.zeros(self.num_envs, dtype=np.float64)
        else:
            obs = self._open_obs
            builders = self._open_builders
            taken = self._open_taken
            raw_return = self._open_return

        trajectories: list[Trajectory] = []
        total = 0
        with torch.no_grad():
            while total < batch_size:
                actions, log_prob = self._vector_actions(policy, obs, not exploration)
                nxt, rewards, terminated, truncated, _ = vec.step(actions)
                nxt = self._preprocess(np.asarray(nxt))
                ended = np.zeros(self.num_envs, dtype=bool)
                for i in range(self.num_envs):
                    raw_return[i] += float(rewards[i])
                    shaped = self._shape_reward(float(rewards[i]))
                    self.reward_filter.register(np.asarray([shaped], dtype=np.float64))
                    self.state_filter.register(nxt[i])
                    builders[i].observe(actions[i], shaped, nxt[i], float(log_prob[i]), int(terminated[i]))
                    taken[i] += 1
                    if terminated[i] or truncated[i] or taken[i] >= max_steps:
                        traj = builders[i].finalize()
                        traj.raw_return = float(raw_return[i])
                        trajectories.append(traj)
                        total += len(traj)
                        ended[i] = True
                        raw_return[i] = 0.0
                        taken[i] = 0
                if not np.any(ended):
                    obs = nxt
                    continue
                reset_obs, _ = vec.reset(options={"reset_mask": ended})
                fresh = self._preprocess(np.asarray(reset_obs)[ended])
                obs = nxt.copy()
                obs[ended] = fresh
                cursor = 0
                for i in range(self.num_envs):
                    if not ended[i]:
                        continue
                    builders[i] = self._new_builder(fresh[cursor])
                    cursor += 1
        self._open_obs = obs
        self._open_builders = builders
        self._open_taken = taken
        self._open_return = raw_return
        return trajectories

    def _new_builder(self, state: ndarray) -> TrajectoryBuilder:
        self.state_filter.register(state)
        with self.thread_lock:
            new_id = self.created_trajectories
            self.created_trajectories += 1
        return TrajectoryBuilder(new_id, self, state)

    def _vector_actions(self, policy: ContinuousPolicy, obs: ndarray, opt: bool) -> tuple[ndarray, ndarray]:
        """Actions and log probabilities for a batch of observations. Actions have shape ``[num_envs, action_dim]``."""

        states = torch.as_tensor(obs, dtype=torch.float32)
        stats = policy.forward(states)
        if opt:
            chosen = policy.mode(suff_stats=stats)
        else:
            sampled = policy.sample_actions(suff_stats=stats, num_actions=1)
            chosen = sampled if sampled.ndim == 2 else sampled[:, 0]
        log_prob = policy.log_prob(chosen, suff_stats=stats).reshape(-1)
        actions = chosen.reshape(states.shape[0], self.action_dim).detach().cpu().numpy().astype(np.float64)
        return actions, log_prob.detach().cpu().numpy()

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
        return int(np.prod(self.observation_space.shape))

    @property
    def action_dim(self) -> int:
        """
        Dimension of action space.
        :return: Dimension of the action space.
        """
        return int(np.prod(self.action_space.shape))
