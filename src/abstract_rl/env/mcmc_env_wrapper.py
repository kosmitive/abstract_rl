"""Gymnasium wrapper that rolls out whole episodes of a policy."""

import math
import os
import threading
from typing import Any, cast

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from numpy import ndarray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.jax_classic import JAX_TRAIN_ENVS
from abstract_rl.env.jax_rollout import collect_classic, trajectories_from_buffer
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy


def collection_request(done_steps: int, epoch_steps: int, max_env_steps: int | None) -> int | None:
    """Transitions to ask for this epoch.

    ``None`` means ``done_steps`` is already at ``max_env_steps`` and the run should stop. The last
    epoch asks only for the steps still inside the budget. A missing budget keeps the full epoch.

    Args:
        done_steps: Environment steps already stored.
        epoch_steps: Transitions one epoch asks for when the budget allows it.
        max_env_steps: Hard cap. ``None`` or ``0`` leaves the epoch unchanged.

    Returns:
        The request for this epoch, or ``None`` when the budget is spent.
    """
    if not max_env_steps:
        return epoch_steps
    if done_steps >= max_env_steps:
        return None
    return min(epoch_steps, max_env_steps - done_steps)


def _make_env(name: str) -> Any:
    """Build one environment. The default argument avoids a late-binding closure."""

    def factory(env_name: str = name) -> gym.Env:
        return gym.make(env_name)

    return factory


class MCMCEnvWrapper:
    """Runs complete episodes of a policy and returns a trajectory collection.

    One environment is used when the step budget is small, so seeding and the smoke tests stay
    on the original path. Larger budgets use a synchronous vector env. ``env_vector_envs`` (read
    while the ``env`` namespace is active) overrides the automatic choice; ``0`` means automatic.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        """Create the environment from the model configuration.

        Args:
            mc: Model configuration. The active namespace is ``env``.
        """
        conf = mc["conf"]
        self.mc = mc
        self.num_threads = conf.get("num_threads", 1)
        self.render = conf.get("render", False)
        self._vector_envs = int(conf.get("vector_envs", 0))

        self.conf = conf
        self.num_epochs = conf.get_root("num_epochs")
        self.env_name = conf["name"]
        self.env = gym.make(self.env_name, render_mode="human" if self.render else None)
        self._seed = conf.get_root("seed")
        self._vec: SyncVectorEnv | None = None

        self._normalize = conf.get("normalize", False)
        self._discount = conf["discount"]

        self.epoch = 0
        self.best_reward: float = float(np.finfo(np.float64).min)
        self.max_traj_len = 0
        self.cli = CliPrinter().instance
        self.created_trajectories = 0
        self.obs_sp = cast(Box, self.env.observation_space)
        self.act_sp = cast(Box, self.env.action_space)
        self.thread_lock = threading.Lock()
        self.total_steps = 0

        high = np.asarray(self.obs_sp.high, dtype=np.float64)
        self._obs_scale = np.where(np.isfinite(high) & (high > 0), high, 1.0)
        self._reward_scale = float(conf.get("reward_scale", 1.0))
        self._reward_offset = float(conf.get("reward_offset", 0.0))

        self.reward_filter = RunningCenteredFilter("rewards", 1)

    def discount(self) -> float:
        """Discount factor of the environment."""
        return self._discount

    def _preprocess(self, obs: ndarray) -> ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return obs / self._obs_scale if self._normalize else obs

    def reset(self) -> ndarray:
        """Reset the single environment. The first reset uses the configured seed."""
        if self._seed is not None:
            cs, _info = self.env.reset(seed=int(self._seed))
            self.act_sp.seed(int(self._seed))
            self._seed = None
        else:
            cs, _info = self.env.reset()
        return self._preprocess(cs)

    def _resolve_n_envs(self, batch_size: int, max_steps: int | None, render: bool) -> int:
        """How many environments to step for this collection."""
        if render:
            return 1
        if self._vector_envs > 0:
            n_envs = self._vector_envs
        elif batch_size < 800:
            return 1
        else:
            n_envs = min(8, os.cpu_count() or 1)
        if max_steps is not None and max_steps > 0:
            n_envs = min(n_envs, max(1, math.ceil(batch_size / max_steps)))
        return max(1, n_envs)

    def _ensure_vector(self, n_envs: int) -> SyncVectorEnv:
        if self._vec is not None and self._vec.num_envs == n_envs:
            return self._vec
        if self._vec is not None:
            self._vec.close()
        self._vec = SyncVectorEnv(
            [_make_env(self.env_name) for _ in range(n_envs)],
            autoreset_mode=AutoresetMode.DISABLED,
        )
        return self._vec

    def _new_builder(self, state: ndarray) -> TrajectoryBuilder:
        with self.thread_lock:
            new_id = self.created_trajectories
            self.created_trajectories += 1
        return TrajectoryBuilder(new_id, self, state)

    def execute_policy(
        self,
        policy: ContinuousPolicy,
        max_steps: int,
        batch_size: int,
        exploration: bool = True,
        render: bool = False,
        rew_field_name: str | None = None,
    ) -> TrajectoryCollection:
        """Collect episodes until exactly ``batch_size`` transitions are stored.

        Args:
            policy: Policy to roll out.
            max_steps: Maximum length of one episode.
            batch_size: Number of transitions to store. An episode that would run past this count
                is cut the way ``max_steps`` cuts it, and that cut is not a done.
            exploration: Sample actions when true, otherwise use the mode.
            render: Render the first episode. Forces a single environment.
            rew_field_name: Log the mean and std of the raw episodic return under this name.

        Returns:
            The collected trajectories.
        """
        # Pendulum and MountainCarContinuous step in JAX. MuJoCo and rendering stay on Gymnasium.
        if self.env_name in JAX_TRAIN_ENVS and not render:
            trajectories = self._execute_jax(policy, max_steps, batch_size, exploration)
        else:
            n_envs = self._resolve_n_envs(batch_size, max_steps, render)
            if n_envs == 1:
                trajectories = self._execute_serial(policy, max_steps, batch_size, exploration, render)
            else:
                trajectories = self._execute_vector(policy, max_steps, batch_size, exploration, n_envs)

        collected = sum(len(traj) for traj in trajectories)
        if rew_field_name is not None:
            returns = [traj.raw_return for traj in trajectories]
            mean_return = float(np.mean(returns))
            self.best_reward = max(self.best_reward, mean_return)
            log = self.mc["log"]
            log.log({rew_field_name: [float(np.mean(returns)), float(np.std(returns))]})
            if log.has_field("steps"):
                log.log({"steps": [collected]})
            self.total_steps += collected

        self.epoch += 1
        collection = TrajectoryCollection(self, collected)
        collection.extend(trajectories)
        collection.print()
        return collection

    def _execute_serial(
        self,
        policy: ContinuousPolicy,
        max_steps: int | None,
        batch_size: int,
        exploration: bool,
        render: bool,
    ) -> list[Trajectory]:
        collected = 0
        trajectories: list[Trajectory] = []
        while collected < batch_size:
            room = batch_size - collected
            cap = room if max_steps is None else min(int(max_steps), room)
            trajectory = self.execute_policy_once(cap, policy, render and collected == 0, opt=not exploration)
            trajectories.append(trajectory)
            collected += len(trajectory)
        return trajectories

    def execute_policy_once(
        self,
        max_steps: int | None,
        policy: ContinuousPolicy,
        render: bool = False,
        opt: bool = False,
    ) -> Trajectory:
        """Roll out one episode on the single environment.

        Args:
            max_steps: Stop after this many steps. ``None`` runs until the environment ends.
            policy: Policy to roll out.
            render: Render each step.
            opt: Use the mode of the policy instead of sampling.

        Returns:
            The finished trajectory. Time-limit truncations are not marked done.
        """
        env = self.env
        state = self.reset()
        builder = self._new_builder(state)
        steps = 0
        raw_return = 0.0

        while max_steps is None or steps < max_steps:
            action, log_density = policy.act(np.asarray(state, dtype=np.float64).reshape(1, -1), deterministic=opt)
            action_np = np.asarray(action, dtype=np.float64).reshape(-1)
            log_prob = float(np.asarray(log_density).reshape(-1)[0])
            state, reward, terminated, truncated, _info = env.step(action_np)
            state = self._preprocess(state)
            raw_return += float(reward)
            shaped = self._reward_scale * float(reward) + self._reward_offset
            self.reward_filter.register(np.asarray([shaped], dtype=np.float64))
            steps += 1
            builder.observe(action_np, shaped, state, log_prob, int(terminated))
            if render:
                env.render()
            if terminated or truncated:
                break

        trajectory = builder.finalize()
        trajectory.raw_return = raw_return
        self.max_traj_len = max(self.max_traj_len, steps)
        return trajectory

    def _execute_jax(
        self,
        policy: ContinuousPolicy,
        max_steps: int,
        batch_size: int,
        exploration: bool,
    ) -> list[Trajectory]:
        """Jitted classic-control collection. The returned trajectories match the Gymnasium path."""
        n_envs = self._resolve_n_envs(batch_size, max_steps, render=False)
        obs_norm = policy.observation_norm()
        obs_mean, obs_inv_std, obs_clip, obs_ready = obs_norm.rollout_params()
        buf = collect_classic(
            policy.params,
            self.mc.rng.split(),
            jnp.asarray(self._obs_scale, dtype=jnp.float32),
            jnp.float32(self._reward_scale),
            jnp.float32(self._reward_offset),
            self.env_name,
            int(max_steps),
            int(batch_size),
            int(n_envs),
            not exploration,
            bool(self._normalize),
            obs_norm.enabled,
            obs_mean,
            obs_inv_std,
            obs_clip,
            obs_ready,
        )
        trajectories = trajectories_from_buffer(self, buf)
        for trajectory in trajectories:
            with self.thread_lock:
                trajectory.traj_id = self.created_trajectories
                self.created_trajectories += 1
            self.reward_filter.register(trajectory.rewards)
            self.max_traj_len = max(self.max_traj_len, len(trajectory))
        return trajectories

    def _execute_vector(
        self,
        policy: ContinuousPolicy,
        max_steps: int | None,
        batch_size: int,
        exploration: bool,
        n_envs: int,
    ) -> list[Trajectory]:
        """Step ``n_envs`` environments and store exactly ``batch_size`` transitions.

        An episode that would run past the budget is cut the way ``max_steps`` cuts it. Steps of a
        vector tick that would pass ``batch_size`` are not recorded.
        """
        vec = self._ensure_vector(n_envs)
        if self._seed is not None:
            obs, _info = vec.reset(seed=[int(self._seed) + i for i in range(n_envs)])
            self.act_sp.seed(int(self._seed))
            self._seed = None
        else:
            obs, _info = vec.reset()
        obs = self._preprocess(obs)
        recording = np.ones(n_envs, dtype=bool)
        steps = np.zeros(n_envs, dtype=np.int32)
        raw = np.zeros(n_envs, dtype=np.float64)
        builders = [self._new_builder(obs[i]) for i in range(n_envs)]

        trajectories: list[Trajectory] = []
        collected = 0

        def close_episode(index: int) -> None:
            trajectory = builders[index].finalize()
            trajectory.raw_return = float(raw[index])
            self.max_traj_len = max(self.max_traj_len, int(steps[index]))
            trajectories.append(trajectory)
            raw[index] = 0.0
            steps[index] = 0

        while np.any(recording) and collected < batch_size:
            actions, log_density = policy.act(obs, deterministic=not exploration)
            next_obs, rewards, terminated, truncated, _info = vec.step(np.asarray(actions, dtype=np.float64))
            next_obs = self._preprocess(next_obs)
            reset_mask = np.zeros(n_envs, dtype=bool)
            for i in range(n_envs):
                if not recording[i]:
                    if bool(terminated[i] or truncated[i]):
                        reset_mask[i] = True
                    continue
                if collected >= batch_size:
                    if int(steps[i]) > 0:
                        close_episode(i)
                    recording[i] = False
                    reset_mask[i] = True
                    continue
                reward = float(rewards[i])
                raw[i] += reward
                shaped = self._reward_scale * reward + self._reward_offset
                self.reward_filter.register(np.asarray([shaped], dtype=np.float64))
                steps[i] += 1
                action_np = np.asarray(actions[i], dtype=np.float64).reshape(-1)
                log_prob = float(np.asarray(log_density[i]))
                builders[i].observe(action_np, shaped, next_obs[i], log_prob, int(bool(terminated[i])))
                collected += 1
                hit_limit = max_steps is not None and int(steps[i]) >= max_steps
                if bool(terminated[i] or truncated[i] or hit_limit or collected >= batch_size):
                    close_episode(i)
                    reset_mask[i] = True
                    recording[i] = collected < batch_size
            if collected >= batch_size or not np.any(recording):
                for i in range(n_envs):
                    if int(steps[i]) > 0:
                        close_episode(i)
                        recording[i] = False
                break
            obs = next_obs
            if np.any(reset_mask):
                reset_obs, _info = vec.reset(options={"reset_mask": reset_mask})
                reset_obs = self._preprocess(reset_obs)
                for i in range(n_envs):
                    if not reset_mask[i]:
                        continue
                    obs[i] = reset_obs[i]
                    if recording[i]:
                        builders[i] = self._new_builder(obs[i])
                        raw[i] = 0.0
                        steps[i] = 0
        return trajectories

    def close(self) -> None:
        """Close the single environment and the vector env, if one was created."""
        self.env.close()
        if self._vec is not None:
            self._vec.close()
            self._vec = None

    @property
    def observation_space(self) -> Box:
        """Bounds of the observation space."""
        return self.obs_sp

    @property
    def action_space(self) -> Box:
        """Bounds of the action space."""
        return self.act_sp

    @property
    def observation_dim(self) -> int:
        """Dimension of the observation space."""
        return int(np.prod(self.observation_space.shape))

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return int(np.prod(self.action_space.shape))
