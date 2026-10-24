"""Opt-in rollout. The Gymnasium wrapper is untouched; this is the other backend.

``execute_policy`` has the same job as ``RolloutEnvWrapper.execute_policy``:
episodes of at most ``max_steps``, at least ``batch_size`` transitions, the
training reward scaled, the logged return unscaled, and ``done`` set only on a
real termination. The steps run in JAX, sharded over ``env_jax_devices`` host
devices, and the trajectories are slices of the MLX arena.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from gymnasium.spaces import Box

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.interop.envs import make_jax_env
from abstract_rl.interop.scan import sharded_rollout
from abstract_rl.misc.cli_printer import CliPrinter


def _key(seed: int) -> Any:
    """A JAX key. ``jax.random.key`` is current; ``PRNGKey`` remains on older JAX."""
    import jax

    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    return jax.random.PRNGKey(seed)


class JaxRolloutEnv:
    """Vectorized JAX rollout that hands MLX a trajectory arena."""

    def __init__(self, mc: Any) -> None:
        """Build the JAX environment named in the current ``env`` namespace.

        Args:
            mc: The model configuration. ``rollout_backend`` is already ``jax``.

        Raises:
            RuntimeError: When observation normalization, reward or state filters,
                or an unsupported environment was requested. Those paths stay on Gymnasium.
        """
        conf = mc["conf"]
        raw = conf.to_dict()
        if raw.get("obs_norm", False):
            raise RuntimeError("obs_norm is not supported with rollout_backend=jax; use the Gymnasium backend")
        if raw.get("filter_states", False) or raw.get("filter_rewards", False):
            raise RuntimeError("filter_states and filter_rewards are not supported with rollout_backend=jax")
        self.mc = mc
        self.conf = conf
        self.env_name = conf["name"]
        self.render = bool(conf.get("render", False))
        if self.render:
            raise RuntimeError("render is not supported with rollout_backend=jax")
        dtype = str(conf.get("jax_dtype", "float64"))
        self.num_envs = int(conf.get("num_envs", 32))
        self.n_devices = int(conf.get("jax_devices", 1))
        self._env = make_jax_env(self.env_name, dtype)
        self._normalize = bool(conf.get("normalize", False))
        self._discount = float(conf["discount"])
        self._reward_scale = float(conf.get("reward_scale", 1.0))
        self._reward_offset = float(conf.get("reward_offset", 0.0))
        high = np.asarray(self._env.obs_high, dtype=np.float64)
        self._obs_scale = np.where(np.isfinite(high) & (high > 0), high, 1.0).astype(np.float32)
        self.obs_sp = Box(self._env.obs_low, self._env.obs_high, dtype=np.float32)
        self.act_sp = Box(self._env.act_low, self._env.act_high, dtype=np.float32)
        self.epoch = 0
        self.total_steps = 0
        self.created_trajectories = 0
        self.best_reward = float(np.finfo(np.float64).min)
        self.max_traj_len = 0
        self._seed = int(conf.get_root("seed"))
        self.cli = CliPrinter().instance
        self.params_are_views = True
        # arenas stay reachable for the run; episode arrays are views into them
        self._arenas: list[Any] = []

    def discount(self) -> float:
        """The environment discount."""
        return self._discount

    def close(self) -> None:
        """Nothing to close. There is no Gymnasium process."""
        return None

    @property
    def observation_space(self) -> Box:
        """Observation bounds."""
        return self.obs_sp

    @property
    def action_space(self) -> Box:
        """Action bounds."""
        return self.act_sp

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""
        return int(self._env.obs_dim)

    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return int(self._env.act_dim)

    def execute_policy(
        self,
        policy: Any,
        max_steps: int | None,
        batch_size: int,
        exploration: bool = True,
        render: bool = False,
        rew_field_name: str | None = None,
    ) -> TrajectoryCollection:
        """Collect episodes until ``batch_size`` transitions, sharded across host devices.

        The last wave can overshoot by less than one episode, and a wave is padded
        up to a multiple of ``env_jax_devices``. Padded episodes are dropped, so
        the padding does not inflate the batch. A wave that ends early is followed
        by another wave, as the Gymnasium loop collects another episode.

        Args:
            policy: A ``SharedGaussianPolicy``.
            max_steps: Steps per episode. This is the time limit, not a termination.
            batch_size: Minimum number of transitions.
            exploration: Sample when true, otherwise use the mode.
            render: Not supported.
            rew_field_name: Log the mean and std of the unscaled return under this name.

        Returns:
            The collected trajectories. Their arrays are views of the rollout arena.
        """
        if render or self.render:
            raise RuntimeError("render is not supported with rollout_backend=jax")
        if max_steps is None or max_steps < 1:
            raise ValueError("the JAX rollout needs a positive max_steps")
        shared = policy.export_for_rollout()
        trajectories: list[Trajectory] = []
        collected = 0
        while collected < batch_size:
            remaining = batch_size - collected
            launch = min(self.num_envs, int(np.ceil(remaining / max_steps)))
            launch = max(launch, 1)
            padded = int(np.ceil(launch / self.n_devices)) * self.n_devices
            result = self._roll(policy, shared, padded, max_steps, not exploration)
            self.params_are_views = result.params_are_views
            added = 0
            for index in range(launch):
                traj = self._trajectory(result, index)
                trajectories.append(traj)
                collected += len(traj)
                added += len(traj)
            if added == 0:
                raise RuntimeError("the JAX rollout produced an empty episode")
        self._log(trajectories, collected, rew_field_name)
        self.epoch += 1
        bundle = TrajectoryCollection(self, sum(len(traj) for traj in trajectories))
        bundle.extend(trajectories)
        bundle.print()
        return bundle

    def _roll(self, policy: Any, shared: Any, n_envs: int, horizon: int, deterministic: bool) -> Any:
        import jax

        master = _key(self._seed + self.epoch * 100003 + self.created_trajectories)
        keys = jax.random.split(master, n_envs)
        return sharded_rollout(
            self._env,
            policy.spec,
            shared,
            policy.fixed_numpy(),
            keys,
            horizon,
            self.n_devices,
            deterministic=deterministic,
            normalize=self._normalize,
            obs_scale=self._obs_scale,
            reward_scale=self._reward_scale,
            reward_offset=self._reward_offset,
            n_keep=n_envs,
        )

    def _trajectory(self, result: Any, index: int) -> Trajectory:
        episode = result.episode(index)
        length = int(result.lengths[index])
        rewards = _column(episode["rewards"])
        actions = episode["actions"]
        if actions.ndim == 1:
            actions = actions[:, None]
        traj = Trajectory.from_arrays(
            self.created_trajectories,
            self,
            episode["states"],
            actions,
            rewards,
            _column(episode["terminated"]),
            _column(episode["log_likelihoods"]),
            episode["raw_return"],
        )
        self._arenas.append(result)
        self.created_trajectories += 1
        self.max_traj_len = max(self.max_traj_len, length)
        return traj

    def _log(self, trajectories: list[Trajectory], steps: int, rew_field_name: str | None) -> None:
        if rew_field_name is None:
            self.total_steps += steps
            return
        returns = [traj.raw_return for traj in trajectories]
        mean_return = float(np.mean(returns))
        self.best_reward = max(self.best_reward, mean_return)
        log = self.mc["log"]
        log.log({rew_field_name: [np.mean(returns), np.std(returns)]})
        if log.has_field("steps"):
            log.log({"steps": [steps]})
        self.total_steps += steps


def _column(values: mx.array) -> mx.array:
    """A row vector as a column, matching the arrays the pandas trajectory stores."""
    if values.ndim == 1:
        return values[:, None]
    return values
