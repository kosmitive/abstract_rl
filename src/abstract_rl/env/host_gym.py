"""Gymnasium environments stepped on the host.

Humanoid-v5 is one of these. An MJX or Brax humanoid is a different robot and reward from Gymnasium v5, so it is not
used here. A later MJX port of Gymnasium's own Humanoid XML could sit next to the pendulum adapters in ``jax_envs``.
"""

from typing import Any

import numpy as np


class HostGymEnv:
    """One Gymnasium environment, stepped with NumPy outside of ``jax.jit``.

    The class is not a ``JaxEnv``: ``reset`` and ``step`` of that protocol carry a functional state, and a Gymnasium
    env cannot be vmapped. Rollouts call ``reset_obs`` and ``step_obs`` instead. Gymnasium is imported on construction,
    not when this module is imported.
    """

    def __init__(self, env_id: str) -> None:
        """Open ``env_id``.

        Args:
            env_id: A Gymnasium id, for example ``Humanoid-v5``.

        Raises:
            ModuleNotFoundError: The ``gym`` extra is not installed (``gymnasium[mujoco]``, which brings ``imageio``).
        """
        import gymnasium as gym

        self.env_id = env_id
        try:
            gym_env: Any = gym.make(env_id)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{env_id} is a host Gymnasium environment; install the gym extra (gymnasium[mujoco])"
            ) from exc
        self._gym: Any = gym_env
        obs_space: Any = gym_env.observation_space
        act_space: Any = gym_env.action_space
        if obs_space.shape is None or act_space.shape is None:
            raise TypeError(f"{env_id} needs box observation and action spaces")
        self.observation_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(act_space.shape))
        self.action_low = np.asarray(act_space.low, np.float32).reshape(-1)
        self.action_high = np.asarray(act_space.high, np.float32).reshape(-1)
        self.obs_high = np.asarray(obs_space.high, np.float32).reshape(-1)

    @property
    def mj_model(self) -> Any:
        """The MuJoCo model, for offscreen rendering of recorded ``qpos``."""
        return self._gym.unwrapped.model

    def qpos(self) -> np.ndarray:
        """Joint positions after the last reset or step, for rendering."""
        return np.asarray(self._gym.unwrapped.data.qpos, np.float32).copy()

    def reset_obs(self, seed: int) -> np.ndarray:
        """Reset and return the flat float32 observation.

        Args:
            seed: Seed forwarded to Gymnasium's ``reset``.

        Returns:
            The observation, shape ``(observation_dim,)``.
        """
        obs, _info = self._gym.reset(seed=int(seed))
        return np.asarray(obs, np.float32).reshape(-1)

    def step_obs(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
        """Step with one action.

        Args:
            action: Flat action inside the action bounds.

        Returns:
            The next observation, the raw reward, ``terminated`` and ``truncated``.
        """
        obs, reward, terminated, truncated, _info = self._gym.step(np.asarray(action, np.float32))
        return (
            np.asarray(obs, np.float32).reshape(-1),
            float(reward),
            bool(terminated),
            bool(truncated),
        )

    def close(self) -> None:
        """Close the Gymnasium environment. A second call does nothing."""
        gym_env = getattr(self, "_gym", None)
        if gym_env is not None:
            gym_env.close()
            self._gym = None
