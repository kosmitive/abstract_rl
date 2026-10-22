"""Pure-JAX environments behind one small protocol, keyed by the Gymnasium ids used in ``abstract_rl/conf/env``.

Every adapter maps its backend onto Gymnasium's ``terminated`` / ``truncated`` semantics: a time limit is a
truncation, only a real terminal state is a termination (it is stored as ``done`` and not bootstrapped).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from abstract_rl.misc.typing import DTYPE


@dataclass(frozen=True)
class Bounds:
    """Box bounds of an observation or action space (``low`` / ``high`` like ``gymnasium.spaces.Box``)."""

    low: np.ndarray
    high: np.ndarray


@runtime_checkable
class JaxEnv(Protocol):
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_high: np.ndarray

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]: ...

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]: ...


_REGISTRY: dict[str, Callable[[], JaxEnv]] = {}


def register_env(env_id: str, factory: Callable[[], JaxEnv]) -> None:
    """Register (or replace) the factory of an environment id."""
    _REGISTRY[env_id] = factory


def make_env(env_id: str) -> JaxEnv:
    """Build the JAX environment of a Gymnasium id."""
    if env_id not in _REGISTRY:
        raise KeyError(f"unknown environment {env_id}, known: {sorted(_REGISTRY)}")
    return _REGISTRY[env_id]()


def _flat(x: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, np.float32), shape).reshape(-1).astype(np.float32)


class GymnaxEnv:
    """gymnax environment, stepped with ``step_env`` (no auto reset); termination and time limit give the two flags."""

    def __init__(self, env_id: str) -> None:
        import gymnax

        self._env, self._params = gymnax.make(env_id)

        act, obs = self._env.action_space(self._params), self._env.observation_space(self._params)
        self._action_shape = tuple(act.shape)
        self.action_dim = int(np.prod(act.shape))
        self.observation_dim = int(np.prod(obs.shape))
        self.action_low, self.action_high = _flat(act.low, act.shape), _flat(act.high, act.shape)
        self.obs_high = _flat(obs.high, obs.shape)

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        obs, env_state = self._env.reset_env(key, self._params)
        return env_state, jnp.asarray(obs, DTYPE).reshape(-1)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        obs, env_state, reward, done, _ = self._env.step_env(
            key, env_state, action.reshape(self._action_shape), self._params
        )
        # gymnax 1.0 returns only the natural termination from ``step_env``; the time limit is ``is_truncated``
        terminated = jnp.asarray(done, bool).reshape(())
        truncated = jnp.asarray(self._env.is_truncated(env_state, self._params), bool).reshape(()) & ~terminated
        return (
            env_state,
            jnp.asarray(obs, DTYPE).reshape(-1),
            jnp.asarray(reward, DTYPE).reshape(()),
            terminated,
            truncated,
        )


register_env("Pendulum-v1", lambda: GymnaxEnv("Pendulum-v1"))
register_env("MountainCarContinuous-v0", lambda: GymnaxEnv("MountainCarContinuous-v0"))


def _ctrl_bounds(ctrlrange: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Actuator bounds of a MuJoCo model; unlimited actuators (an all-zero range) fall back to [-1, 1]."""
    ctrl = np.asarray(ctrlrange, np.float32).reshape(action_dim, 2)
    if not np.all(ctrl[:, 0] < ctrl[:, 1]):
        return np.full(action_dim, -1.0, np.float32), np.full(action_dim, 1.0, np.float32)
    return ctrl[:, 0].copy(), ctrl[:, 1].copy()


class BraxEnv:
    """Brax environment on the MJX backend. Brax ``done`` is a termination; time limits are applied by ``max_steps``."""

    def __init__(self, name: str) -> None:
        from brax import envs

        self._env = envs.get_environment(name, backend="mjx")
        self.observation_dim = int(self._env.observation_size)
        self.action_dim = int(self._env.action_size)
        self.action_low, self.action_high = _ctrl_bounds(self._env.sys.actuator_ctrlrange, self.action_dim)
        self.obs_high = np.full(self.observation_dim, np.inf, np.float32)

    @property
    def mj_model(self) -> Any:
        """The MuJoCo model of the system, for rendering."""
        return self._env.sys.mj_model

    def qpos(self, env_state: Any) -> Float[Array, " Q"]:
        """The joint positions of an environment state, for rendering."""
        return env_state.pipeline_state.qpos

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        env_state = self._env.reset(key)
        return env_state, jnp.asarray(env_state.obs, DTYPE)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        env_state = self._env.step(env_state, action)
        return (env_state, jnp.asarray(env_state.obs, DTYPE), jnp.asarray(env_state.reward, DTYPE).reshape(()),
                jnp.asarray(env_state.done).reshape(()) > 0.5, jnp.asarray(False))


class PlaygroundEnv:
    """MuJoCo Playground environment (MJX). Its ``done`` is a termination; time limits are applied by ``max_steps``."""

    def __init__(self, name: str) -> None:
        from mujoco_playground import registry

        self._env = registry.load(name)
        self.observation_dim = int(np.prod(np.asarray(self._env.observation_size)))
        self.action_dim = int(self._env.action_size)
        self.action_low, self.action_high = _ctrl_bounds(self._env.mj_model.actuator_ctrlrange, self.action_dim)
        self.obs_high = np.full(self.observation_dim, np.inf, np.float32)

    @property
    def mj_model(self) -> Any:
        """The MuJoCo model of the environment, for rendering."""
        return self._env.mj_model

    def qpos(self, env_state: Any) -> Float[Array, " Q"]:
        """The joint positions of an environment state, for rendering."""
        return env_state.data.qpos

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        env_state = self._env.reset(key)
        return env_state, jnp.asarray(env_state.obs, DTYPE).reshape(-1)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        env_state = self._env.step(env_state, action)
        obs = jnp.asarray(env_state.obs, DTYPE).reshape(-1)
        return (env_state, obs, jnp.asarray(env_state.reward, DTYPE).reshape(()),
                jnp.asarray(env_state.done).reshape(()) > 0.5, jnp.asarray(False))


register_env("InvertedPendulum-v5", lambda: BraxEnv("inverted_pendulum"))
register_env("InvertedDoublePendulum-v5", lambda: BraxEnv("inverted_double_pendulum"))
register_env("HalfCheetah-v5", lambda: BraxEnv("halfcheetah"))
register_env("AcrobotSwingup-v0", lambda: PlaygroundEnv("AcrobotSwingup"))
