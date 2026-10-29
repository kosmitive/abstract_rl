"""Pure-JAX environments behind one small protocol, keyed by the Gymnasium ids used in ``abstract_rl/conf/env``.

Every adapter maps its backend onto Gymnasium's ``terminated`` / ``truncated`` semantics: a time limit is a
truncation, only a real terminal state is a termination (it is stored as ``done`` and not bootstrapped).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from abstract_rl.env.host_gym import HostGymEnv
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


_REGISTRY: dict[str, Callable[[], JaxEnv | HostGymEnv]] = {}


def register_env(env_id: str, factory: Callable[[], JaxEnv | HostGymEnv]) -> None:
    """Register (or replace) the factory of an environment id."""
    _REGISTRY[env_id] = factory


def make_env(env_id: str) -> JaxEnv | HostGymEnv:
    """Build the environment of a Gymnasium id (a JAX env, or a host Gymnasium env such as Humanoid-v5)."""
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
        return (
            env_state,
            jnp.asarray(env_state.obs, DTYPE),
            jnp.asarray(env_state.reward, DTYPE).reshape(()),
            jnp.asarray(env_state.done).reshape(()) > 0.5,
            jnp.asarray(False),
        )


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
        return (
            env_state,
            obs,
            jnp.asarray(env_state.reward, DTYPE).reshape(()),
            jnp.asarray(env_state.done).reshape(()) > 0.5,
            jnp.asarray(False),
        )


_ASSETS = Path(__file__).parent / "assets"


class MjxGymEnv:
    """A Gymnasium v5 MuJoCo environment rebuilt on MJX from Gymnasium's own XML (vendored under ``env/assets``).

    The env state is the ``mjx.Data``. A step sets ``ctrl = action`` (as Gymnasium's ``do_simulation``; MuJoCo clamps
    to the ``ctrlrange``, which is also the reported action bound) and calls ``mjx.step`` ``frame_skip`` times, so
    ``dt = timestep * frame_skip`` and the XML's integrator (RK4 for both pendulums) are Gymnasium's. Like MuJoCo's
    ``mj_step``, ``mjx.step`` runs the forward pass before integrating, so the derived quantities read after a step
    (``site_xpos``, ``qfrc_constraint``) lag the final substep exactly as in Gymnasium.

    Residual differences to Gymnasium: MJX computes in float32 (Gymnasium in float64), so from the same state and
    actions observations agree to about 1e-6 while upright and drift to about 1e-3 once a pendulum has fallen (chaotic
    motion); observations are float32; reset noise uses JAX's RNG, so it has the same distribution but not the same
    samples as a seeded Gymnasium env.
    """

    xml: str
    frame_skip: int
    observation_dim: int

    def __init__(self) -> None:
        import mujoco
        from mujoco import mjx

        self._mj_model = mujoco.MjModel.from_xml_path(str(_ASSETS / self.xml))
        self._model = mjx.put_model(self._mj_model)
        self._data0 = mjx.make_data(self._model)
        self._mjx = mjx
        self.action_dim = int(self._mj_model.nu)
        self.action_low, self.action_high = _ctrl_bounds(self._mj_model.actuator_ctrlrange, self.action_dim)
        self.obs_high = np.full(self.observation_dim, np.inf, np.float32)

    @property
    def mj_model(self) -> Any:
        """The MuJoCo model of the system, for rendering."""
        return self._mj_model

    def qpos(self, env_state: Any) -> Float[Array, " Q"]:
        """The joint positions of an environment state, for rendering."""
        return env_state.qpos

    def set_state(self, qpos: Float[Array | np.ndarray, " Q"], qvel: Float[Array | np.ndarray, " V"]) -> Any:
        """The env state of the given joint positions and velocities, like Gymnasium's ``set_state`` (with forward)."""
        data = self._data0.replace(qpos=jnp.asarray(qpos, DTYPE), qvel=jnp.asarray(qvel, DTYPE))
        return self._mjx.forward(self._model, data)

    def observe(self, env_state: Any) -> Float[Array, " Ds"]:
        """The observation of an env state."""
        raise NotImplementedError

    def _reset_qpos_qvel(self, key: PRNGKeyArray) -> tuple[Float[Array, " Q"], Float[Array, " V"]]:
        raise NotImplementedError

    def _reward_terminated(self, env_state: Any, obs: Float[Array, " Ds"]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
        raise NotImplementedError

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        env_state = self.set_state(*self._reset_qpos_qvel(key))
        return env_state, self.observe(env_state)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        data = env_state.replace(ctrl=jnp.asarray(action, DTYPE).reshape(self.action_dim))
        data = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: self._mjx.step(self._model, d), data)
        obs = self.observe(data)
        reward, terminated = self._reward_terminated(data, obs)
        return data, obs, reward, terminated, jnp.asarray(False)


class InvertedPendulumV5(MjxGymEnv):
    """Gymnasium ``InvertedPendulum-v5``: frame_skip 2 (dt 0.04), RK4, obs ``[qpos, qvel]`` (4), action = force in
    [-3, 3]. Reward 1 per step and 0 on the terminating step; terminates when ``|qpos[1]| > 0.2`` or the observation
    is not finite. Reset noise uniform in +-0.01 on qpos and qvel."""

    xml = "inverted_pendulum.xml"
    frame_skip = 2
    observation_dim = 4
    reset_noise_scale = 0.01

    def observe(self, env_state: Any) -> Float[Array, " Ds"]:
        return jnp.concatenate([env_state.qpos, env_state.qvel]).astype(DTYPE)

    def _reset_qpos_qvel(self, key: PRNGKeyArray) -> tuple[Float[Array, " Q"], Float[Array, " V"]]:
        k_pos, k_vel = jax.random.split(key)
        noise = self.reset_noise_scale
        qpos = self._mj_model.qpos0 + jax.random.uniform(k_pos, (self._mj_model.nq,), DTYPE, -noise, noise)
        qvel = jax.random.uniform(k_vel, (self._mj_model.nv,), DTYPE, -noise, noise)
        return qpos, qvel

    def _reward_terminated(self, env_state: Any, obs: Float[Array, " Ds"]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
        terminated = ~jnp.all(jnp.isfinite(obs)) | (jnp.abs(obs[1]) > 0.2)
        return jnp.where(terminated, 0.0, 1.0).astype(DTYPE), terminated


class InvertedDoublePendulumV5(MjxGymEnv):
    """Gymnasium ``InvertedDoublePendulum-v5``: frame_skip 5 (dt 0.05), RK4, action = force in [-1, 1].

    Observation (9): cart x, sin and cos of both hinge angles, ``clip(qvel, -10, 10)`` (3) and
    ``clip(qfrc_constraint, -10, 10)[:1]``. Reward ``10 * (not terminated) - (0.01 x^2 + (y - 2)^2)
    - (1e-3 v1^2 + 5e-3 v2^2)`` with ``(x, y)`` the tip site position and ``v1, v2`` the hinge velocities; terminates
    when ``y <= 1``. Reset noise uniform in +-0.1 on qpos and ``0.1 * normal`` on qvel."""

    xml = "inverted_double_pendulum.xml"
    frame_skip = 5
    observation_dim = 9
    reset_noise_scale = 0.1
    healthy_reward = 10.0

    def observe(self, env_state: Any) -> Float[Array, " Ds"]:
        qpos, qvel = env_state.qpos, env_state.qvel
        return jnp.concatenate(
            [
                qpos[:1],
                jnp.sin(qpos[1:]),
                jnp.cos(qpos[1:]),
                jnp.clip(qvel, -10, 10),
                jnp.clip(env_state.qfrc_constraint, -10, 10)[:1],
            ]
        ).astype(DTYPE)

    def _reset_qpos_qvel(self, key: PRNGKeyArray) -> tuple[Float[Array, " Q"], Float[Array, " V"]]:
        k_pos, k_vel = jax.random.split(key)
        noise = self.reset_noise_scale
        qpos = self._mj_model.qpos0 + jax.random.uniform(k_pos, (self._mj_model.nq,), DTYPE, -noise, noise)
        qvel = jax.random.normal(k_vel, (self._mj_model.nv,), DTYPE) * noise
        return qpos, qvel

    def _reward_terminated(self, env_state: Any, obs: Float[Array, " Ds"]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
        x, _, y = env_state.site_xpos[0]
        v1, v2 = env_state.qvel[1], env_state.qvel[2]
        terminated = y <= 1
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        reward = self.healthy_reward * (~terminated) - dist_penalty - vel_penalty
        return jnp.asarray(reward, DTYPE).reshape(()), terminated


register_env("InvertedPendulum-v5", InvertedPendulumV5)
register_env("InvertedDoublePendulum-v5", InvertedDoublePendulumV5)
# HalfCheetah-v5 on Brax matches Gymnasium v5 in dt (0.05), gear, ctrl = action in [-1, 1], reward weights, 17-dim
# observation, reset noise and no termination. Residuals: Brax's XML runs the solver for 4 instead of 100 iterations and
# lets only head, thighs and feet collide with the floor; the forward velocity uses the torso ``xpos``, which
# ``mjx.step`` leaves one 0.01 s substep behind ``qpos``, so a step's reward is shifted by up to about 0.3 while an
# episode's return differs only by the final substep's displacement over dt.
register_env("HalfCheetah-v5", lambda: BraxEnv("halfcheetah"))
register_env("AcrobotSwingup-v0", lambda: PlaygroundEnv("AcrobotSwingup"))


def _humanoid_v5() -> HostGymEnv:
    """Gymnasium ``Humanoid-v5`` stepped on the host. An MJX or Brax humanoid is not this environment."""
    return HostGymEnv("Humanoid-v5")


register_env("Humanoid-v5", _humanoid_v5)
