"""JAX environments for the opt-in rollout.

Pendulum and MountainCarContinuous are gymnax. InvertedPendulum and
InvertedDoublePendulum are the Gymnasium v5 / MuJoCo rigid-body equations,
integrated with the same classical RK4 MuJoCo uses for a slide and a hinge.
HalfCheetah is not here: MJX on CPU is far slower than Gymnasium, and a
hand-written cheetah is not this port.

``dtype`` is the dtype of the inverted-pendulum state. On an M3 Max (JAX 0.11.2,
MuJoCo 3.17) float64 matched MuJoCo to 1e-14 and float32 drifted by more than
1e-3 after about 50 steps; the returns still matched. Gymnax stays float32.
Stored observations and rewards are float32 either way. Float64 turns on JAX's
process-wide x64 mode.

The slider's joint limits and the constraint forces are not modelled.
``qfrc_constraint`` in the double-pendulum observation is 0. Stay away from
the rail if you compare against MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Inertia-from-geometry, density 1000, Gymnasium 1.4 / MuJoCo 3.17.
_IP_CART_MASS = 10.471975511965978
_IP_POLE_MASS = 5.018591641363306
_IP_POLE_IYY = 0.1887497668730885
_IP_COM_X = 0.0005
_IP_COM_Z = 0.3
_IP_GEAR = 100.0
_IP_DAMPING = 1.0
_IP_DT = 0.02
_IP_FRAME_SKIP = 2
_IP_CTRL = 3.0
_IP_ANGLE = 0.2
_IP_RESET = 0.01
_G = 9.81

_IDP_CART_MASS = 10.471975511965978
_IDP_POLE_MASS = 4.1987385815227585
_IDP_POLE_IYY = 0.15497066975016235
_IDP_COM_Z = 0.3
_IDP_LINK = 0.6
_IDP_GEAR = 500.0
_IDP_DAMPING = 0.05
_IDP_DT = 0.01
_IDP_FRAME_SKIP = 5
_IDP_CTRL = 1.0
_IDP_GX = 1e-5
_IDP_RESET = 0.1


def _dtype(name: str) -> Any:
    """Resolve the configured dynamics dtype. JAX is imported here, not at module import.

    float64 turns on JAX's x64 mode for this process. Without that, JAX silently
    stores the state as float32. Gymnax stays on the dtype it asks for.
    """
    import jax
    import jax.numpy as jnp

    match name:
        case "float64":
            jax.config.update("jax_enable_x64", True)
            return jnp.float64
        case "float32":
            return jnp.float32
        case _:
            raise ValueError(f"env_jax_dtype must be float32 or float64, got {name}")


def _rk4(acc: Any, q: Any, v: Any, u: Any, dt: Any) -> tuple[Any, Any]:
    """One classical RK4 step of ``(q, v)`` with ``qdot = v`` and ``vdot = acc``."""
    a1 = acc(q, v, u)
    q2, v2 = q + 0.5 * dt * v, v + 0.5 * dt * a1
    a2 = acc(q2, v2, u)
    q3, v3 = q + 0.5 * dt * v2, v + 0.5 * dt * a2
    a3 = acc(q3, v3, u)
    q4, v4 = q + dt * v3, v + dt * a3
    a4 = acc(q4, v4, u)
    qn = q + (dt / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
    vn = v + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return qn, vn


def _integrate(acc: Any, q: Any, v: Any, u: Any, dt: Any, frame_skip: int) -> tuple[Any, Any]:
    """``frame_skip`` RK4 substeps, which is one Gymnasium ``step``."""
    import jax

    def body(_i: Any, qv: tuple[Any, Any]) -> tuple[Any, Any]:
        return _rk4(acc, qv[0], qv[1], u, dt)

    return jax.lax.fori_loop(0, frame_skip, body, (q, v))


class InvertedPendulum:
    """Gymnasium ``InvertedPendulum-v5`` dynamics."""

    env_id = "InvertedPendulum-v5"
    obs_dim = 4
    act_dim = 1
    max_episode_steps = 1000

    def __init__(self, dtype: str = "float64") -> None:
        """Args:
        dtype: ``float64`` or ``float32`` for the integrated state.
        """
        import jax.numpy as jnp

        self.dtype_name = dtype
        self.dtype = _dtype(dtype)
        self.obs_low = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        self.obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.act_low = np.array([-_IP_CTRL], dtype=np.float32)
        self.act_high = np.array([_IP_CTRL], dtype=np.float32)
        self._dt = jnp.asarray(_IP_DT, dtype=self.dtype)

    def _acc(self, q: Any, v: Any, u: Any) -> Any:
        import jax.numpy as jnp

        _x, theta = q[0], q[1]
        x_dot, theta_dot = v[0], v[1]
        s, c = jnp.sin(theta), jnp.cos(theta)
        cart, pole, inertia = _IP_CART_MASS, _IP_POLE_MASS, _IP_POLE_IYY
        px, pz = _IP_COM_X, _IP_COM_Z
        m00 = cart + pole
        m01 = pole * (-px * s + pz * c)
        m11 = inertia + pole * (px**2 + pz**2)
        h0 = theta_dot**2 * pole * (pz * s + px * c) + _IP_GEAR * u - _IP_DAMPING * x_dot
        h1 = -_IP_DAMPING * theta_dot + pole * _G * (pz * s + px * c)
        det = m00 * m11 - m01 * m01
        a0 = (m11 * h0 - m01 * h1) / det
        a1 = (m00 * h1 - m01 * h0) / det
        return jnp.stack([a0, a1]).astype(self.dtype)

    def reset(self, key: Any) -> tuple[Any, tuple[Any, Any, Any]]:
        """Reset. Noise is uniform on position and velocity, scale 0.01, matching Gymnasium."""
        import jax.numpy as jnp

        noise = jax_uniform(key, (4,), -_IP_RESET, _IP_RESET, self.dtype)
        q, v = noise[:2], noise[2:]
        state = (q, v, jnp.asarray(0, dtype=jnp.int32))
        return self.observe(state), state

    def at(self, q: np.ndarray, v: np.ndarray) -> tuple[Any, Any, Any]:
        """A state at an explicit ``(q, v)``, for a comparison against MuJoCo."""
        import jax.numpy as jnp

        return (
            jnp.asarray(q, dtype=self.dtype),
            jnp.asarray(v, dtype=self.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def observe(self, state: tuple[Any, Any, Any]) -> Any:
        """``concat(qpos, qvel)`` as float32."""
        import jax.numpy as jnp

        q, v, _t = state
        return jnp.concatenate([q, v]).astype(jnp.float32)

    def step(self, state: tuple[Any, Any, Any], action: Any) -> tuple[Any, tuple[Any, Any, Any], Any, Any, Any]:
        """One environment step.

        Returns:
            Observation float32, next state, reward float32, terminated, truncated.
            Truncation is always false: the rollout's horizon is the time limit,
            and a time limit is not a termination.
        """
        import jax.numpy as jnp

        q, v, t = state
        u = jnp.clip(action.reshape(()), -_IP_CTRL, _IP_CTRL).astype(self.dtype)
        qn, vn = _integrate(self._acc, q, v, u, self._dt, _IP_FRAME_SKIP)
        nxt = (qn, vn, t + 1)
        obs = self.observe(nxt)
        terminated = jnp.logical_or(~jnp.isfinite(obs).all(), jnp.abs(qn[1]) > _IP_ANGLE)
        reward = jnp.where(terminated, jnp.float32(0.0), jnp.float32(1.0))
        return obs, nxt, reward, terminated, jnp.bool_(False)


class InvertedDoublePendulum:
    """Gymnasium ``InvertedDoublePendulum-v5`` dynamics, with no constraint solver."""

    env_id = "InvertedDoublePendulum-v5"
    obs_dim = 9
    act_dim = 1
    max_episode_steps = 1000

    def __init__(self, dtype: str = "float64") -> None:
        """Args:
        dtype: ``float64`` or ``float32`` for the integrated state.
        """
        import jax.numpy as jnp

        self.dtype_name = dtype
        self.dtype = _dtype(dtype)
        self.obs_low = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        self.obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.act_low = np.array([-_IDP_CTRL], dtype=np.float32)
        self.act_high = np.array([_IDP_CTRL], dtype=np.float32)
        self._dt = jnp.asarray(_IDP_DT, dtype=self.dtype)

    def _acc(self, q: Any, v: Any, u: Any) -> Any:
        import jax.numpy as jnp

        _x, a, c = q[0], q[1], q[2]
        x_dot, a_dot, c_dot = v[0], v[1], v[2]
        cart, m1, m2 = _IDP_CART_MASS, _IDP_POLE_MASS, _IDP_POLE_MASS
        i1, i2 = _IDP_POLE_IYY, _IDP_POLE_IYY
        pz, az = _IDP_COM_Z, _IDP_LINK
        s1, c1 = jnp.sin(a), jnp.cos(a)
        s2, c2 = jnp.sin(a + c), jnp.cos(a + c)
        sc, cc = jnp.sin(c), jnp.cos(c)
        m00 = cart + m1 + m2
        m01 = (m1 * pz + m2 * az) * c1 + (m2 * pz) * c2
        m02 = m2 * pz * c2
        m11 = 2.0 * m2 * az * pz * cc + (i1 + i2 + m1 * pz**2 + m2 * (az**2 + pz**2))
        m12 = m2 * az * pz * cc + (i2 + m2 * pz**2)
        m22 = i2 + m2 * pz**2
        mass = jnp.stack([jnp.stack([m00, m01, m02]), jnp.stack([m01, m11, m12]), jnp.stack([m02, m12, m22])])
        # gravity is (gx, 0, gz) with gz negative, so the potential term carries -gz on the sines
        gz = -_G
        h0 = (
            (m1 * pz + m2 * az) * a_dot**2 * s1
            + m2 * pz * (a_dot + c_dot) ** 2 * s2
            + _IDP_GEAR * u
            - _IDP_DAMPING * x_dot
            + m00 * _IDP_GX
        )
        h1 = (
            (2.0 * m2 * az * pz) * a_dot * c_dot * sc
            + (m2 * az * pz) * c_dot**2 * sc
            - _IDP_DAMPING * a_dot
            + (-gz) * (m1 * pz + m2 * az) * s1
            + (-gz) * (m2 * pz) * s2
            + _IDP_GX * (m1 * pz + m2 * az) * c1
            + _IDP_GX * (m2 * pz) * c2
        )
        h2 = -(m2 * az * pz) * a_dot**2 * sc - _IDP_DAMPING * c_dot + (-gz) * (m2 * pz) * s2 + _IDP_GX * (m2 * pz) * c2
        rhs = jnp.stack([h0, h1, h2]).astype(self.dtype)
        return jnp.linalg.solve(mass.astype(self.dtype), rhs)

    def tip(self, q: Any) -> tuple[Any, Any]:
        """World ``(x, z)`` of the second pole's tip. Upright, both angles 0, this is ``(x, 1.2)``."""
        import jax.numpy as jnp

        a, c = q[1], q[2]
        az = _IDP_LINK
        x = q[0] + az * jnp.sin(a) + az * jnp.sin(a + c)
        z = az * jnp.cos(a) + az * jnp.cos(a + c)
        return x, z

    def reset(self, key: Any) -> tuple[Any, tuple[Any, Any, Any]]:
        """Reset. Position noise is uniform, velocity noise is scaled normal, scale 0.1."""
        import jax
        import jax.numpy as jnp

        k_q, k_v = jax.random.split(key)
        q = jax_uniform(k_q, (3,), -_IDP_RESET, _IDP_RESET, self.dtype)
        v = (_IDP_RESET * jax.random.normal(k_v, (3,), dtype=self.dtype)).astype(self.dtype)
        state = (q, v, jnp.asarray(0, dtype=jnp.int32))
        return self.observe(state), state

    def at(self, q: np.ndarray, v: np.ndarray) -> tuple[Any, Any, Any]:
        """A state at an explicit ``(q, v)``."""
        import jax.numpy as jnp

        return (
            jnp.asarray(q, dtype=self.dtype),
            jnp.asarray(v, dtype=self.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def observe(self, state: tuple[Any, Any, Any]) -> Any:
        """The 9-d Gymnasium observation. The constraint force is 0."""
        import jax.numpy as jnp

        q, v, _t = state
        obs = jnp.concatenate(
            [
                q[:1],
                jnp.sin(q[1:]),
                jnp.cos(q[1:]),
                jnp.clip(v, -10.0, 10.0),
                jnp.zeros((1,), dtype=self.dtype),
            ]
        )
        return obs.astype(jnp.float32)

    def step(self, state: tuple[Any, Any, Any], action: Any) -> tuple[Any, tuple[Any, Any, Any], Any, Any, Any]:
        """One environment step. Terminated when the tip height is at most 1."""
        import jax.numpy as jnp

        q, v, t = state
        u = jnp.clip(action.reshape(()), -_IDP_CTRL, _IDP_CTRL).astype(self.dtype)
        qn, vn = _integrate(self._acc, q, v, u, self._dt, _IDP_FRAME_SKIP)
        nxt = (qn, vn, t + 1)
        obs = self.observe(nxt)
        x, z = self.tip(qn)
        terminated = z <= 1.0
        alive = jnp.where(terminated, jnp.float32(0.0), jnp.float32(10.0))
        dist = 0.01 * x**2 + (z - 2.0) ** 2
        vel = 1e-3 * vn[1] ** 2 + 5e-3 * vn[2] ** 2
        reward = (alive - dist - vel).astype(jnp.float32)
        return obs, nxt, reward, terminated, jnp.bool_(False)


class GymnaxEnv:
    """Pendulum or MountainCarContinuous via gymnax.

    Gymnax folds the time limit into ``done``. The rollout stores only real
    terminations, so a Pendulum time limit is a truncation and a MountainCar
    goal is a termination.
    """

    def __init__(self, env_id: str) -> None:
        """Args:
        env_id: ``Pendulum-v1`` or ``MountainCarContinuous-v0``.
        """
        import gymnax
        import jax.numpy as jnp

        self.env_id = env_id
        self.env, self.params = gymnax.make(env_id)
        self.dtype_name = "float32"
        obs_space = self.env.observation_space(self.params)
        act_space = self.env.action_space(self.params)
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = int(np.prod(act_space.shape))
        self.obs_low = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
        self.obs_high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)
        self.act_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
        self.act_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)
        self._zero = jnp.zeros((), dtype=jnp.int32)

    def reset(self, key: Any) -> tuple[Any, Any]:
        """gymnax reset."""
        import jax.numpy as jnp

        obs, state = self.env.reset(key, self.params)
        return jnp.asarray(obs, dtype=jnp.float32), _float32_state(state)

    def step(self, state: Any, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        """One gymnax step, with termination and truncation split apart."""
        import jax
        import jax.numpy as jnp

        key = jax.random.key(0) if hasattr(jax.random, "key") else jax.random.PRNGKey(0)
        obs, nxt, reward, _done, _info = self.env.step_env(key, state, action, self.params)
        terminated, truncated = self._flags(nxt)
        return (
            jnp.asarray(obs, dtype=jnp.float32),
            _float32_state(nxt),
            jnp.asarray(reward, dtype=jnp.float32).reshape(()),
            terminated,
            truncated,
        )

    def _flags(self, state: Any) -> tuple[Any, Any]:
        import jax.numpy as jnp

        params = self.params
        match self.env_id:
            case "Pendulum-v1":
                # gymnax's done is the time limit. The pole never terminates.
                truncated = state.time >= params.max_steps_in_episode
                return jnp.bool_(False), truncated
            case "MountainCarContinuous-v0":
                goal = (state.position >= params.goal_position) & (state.velocity >= params.goal_velocity)
                timed = state.time >= params.max_steps_in_episode
                return goal, timed & ~goal
            case _:
                raise NotImplementedError(self.env_id)


@dataclass(frozen=True)
class JaxEnv:
    """What the rollout needs from an environment, plus the spaces the policy reads."""

    env_id: str
    obs_dim: int
    act_dim: int
    obs_low: np.ndarray
    obs_high: np.ndarray
    act_low: np.ndarray
    act_high: np.ndarray
    dtype_name: str
    step: Any
    reset: Any


def make_jax_env(env_id: str, dtype: str = "float64") -> Any:
    """The JAX environment for a Gymnasium id.

    Args:
        env_id: One of the training environment ids.
        dtype: Dynamics dtype for the inverted pendulums. Gymnax ignores it.

    Returns:
        An environment with ``reset`` and ``step``.

    Raises:
        ValueError: For HalfCheetah and for anything else this backend does not implement.
    """
    match env_id:
        case "Pendulum-v1" | "MountainCarContinuous-v0":
            return GymnaxEnv(env_id)
        case "InvertedPendulum-v5":
            return InvertedPendulum(dtype)
        case "InvertedDoublePendulum-v5":
            return InvertedDoublePendulum(dtype)
        case "HalfCheetah-v5":
            raise ValueError(
                "HalfCheetah-v5 stays on Gymnasium. MJX on CPU is far slower than the "
                "existing env (measured on an M3 Max: 0.4k to 2.9k steps/s against 37k), "
                "and this backend does not add it. Use rollout_backend=gymnasium."
            )
        case _:
            raise ValueError(
                f"{env_id} has no JAX rollout. Supported: Pendulum-v1, MountainCarContinuous-v0, "
                "InvertedPendulum-v5, InvertedDoublePendulum-v5."
            )


def _float32_state(state: Any) -> Any:
    """Floating leaves as float32. x64 mode must not change the gymnax carry dtype."""
    import jax
    import jax.numpy as jnp

    def cast(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating) and dtype != jnp.float32:
            return leaf.astype(jnp.float32)
        return leaf

    return jax.tree.map(cast, state)


def jax_uniform(key: Any, shape: tuple[int, ...], lo: float, hi: float, dtype: Any) -> Any:
    """Uniform sample in ``[lo, hi]`` at ``dtype``."""
    import jax

    return jax.random.uniform(key, shape, minval=lo, maxval=hi, dtype=dtype)
