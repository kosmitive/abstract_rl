"""JAX steps of the classic-control environments used for training rollouts.

Gymnasium stays the evaluation environment, the renderer, and the fallback for every environment
that has no matching step here. The formulas are the Gymnasium ones (Pendulum-v1 and
MountainCarContinuous-v0), not gymnax.

gymnax Pendulum-v1 uses the same ODE, reward and observation. gymnax MountainCarContinuous-v0 does
not: it zeros the velocity when the car is past the goal with a negative velocity, and it folds the
time limit into ``done``. This module zeros the velocity only on the left wall, and a time limit is
not a termination. Reset ranges match Gymnasium. The generator is JAX, so a seeded Gymnasium reset
does not reproduce the same sequence.

The training rollout runs these steps in float32, which is the dtype of the networks. A float64
step from the same state tracks Gymnasium (its Pendulum state is float64). Float32 drifts; the
parity test records both.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

PENDULUM = "Pendulum-v1"
MOUNTAIN_CAR = "MountainCarContinuous-v0"
JAX_TRAIN_ENVS = frozenset({PENDULUM, MOUNTAIN_CAR})


def angle_normalize(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Wrap an angle into ``[-pi, pi]``, matching Gymnasium's ``angle_normalize``."""
    pi = jnp.asarray(jnp.pi, dtype=x.dtype)
    return jnp.mod(x + pi, 2 * pi) - pi


def pendulum_reset(key: PRNGKeyArray) -> Float[Array, "2"]:
    """Pendulum physics state ``(theta, theta_dot)``. Angle on ``[-pi, pi]``, speed on ``[-1, 1]``."""
    key_th, key_vel = jax.random.split(key)
    theta = jax.random.uniform(key_th, (), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)
    theta_dot = jax.random.uniform(key_vel, (), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    return jnp.stack([theta, theta_dot])


def pendulum_obs(state: Float[Array, "2"]) -> Float[Array, "3"]:
    """Observation ``(cos theta, sin theta, theta_dot)``."""
    return jnp.stack([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])


def pendulum_step(
    state: Float[Array, "2"], action: Float[Array, " act"]
) -> tuple[Float[Array, "2"], Float[Array, ""], Float[Array, ""]]:
    """One Pendulum-v1 transition. The environment itself never terminates.

    Torque is clipped to ±2. The cost is ``angle_normalize(theta)² + 0.1 theta_dot² + 0.001 u²``.
    The integrator is the semi-implicit Euler step of Gymnasium, with ``theta_dot`` clipped to ±8.
    """
    theta, theta_dot = state[0], state[1]
    torque = jnp.clip(action.reshape(-1)[0], -2.0, 2.0)
    costs = jnp.square(angle_normalize(theta)) + 0.1 * jnp.square(theta_dot) + 0.001 * jnp.square(torque)
    new_theta_dot = theta_dot + (3.0 * 10.0 / 2.0 * jnp.sin(theta) + 3.0 * torque) * 0.05
    new_theta_dot = jnp.clip(new_theta_dot, -8.0, 8.0)
    new_theta = theta + new_theta_dot * 0.05
    nxt = jnp.stack([new_theta, new_theta_dot]).astype(state.dtype)
    return nxt, -costs.astype(state.dtype), jnp.zeros((), dtype=jnp.bool_)


def mountain_reset(key: PRNGKeyArray) -> Float[Array, "2"]:
    """Mountain-car state ``(position, velocity)``. Position on ``[-0.6, -0.4]``, velocity 0."""
    position = jax.random.uniform(key, (), minval=-0.6, maxval=-0.4, dtype=jnp.float32)
    return jnp.stack([position, jnp.zeros((), dtype=jnp.float32)])


def mountain_obs(state: Float[Array, "2"]) -> Float[Array, "2"]:
    """The observation is the physics state."""
    return state


def mountain_step(
    state: Float[Array, "2"], action: Float[Array, " act"]
) -> tuple[Float[Array, "2"], Float[Array, ""], Float[Array, ""]]:
    """One MountainCarContinuous-v0 transition.

    The force is the action clipped to ±1. Velocity uses power 0.0015 and gravity
    ``0.0025 cos(3 position)``, then speed is clipped to ±0.07 and position to ``[-1.2, 0.6]``.
    Velocity is set to 0 only when the position equals the left wall and the velocity is negative.
    The reward penalises the raw (unclipped) action and adds 100 at the goal. The 999-step cutoff
    is a time limit and is not returned as a termination.
    """
    position, velocity = state[0], state[1]
    raw = action.reshape(-1)[0]
    force = jnp.clip(raw, -1.0, 1.0)
    velocity = velocity + force * 0.0015 - 0.0025 * jnp.cos(3.0 * position)
    velocity = jnp.clip(velocity, -0.07, 0.07)
    position = position + velocity
    position = jnp.clip(position, -1.2, 0.6)
    wall = jnp.asarray(-1.2, dtype=position.dtype)
    velocity = jnp.where((position == wall) & (velocity < 0), jnp.zeros((), dtype=velocity.dtype), velocity)
    terminated = (position >= jnp.asarray(0.45, dtype=position.dtype)) & (velocity >= 0)
    reward = jnp.where(terminated, jnp.asarray(100.0, dtype=state.dtype), jnp.zeros((), dtype=state.dtype))
    reward = reward - jnp.square(raw) * 0.1
    nxt = jnp.stack([position, velocity]).astype(state.dtype)
    return nxt, reward.astype(state.dtype), terminated
