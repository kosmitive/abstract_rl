"""Parity of the in-repo JAX steps against Gymnasium, from an identical physics state and action."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.env.jax_classic import (
    mountain_obs,
    mountain_step,
    pendulum_obs,
    pendulum_step,
)

jax.config.update("jax_enable_x64", True)


def _unwrapped(env_id: str):
    return gym.make(env_id).unwrapped


def _roll(step, observe, gym_env, state0: np.ndarray, actions: np.ndarray, dtype) -> tuple[float, float, float]:
    """Max absolute error of state, observation and reward over a shared action sequence."""
    gym_env.state = np.array(state0, dtype=np.float64)
    jax_state = jnp.asarray(state0, dtype=dtype)
    state_err = 0.0
    obs_err = 0.0
    rew_err = 0.0
    for action in actions:
        gym_obs, gym_rew, gym_term, _trunc, _info = gym_env.step(np.asarray(action, dtype=np.float64))
        jax_state, jax_rew, jax_term = step(jax_state, jnp.asarray(action, dtype=dtype))
        state_gap = np.asarray(gym_env.state, dtype=np.float64) - np.asarray(jax_state)
        obs_gap = np.asarray(gym_obs, dtype=np.float64) - np.asarray(observe(jax_state))
        state_err = max(state_err, float(np.max(np.abs(state_gap))))
        obs_err = max(obs_err, float(np.max(np.abs(obs_gap))))
        rew_err = max(rew_err, abs(float(gym_rew) - float(jax_rew)))
        assert bool(gym_term) == bool(jax_term)
    return state_err, obs_err, rew_err


def test_pendulum_float64_matches_gym_and_float32_stays_close():
    env = _unwrapped("Pendulum-v1")
    actions = np.linspace(-2.5, 2.5, 40).reshape(-1, 1)
    state0 = np.array([0.4, -0.7])
    f64 = _roll(pendulum_step, pendulum_obs, env, state0, actions, jnp.float64)
    f32 = _roll(pendulum_step, pendulum_obs, env, state0, actions, jnp.float32)
    # Gymnasium keeps the Pendulum state in float64. A float64 JAX step tracks it.
    # Float32 is the training dtype and drifts; 40 steps stay well under 1e-3 on this state.
    assert f64[0] < 1e-8 and f64[1] < 1e-6 and f64[2] < 1e-8
    assert f32[0] < 1e-3 and f32[1] < 1e-3 and f32[2] < 1e-4


def test_mountain_car_matches_gym_including_left_wall():
    env = _unwrapped("MountainCarContinuous-v0")
    # Push into the left wall, then a second action that would keep a negative velocity.
    wall_actions = np.array([[-1.0], [-1.0], [0.3], [1.0]], dtype=np.float64)
    state0 = np.array([-1.19, -0.05])
    # Gymnasium stores this state as float32 after every step, so an open-loop float64
    # rollout drifts by about one float32 ulp per step. Rewards still match.
    f64 = _roll(mountain_step, mountain_obs, env, state0, wall_actions, jnp.float64)
    assert f64[0] < 1e-6 and f64[2] < 1e-8
    env.state = np.array(state0, dtype=np.float64)
    mountain_step(jnp.asarray(state0, dtype=jnp.float64), jnp.asarray(wall_actions[0]))
    # The wall rule zeros a negative velocity at position -1.2. gymnax instead zeros velocity at the goal.
    after = mountain_step(jnp.asarray([-1.2, -0.01], dtype=jnp.float64), jnp.zeros(1, dtype=jnp.float64))[0]
    assert float(after[0]) == -1.2
    assert float(after[1]) == 0.0

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.5, 1.5, size=(30, 1))
    f64 = _roll(mountain_step, mountain_obs, env, np.array([-0.5, 0.0]), actions, jnp.float64)
    f32 = _roll(mountain_step, mountain_obs, env, np.array([-0.5, 0.0]), actions, jnp.float32)
    assert f64[0] < 1e-6 and f64[2] < 1e-6
    assert f32[0] < 1e-4 and f32[2] < 1e-4


def test_mountain_car_goal_is_a_termination_and_time_limit_is_not():
    state = jnp.array([0.44, 0.06], dtype=jnp.float64)
    nxt, reward, terminated = mountain_step(state, jnp.array([0.5], dtype=jnp.float64))
    assert bool(terminated)
    assert float(nxt[0]) >= 0.45
    # +100 minus 0.1 * action^2, on the raw action
    assert abs(float(reward) - (100.0 - 0.1 * 0.25)) < 1e-8
    # A non-goal step does not terminate, including one that would be step 999 of a time limit.
    _nxt, _reward, still = mountain_step(jnp.array([-0.5, 0.0], dtype=jnp.float64), jnp.zeros(1))
    assert not bool(still)


def test_jax_collection_matches_the_gym_trajectory_contract(make_pendulum_mc):
    mc = make_pendulum_mc(env_name="MountainCarContinuous-v0", env_vector_envs=2, max_steps=20)
    with mc["conf"].ns("policy"):
        from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

        policy = UnivariateGaussianPolicy(mc)
    collection = mc.env.execute_policy(policy, 20, 30, exploration=True)
    assert sum(len(traj) for traj in collection) == 30
    assert all(len(traj) <= 20 for traj in collection)
    assert all(traj.states.shape[1] == 2 for traj in collection)
    assert all(np.all((traj.dones == 0) | (traj.dones == 1)) for traj in collection)
    # A done, when present, is the last transition. A time-limit or budget cut is not a done.
    for traj in collection:
        dones = np.asarray(traj.dones).reshape(-1)
        if np.any(dones == 1):
            assert dones[-1] == 1
            assert np.all(dones[:-1] == 0)
