"""JAX environments against Gymnasium, and the dtype of the inverted pendulums."""

import numpy as np
import pytest

pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("gymnax")

from abstract_rl.interop.envs import InvertedDoublePendulum, InvertedPendulum, make_jax_env  # noqa: E402


def test_halfcheetah_stays_on_gymnasium():
    """MJX is not part of this backend."""
    with pytest.raises(ValueError, match="HalfCheetah"):
        make_jax_env("HalfCheetah-v5")


def test_unknown_env_is_rejected():
    """Anything else is an explicit error, not a silent Gymnasium fallback."""
    with pytest.raises(ValueError, match="no JAX rollout"):
        make_jax_env("CartPole-v1")


def _roll(jax_env, gym_env, q, v, actions):
    gym_env.reset(seed=0)
    gym_env.unwrapped.set_state(np.asarray(q, np.float64), np.asarray(v, np.float64))
    state = jax_env.at(q, v)
    errors = []
    for action in actions:
        gym_env.step(np.asarray(action, np.float64))
        _obs, state, _reward, terminated, _truncated = jax_env.step(state, jnp.asarray(action))
        got = np.concatenate([np.asarray(state[0]), np.asarray(state[1])])
        want = np.concatenate([gym_env.unwrapped.data.qpos, gym_env.unwrapped.data.qvel])
        errors.append(np.max(np.abs(got - want)))
        if bool(np.asarray(terminated)) or abs(float(gym_env.unwrapped.data.qpos[1])) > 0.19:
            break
    return max(errors), len(errors)


def test_inverted_pendulum_float64_matches_mujoco():
    """float64 RK4 matches MuJoCo's q and v. float32 is the fast path and is not this close."""
    mujoco = pytest.importorskip("mujoco")
    assert mujoco.__version__
    gym = pytest.importorskip("gymnasium")
    actions = np.linspace(-0.4, 0.4, 40)[:, None]
    q, v = np.array([0.0, 0.01]), np.array([0.0, 0.0])
    gym64 = gym.make("InvertedPendulum-v5")
    err64, n64 = _roll(InvertedPendulum("float64"), gym64, q, v, actions)
    gym64.close()
    gym32 = gym.make("InvertedPendulum-v5")
    err32, _n32 = _roll(InvertedPendulum("float32"), gym32, q, v, actions)
    gym32.close()
    assert n64 >= 5
    assert err64 < 1e-12
    assert err32 > err64 * 1e3


def test_inverted_double_pendulum_float64_matches_mujoco_state():
    """q and v match. The reward's site position is read by Gymnasium from inside the step and can differ by ~1e-6."""
    pytest.importorskip("mujoco")
    gym = pytest.importorskip("gymnasium")
    env = InvertedDoublePendulum("float64")
    gym_env = gym.make("InvertedDoublePendulum-v5")
    q, v = np.zeros(3), np.zeros(3)
    actions = 0.05 * np.sin(np.linspace(0, 4, 15))[:, None]
    gym_env.reset(seed=0)
    gym_env.unwrapped.set_state(q, v)
    state = env.at(q, v)
    worst = 0.0
    for action in actions:
        _obs_g, reward_g, terminated_g, _truncated, _info = gym_env.step(action)
        obs, state, reward, terminated, _truncated_j = env.step(state, jnp.asarray(action))
        qv = np.concatenate([np.asarray(state[0]), np.asarray(state[1])])
        want = np.concatenate([gym_env.unwrapped.data.qpos, gym_env.unwrapped.data.qvel])
        worst = max(worst, float(np.max(np.abs(qv - want))))
        np.testing.assert_allclose(np.asarray(obs)[-1], 0.0, atol=0.0)
        assert abs(float(reward) - float(reward_g)) < 1e-4
        assert bool(np.asarray(terminated)) == bool(terminated_g)
        if terminated_g:
            break
    gym_env.close()
    assert worst < 1e-12


def test_pendulum_time_limit_is_not_a_termination():
    """gymnax ends Pendulum at 200 steps. That is a truncation, so a short horizon stores no termination."""
    import jax

    env = make_jax_env("Pendulum-v1")
    key = jax.random.key(0) if hasattr(jax.random, "key") else jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert obs.shape == (3,)
    for _ in range(5):
        obs, state, reward, terminated, truncated = env.step(state, jnp.zeros((1,)))
        assert bool(terminated) is False
        assert bool(truncated) is False
        assert np.isfinite(float(reward))


def test_mountain_car_goal_is_a_termination_and_the_time_limit_is_not():
    """The goal ends the episode. Reaching the step cap without the goal is a truncation."""
    import jax

    env = make_jax_env("MountainCarContinuous-v0")
    # place the car on the goal by stepping a state the reset just produced and overwriting it
    key = jax.random.key(1) if hasattr(jax.random, "key") else jax.random.PRNGKey(1)
    _obs, state = env.reset(key)
    state = state.replace(position=jnp.float32(0.5), velocity=jnp.float32(0.01), time=jnp.int32(1))
    _obs, _state, reward, terminated, truncated = env.step(state, jnp.ones((1,)))
    assert bool(terminated) is True
    assert bool(truncated) is False
    assert float(reward) > 50

    state = state.replace(position=jnp.float32(-0.5), velocity=jnp.float32(0.0), time=jnp.int32(998))
    _obs, nxt, _reward, terminated, truncated = env.step(state, jnp.zeros((1,)))
    assert bool(terminated) is False
    assert bool(truncated) is True
    assert int(nxt.time) == 999
