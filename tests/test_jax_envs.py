import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.env.jax_envs import JaxEnv, make_env

KEY = jax.random.key(0)


def _roll(env, steps, action=None):
    st, obs = env.reset(KEY)
    out = []
    a = jnp.zeros(env.action_dim) if action is None else action
    for k in range(steps):
        st, obs, r, term, trunc = jax.jit(env.step)(st, a, jax.random.key(k))
        out.append((obs, r, term, trunc))
    return out


def _contract(env, obs_dim, act_dim):
    assert isinstance(env, JaxEnv)
    assert env.observation_dim == obs_dim and env.action_dim == act_dim
    assert env.action_low.shape == (act_dim,) and env.action_high.shape == (act_dim,)
    assert env.obs_high.shape == (obs_dim,)
    st, obs = jax.jit(env.reset)(KEY)
    assert obs.shape == (obs_dim,) and obs.dtype == jnp.float32
    st, obs, r, term, trunc = jax.jit(env.step)(st, jnp.zeros(act_dim), KEY)
    assert obs.shape == (obs_dim,) and r.shape == () and term.shape == () and trunc.shape == ()
    # the env is vmappable over lanes
    sts, _ = jax.vmap(env.reset)(jax.random.split(KEY, 3))
    jax.vmap(env.step)(sts, jnp.zeros((3, act_dim)), jax.random.split(KEY, 3))


def test_pendulum_contract_and_time_limit_is_a_truncation():
    env = make_env("Pendulum-v1")
    _contract(env, 3, 1)
    np.testing.assert_allclose(env.obs_high, [1.0, 1.0, 8.0])
    np.testing.assert_allclose(env.action_low, [-2.0])
    np.testing.assert_allclose(env.action_high, [2.0])
    steps = _roll(env, 200)
    assert not any(bool(t) for _, _, t, _ in steps)
    assert [bool(tr) for _, _, _, tr in steps] == [False] * 199 + [True]
    assert all(-16.3 <= float(r) <= 0.0 for _, r, _, _ in steps)


def test_mountaincar_contract_and_goal_is_a_termination():
    env = make_env("MountainCarContinuous-v0")
    _contract(env, 2, 1)
    np.testing.assert_allclose(env.action_low, [-1.0])
    np.testing.assert_allclose(env.action_high, [1.0])
    # full throttle in the direction of the velocity swings up and reaches the goal within the time limit
    st, obs = env.reset(KEY)
    terminated = False
    for k in range(999):
        a = jnp.where(obs[1] >= 0, 1.0, -1.0).reshape(1)
        st, obs, r, term, trunc = jax.jit(env.step)(st, a, jax.random.key(k))
        if bool(term):
            terminated = True
            assert not bool(trunc) and float(r) > 90.0
            break
    assert terminated


def test_point_mass_contract():
    env = make_env("TestPointMass-v0")
    _contract(env, 4, 2)
    np.testing.assert_allclose(env.action_low, [-1.0, -2.0])
    steps = _roll(env, 30)
    assert [bool(tr) for _, _, _, tr in steps] == [False] * 29 + [True]


def test_unknown_env_lists_the_known_ids():
    with pytest.raises(KeyError, match="Pendulum-v1"):
        make_env("Nope-v0")
