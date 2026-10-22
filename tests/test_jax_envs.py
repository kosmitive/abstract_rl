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


@pytest.mark.parametrize(
    "env_id, obs_dim, act_dim",
    [
        ("InvertedPendulum-v5", None, 1),
        ("InvertedDoublePendulum-v5", None, 1),
        ("HalfCheetah-v5", None, 6),
        ("AcrobotSwingup-v0", None, 1),
    ],
)
def test_mjx_contract(env_id, obs_dim, act_dim):
    env = make_env(env_id)
    _contract(env, env.observation_dim, act_dim)
    assert np.all(env.action_low < env.action_high)
    assert np.all(np.isinf(env.obs_high))


def test_inverted_pendulum_falls_and_terminates():
    env = make_env("InvertedPendulum-v5")
    st, _ = env.reset(KEY)
    step = jax.jit(env.step)
    for k in range(200):
        st, _obs, _r, term, trunc = step(st, jnp.full(1, env.action_high[0]), jax.random.key(k))
        if bool(term):
            break
    assert bool(term) and not bool(trunc)


def test_halfcheetah_never_terminates():
    env = make_env("HalfCheetah-v5")
    steps = _roll(env, 50, jnp.full(6, 0.5))
    assert not any(bool(t) or bool(tr) for _, _, t, tr in steps)
