"""The MJX pendulum environments against Gymnasium v5's MuJoCo environments, from the same state and actions."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.env.jax_envs import make_env

gym = pytest.importorskip("gymnasium")

PENDULUMS = ["InvertedPendulum-v5", "InvertedDoublePendulum-v5"]


def _actions(env, steps, seed):
    """A fixed action sequence: a slow sine for the first seed, uniform noise over the bounds otherwise."""
    if seed == 0:
        return 0.3 * env.action_high * np.sin(0.3 * np.arange(steps))[:, None]
    return np.random.default_rng(seed).uniform(env.action_low, env.action_high, (steps, env.action_dim))


@pytest.mark.parametrize("env_id", PENDULUMS)
@pytest.mark.parametrize("seed", [0, 1])
def test_same_state_and_actions_give_gymnasiums_trajectory(env_id, seed):
    env, ref = make_env(env_id), gym.make(env_id).unwrapped
    ref.reset(seed=seed)
    assert env.observation_dim == ref.observation_space.shape[0]
    np.testing.assert_allclose(env.action_low, ref.action_space.low)
    np.testing.assert_allclose(env.action_high, ref.action_space.high)
    assert env.frame_skip * env.mj_model.opt.timestep == pytest.approx(ref.dt)

    rng = np.random.default_rng(seed)
    qpos = rng.uniform(-0.05, 0.05, ref.model.nq)
    qvel = rng.uniform(-0.05, 0.05, ref.model.nv)
    ref.set_state(qpos, qvel)
    st = env.set_state(qpos, qvel)
    np.testing.assert_allclose(env.observe(st), ref._get_obs(), atol=1e-6)

    step, terminated = jax.jit(env.step), []
    for k, a in enumerate(_actions(env, 30, seed)):
        ref_obs, ref_r, ref_term, _, _ = ref.step(a)
        st, obs, r, term, trunc = step(st, jnp.asarray(a, jnp.float32), jax.random.key(k))
        # float32 MJX vs float64 MuJoCo, both RK4: the measured gap over these 30 steps is at most 3.4e-5
        np.testing.assert_allclose(obs, ref_obs, rtol=1e-4, atol=1e-4, err_msg=f"step {k}")
        assert float(r) == pytest.approx(ref_r, rel=1e-4, abs=1e-4), f"step {k}"
        assert bool(term) == ref_term and not bool(trunc), f"step {k}"
        terminated.append(ref_term)
    assert any(terminated)  # the sequence covers the terminating step and the steps after it


@pytest.mark.parametrize("env_id", PENDULUMS)
def test_ctrl_is_the_action(env_id):
    env = make_env(env_id)
    st, _ = env.reset(jax.random.key(0))
    for a in [env.action_low[0], -1.0, 0.5, env.action_high[0]]:
        nxt, *_ = jax.jit(env.step)(st, jnp.full(1, a, jnp.float32), jax.random.key(0))
        np.testing.assert_allclose(nxt.ctrl, [a])


@pytest.mark.parametrize("env_id", PENDULUMS)
def test_reset_noise_matches_gymnasium(env_id):
    env, ref = make_env(env_id), gym.make(env_id).unwrapped
    ref_states = []
    for s in range(400):
        ref.reset(seed=s)
        ref_states.append(np.concatenate([ref.data.qpos, ref.data.qvel]))
    st, _ = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 400))
    ours = np.concatenate([st.qpos, st.qvel], axis=1)
    ref_states = np.asarray(ref_states)
    noise = env.reset_noise_scale
    assert np.all(np.abs(ours[:, : ref.model.nq]) <= noise)
    # qvel is uniform in +-noise (inverted pendulum) or normal with std noise (double pendulum, 5 sigma bound)
    qvel_bound = noise if env_id == "InvertedPendulum-v5" else 5 * noise
    assert np.all(np.abs(ours[:, ref.model.nq :]) <= qvel_bound)
    assert np.all(np.abs(ref_states[:, ref.model.nq :]) <= qvel_bound)
    np.testing.assert_allclose(ours.std(0), ref_states.std(0), rtol=0.2)
    np.testing.assert_allclose(ours.mean(0), 0.0, atol=3 * noise / np.sqrt(400))


def test_pendulums_do_not_need_gymnasium(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)  # any `import gymnasium` now raises ImportError
    for env_id in PENDULUMS:
        env = make_env(env_id)
        jax.jit(env.reset)(jax.random.key(0))
