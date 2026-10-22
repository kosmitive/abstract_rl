from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.env.jax_envs import register_env
from abstract_rl.policy.continuous.select_policy import select_policy

KEY = jax.random.key(0)


class _CountState(NamedTuple):
    t: jax.Array


class _TerminatesAfterThree:
    """Terminates on the third step; a 1-D action space with symmetric bounds."""

    observation_dim, action_dim = 1, 1
    action_low, action_high = np.array([-1.0], np.float32), np.array([1.0], np.float32)
    obs_high = np.array([np.inf], np.float32)

    def reset(self, key):
        return _CountState(jnp.asarray(0, jnp.int32)), jnp.zeros(1)

    def step(self, st, action, key):
        st = _CountState(st.t + 1)
        return st, st.t.astype(jnp.float32).reshape(1), jnp.asarray(1.0), st.t >= 3, jnp.asarray(False)


register_env("TestTerminates-v0", _TerminatesAfterThree)


def _setup(make_pendulum_mc, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, KEY)
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    return env, policy, state


def test_truncation_is_not_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 200, policy)
    assert int(traj.mask.sum()) == 200  # Pendulum-v1 truncates after 200 steps
    assert not np.any(np.asarray(traj.dones)) and bool(traj.ends[199])


def test_max_steps_cut_is_not_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 10, policy)
    assert int(traj.mask.sum()) == 10 and not np.any(np.asarray(traj.dones)) and bool(traj.ends[9])


def test_termination_is_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_name="TestTerminates-v0", env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 50, policy)
    assert int(traj.mask.sum()) == 3
    np.testing.assert_allclose(np.asarray(traj.dones)[:3, 0], [0.0, 0.0, 1.0])


def test_rollout_collects_whole_episodes_until_batch_size(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_name="TestTerminates-v0", env_num_envs=4)
    _, tc, m = jax.jit(lambda s, k: env.execute_policy(s, k, policy, 50, 10))(state, KEY)
    t = tc.trajectories()
    steps = int(m["steps"])
    assert steps >= 10 and steps % 3 == 0  # whole 3-step episodes only
    assert steps < 10 + 4 * 3  # at most the running episodes of the other lanes on top
    assert int(np.asarray(m["episode_valid"]).sum()) == steps // 3
    # every lane ends its last valid step with an episode end
    for lane in range(4):
        valid = np.asarray(t.mask[lane])
        if valid.any():
            assert bool(t.ends[lane][np.nonzero(valid)[0][-1]])


def test_single_lane_matches_sequential_semantics(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, _, m = env.execute_policy(state, KEY, policy, 200, 450)
    assert int(m["steps"]) == 600 and int(np.asarray(m["episode_valid"]).sum()) == 3  # like torch: 3 whole episodes
