"""``obs_norm`` is Gymnasium's NormalizeObservation: the stats see every valid observation (resets and steps)."""

import jax
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.policy.continuous.select_policy import select_policy


def test_obs_norm_counts_only_valid_observations(make_pendulum_mc):
    mc = make_pendulum_mc(obs_norm=True, env_name="TestPointMass-v0", env_num_envs=3)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    state, tc, m = env.execute_policy(state, jax.random.key(1), policy, 30, 40)
    episodes = int(np.asarray(m["episode_valid"]).sum())
    # one reset observation per episode plus one observation per step, starting from Gymnasium's count of 1e-4
    np.testing.assert_allclose(float(state.get("obs_rms").count), int(m["steps"]) + episodes + 1e-4, rtol=1e-6)
    # stored states are normalized: roughly zero mean, unit scale
    states = np.asarray(tc.trajectories().states)[np.asarray(tc.trajectories().mask)]
    assert np.all(np.abs(states.mean(axis=0)) < 1.0)


def test_without_obs_norm_the_stats_stay_untouched(make_pendulum_mc):
    mc = make_pendulum_mc(env_num_envs=1)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    state, _, _ = env.execute_policy(state, jax.random.key(1), policy, 200, 200)
    np.testing.assert_allclose(float(state.get("obs_rms").count), 1e-4)
