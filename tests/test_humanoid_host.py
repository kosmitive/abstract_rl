"""Gymnasium Humanoid-v5 is stepped on the host; the MPO learner stays the vmapped JAX update."""

import gymnasium as gym
import jax
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.env.host_gym import HostGymEnv
from abstract_rl.env.jax_envs import JaxEnv, make_env
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts.hydra_conf import load, run_dict


def test_humanoid_matches_gymnasium_and_is_not_a_jax_env():
    env = make_env("Humanoid-v5")
    assert isinstance(env, HostGymEnv)
    assert not isinstance(env, JaxEnv)
    ref = gym.make("Humanoid-v5")
    try:
        obs = env.reset_obs(0)
        ref_obs, _ = ref.reset(seed=0)
        np.testing.assert_allclose(obs, np.asarray(ref_obs, np.float32).reshape(-1))
        action = np.zeros(env.action_dim, np.float32)
        nxt, reward, terminated, truncated = env.step_obs(action)
        ref_nxt, ref_reward, ref_terminated, ref_truncated, _ = ref.step(action)
        np.testing.assert_allclose(nxt, np.asarray(ref_nxt, np.float32).reshape(-1))
        assert reward == float(ref_reward)
        assert terminated is bool(ref_terminated) and truncated is bool(ref_truncated)
        assert env.observation_dim == int(np.prod(ref.observation_space.shape))
        np.testing.assert_allclose(env.action_low, np.asarray(ref.action_space.low, np.float32).reshape(-1))
        np.testing.assert_allclose(env.action_high, np.asarray(ref.action_space.high, np.float32).reshape(-1))
    finally:
        env.close()
        ref.close()


def test_jax_envs_stay_on_the_jitted_rollout(make_pendulum_mc):
    mc = make_pendulum_mc()
    assert mc.env.uses_host is False


def _tiny_conf(tmp_path, overrides: list[str]) -> dict:
    conf = run_dict(load(["env=humanoid", "alg=mpo", *overrides]))
    conf["data_root"] = str(tmp_path)
    return conf


def test_host_rollout_collects_whole_episodes(tmp_path):
    conf = _tiny_conf(
        tmp_path,
        [
            "alg.policy_structure=[32]",
            "alg.policy_act_fn=[elu,none]",
            "alg.max_steps=6",
            "alg.env_sample=6",
            "alg.obs_norm=true",
        ],
    )
    mc = ModelConfiguration(conf)
    with mc["conf"].ns("env"):
        env = RolloutEnvWrapper(mc)
        mc.add_environment(env)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    state = ModelState.create({**env.init_state(), "key": jax.random.key(1)}, {"policy": (policy, policy)})
    try:
        state, tc, metrics = env.execute_policy_host(
            state, jax.random.key(2), policy, 6, 6, exploration=True, num_envs=1
        )
        mask = np.asarray(tc.trajectories().mask)
        assert int(metrics["steps"]) == int(mask.sum()) >= 6
        assert np.isfinite(np.asarray(tc.trajectories().raw_rewards)[mask]).all()
        assert np.asarray(tc.trajectories().actions).shape[-1] == env.action_dim
        # normalisation moved the stored state off the raw observation
        assert not np.allclose(np.asarray(tc.trajectories().states)[mask], 0.0)
        assert float(state.get("obs_rms").count) > 1
    finally:
        env.close()


def test_trpo_rejects_the_host_env(tmp_path):
    conf = run_dict(load(["env=humanoid", "alg=trpo", "alg.num_epochs=1"]))
    conf["data_root"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="MPO"):
        run_trpo(conf)


def test_one_vmapped_host_epoch(tmp_path):
    conf = run_dict(
        load(
            [
                "env=humanoid",
                "alg=mpo",
                "alg.num_epochs=1",
                "alg.env_sample=8",
                "alg.max_steps=8",
                "alg.tc_size=64",
                "alg.policy_opt_steps=1",
                "alg.val_opt_steps=1",
                "alg.rnk_max_steps=2",
                "alg.policy_structure=[32]",
                "alg.policy_act_fn=[elu,none]",
                "alg.val_struct=[32]",
                "alg.val_act_fn=[elu,none]",
                "alg.mpo_lm_bound=[-18.0,18.0]",
                "seeds=[1,2]",
            ]
        )
    )
    conf["data_root"] = str(tmp_path)
    returns = run_mpo_non_parametric(conf)
    assert returns.shape == (2, 1, 2)
    assert np.isfinite(returns).all()
