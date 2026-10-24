"""Opt-in wiring. The Gymnasium backend stays the one the existing runs use."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from abstract_rl.run.run_mpo_non_parametric import _make_env as mpo_env  # noqa: E402
from abstract_rl.run.run_trpo import _make_env as trpo_env  # noqa: E402
from abstract_rl.run.run_trpo import run_trpo  # noqa: E402


def test_the_default_backend_is_the_gymnasium_wrapper(make_pendulum_mc):
    """``env_rollout_backend`` defaults to gymnasium, and that constructs the unchanged wrapper."""
    from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper

    mc = make_pendulum_mc()
    assert mc["conf"].to_dict().get("env_rollout_backend", "gymnasium") == "gymnasium"
    with mc["conf"].ns("env"):
        env = trpo_env(mc, "gymnasium")
    assert isinstance(env, RolloutEnvWrapper)
    env.close()
    mc2 = make_pendulum_mc()
    with mc2["conf"].ns("env"):
        env2 = mpo_env(mc2, "gymnasium")
    assert isinstance(env2, RolloutEnvWrapper)
    env2.close()


def test_jax_backend_rejects_filters_and_halfcheetah(trpo_conf, tmp_path):
    """Unsupported combinations fail at construction instead of diverging quietly."""
    pytest.importorskip("jax")
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration

    mc = ModelConfiguration({**trpo_conf, "alg": "trpo", "env_name": "HalfCheetah-v5", "data_root": str(tmp_path)})
    with mc["conf"].ns("env"), pytest.raises(ValueError, match="HalfCheetah"):
        trpo_env(mc, "jax")

    mc = ModelConfiguration({**trpo_conf, "alg": "trpo", "obs_norm": True, "data_root": str(tmp_path / "b")})
    with mc["conf"].ns("env"), pytest.raises(RuntimeError, match="obs_norm"):
        trpo_env(mc, "jax")


def test_stack_field_keeps_mlx_arrays_on_mlx():
    """Training concatenates MLX trajectory fields without a NumPy round trip."""
    from abstract_rl.operator.trajectory_operator import stack_field

    class _Env:
        observation_dim = 2
        action_dim = 1

        def discount(self) -> float:
            return 0.99

    from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory

    def episode(scale: float) -> Trajectory:
        states = mx.arange(6, dtype=mx.float32).reshape(3, 2) * scale
        actions = mx.ones((2, 1))
        rewards = mx.ones((2, 1))
        dones = mx.zeros((2, 1))
        logp = mx.zeros((2, 1))
        return Trajectory.from_arrays(0, _Env(), states, actions, rewards, dones, logp, 2.0)

    stacked = stack_field([episode(1), episode(2)], "states", 6)
    assert isinstance(stacked, mx.array)
    assert stacked.shape == (6, 2)
    np.testing.assert_allclose(np.array(stacked[:2]), np.array(episode(1).states))


def test_checkpoint_roundtrip_loads_into_select_policy(make_pendulum_mc, tmp_path):
    """Saved names are the ones eval loads into ``select_policy``."""
    from abstract_rl.interop.policy import SharedGaussianPolicy
    from abstract_rl.policy.continuous.select_policy import select_policy

    mc = make_pendulum_mc(policy_cov_type="single", policy_structure=[8, 8])
    mx.random.seed(2)
    with mc["conf"].ns("policy"):
        shared = SharedGaussianPolicy(mc)
    path = tmp_path / "policy.safetensors"
    shared.save_weights(str(path))
    mx.random.seed(2)
    with mc["conf"].ns("policy"):
        restored = select_policy(mc)
    restored.load_weights(str(path))
    states = mx.zeros((4, mc.env.observation_dim))
    np.testing.assert_allclose(np.array(restored.forward(states)), np.array(shared.forward(states)), atol=1e-6)


def test_one_trpo_epoch_on_the_jax_rollout(trpo_conf, tmp_path):
    """TRPO trains the shared policy from a sharded-of-one JAX rollout and writes a loadable checkpoint."""
    pytest.importorskip("jax")
    pytest.importorskip("gymnax")
    conf = {
        **trpo_conf,
        "num_epochs": 1,
        "policy_batch_size": 48,
        "max_steps": 12,
        "val_opt_steps": 1,
        "policy_structure": [16, 16],
        "val_struct": [16, 16],
        "env_rollout_backend": "jax",
        "env_num_envs": 4,
        "env_jax_devices": 1,
        "env_jax_dtype": "float32",
        "data_root": str(tmp_path),
        "trpo_cg_max_k": 2,
    }
    returns = run_trpo(conf)
    assert returns.shape[0] >= 1
    assert np.isfinite(returns[0, 0])
