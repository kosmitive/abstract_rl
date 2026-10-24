"""Sharded rollout: one device, several devices, and the arena the trajectories view."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mx = pytest.importorskip("mlx.core")

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory  # noqa: E402
from abstract_rl.interop.envs import make_jax_env  # noqa: E402
from abstract_rl.interop.policy import SharedGaussianPolicy  # noqa: E402
from abstract_rl.interop.scan import sharded_rollout  # noqa: E402
from abstract_rl.interop.shared import buffer_address  # noqa: E402


def _policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_structure=[8, 8], policy_batch_size=32)
    mx.random.seed(0)
    with mc["conf"].ns("policy"):
        return SharedGaussianPolicy(mc)


def _keys(n, seed=0):
    master = jax.random.key(seed) if hasattr(jax.random, "key") else jax.random.PRNGKey(seed)
    return jax.random.split(master, n)


def test_single_device_rollout_fills_an_mlx_arena(make_pendulum_mc):
    """The scan result is a view of the arena, and the parameters were not copied onto the device."""
    policy = _policy(make_pendulum_mc)
    shared = policy.export_for_rollout()
    env = make_jax_env("Pendulum-v1")
    result = sharded_rollout(
        env,
        policy.spec,
        shared,
        policy.fixed_numpy(),
        _keys(4),
        horizon=6,
        n_devices=1,
        deterministic=True,
        normalize=True,
        obs_scale=np.array([1, 1, 8], np.float32),
        reward_scale=0.1,
        reward_offset=0.0,
        n_keep=4,
    )
    assert result.params_are_views
    assert result.lengths.shape == (4,)
    assert np.all(result.lengths == 6)
    episode = result.episode(0)
    assert episode["states"].shape == (7, 3)
    assert episode["actions"].shape == (6, 1)
    # the episode state is a prefix of the arena row, same allocation
    assert buffer_address(episode["states"]) == buffer_address(result.states)
    traj = Trajectory.from_arrays(
        0,
        type("E", (), {"observation_dim": 3, "action_dim": 1, "discount": lambda self: 0.99})(),
        episode["states"],
        episode["actions"],
        episode["rewards"][:, None],
        episode["terminated"][:, None],
        episode["log_likelihoods"][:, None],
        episode["raw_return"],
    )
    assert len(traj) == 6
    assert np.isfinite(traj.raw_return)


def test_sharded_rollout_matches_one_device():
    """Several host devices match one device. JAX must be imported after the device-count flag."""
    script = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        import numpy as np
        import mlx.core as mx
        import jax
        import jax.numpy as jnp
        from abstract_rl.interop.envs import make_jax_env
        from abstract_rl.interop.forward import suff_stats
        from abstract_rl.interop.namespace import JAX
        from abstract_rl.interop.scan import sharded_rollout
        from abstract_rl.interop.spec import make_gaussian_spec
        from abstract_rl.interop.shared import SharedParams

        spec = make_gaussian_spec(
            obs_dim=3, action_dim=1, kind="univariate", cov_type="single",
            structure=[8], act_fn=["tanh", "none"], layer_norm=False, squash=True,
            action_low=np.array([-2.0]), action_high=np.array([2.0]),
        )
        shared = SharedParams(spec.n_params)
        rng = np.random.default_rng(0)
        shared.write_numpy(rng.normal(size=spec.n_params).astype(np.float32) * 0.1)
        env = make_jax_env("Pendulum-v1")
        master = jax.random.key(0)
        keys = jax.random.split(master, 8)
        kwargs = dict(
            deterministic=False, normalize=False, obs_scale=np.ones(3, np.float32),
            reward_scale=1.0, reward_offset=0.0, n_keep=8,
        )
        one = sharded_rollout(env, spec, shared, np.zeros(0, np.float32), keys, 5, 1, **kwargs)
        many = sharded_rollout(env, spec, shared, np.zeros(0, np.float32), keys, 5, 4, **kwargs)
        assert one.params_are_views and many.params_are_views, (one.params_are_views, many.params_are_views)
        a = np.array(one.states)
        b = np.array(many.states)
        err = np.max(np.abs(a - b))
        print("max_abs", err)
        if err > 1e-5:
            raise SystemExit(f"sharded rollout differs by {err}")
        # the forward used inside the scan is the shared one, evaluated here so the import stays honest
        _ = suff_stats(JAX, spec, jnp.asarray(shared.numpy_view()), jnp.zeros((1, 3), np.float32))
        """
    )
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = "src"
    # drop a count inherited from the parent so the child sets it before importing JAX
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)


def test_importing_the_package_does_not_import_jax():
    """CI installs the package without JAX. Importing the rollout must not require it."""
    script = (
        "import sys; import abstract_rl.interop.rollout; import abstract_rl.interop.scan; "
        "assert 'jax' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
