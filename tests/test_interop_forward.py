"""The JAX and MLX forwards match each other and the existing Gaussian modules."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
mx = pytest.importorskip("mlx.core")

from abstract_rl.interop.forward import log_prob_from_stats, mode_from_stats, sample_from_stats, suff_stats  # noqa: E402
from abstract_rl.interop.namespace import JAX, MLX  # noqa: E402
from abstract_rl.interop.spec import make_gaussian_spec, pack_module  # noqa: E402
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy  # noqa: E402
from abstract_rl.policy.continuous.select_policy import select_policy  # noqa: E402
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy  # noqa: E402


def _pair(make_pendulum_mc, env, **overrides):
    """A ``select_policy`` module, the spec of its flat layout, and states."""
    mc = make_pendulum_mc(env_name=env, policy_structure=[16, 16], **overrides)
    mx.random.seed(0)
    with mc["conf"].ns("policy"):
        module = select_policy(mc)
        conf = mc["conf"]
        kind = "univariate" if isinstance(module, UnivariateGaussianPolicy) else "diagonal"
        action_dim = 1 if kind == "univariate" else int(module.action_dim)
        spec = make_gaussian_spec(
            obs_dim=int(mc.env.observation_dim),
            action_dim=action_dim,
            kind=kind,
            cov_type=str(module.cov_type),
            structure=[int(v) for v in conf["structure"]],
            act_fn=list(conf["act_fn"]),
            layer_norm=bool(conf["layer_norm"]),
            squash=bool(module.squash),
            action_low=np.asarray(mc.env.action_space.low, dtype=np.float64),
            action_high=np.asarray(mc.env.action_space.high, dtype=np.float64),
        )
    flat, frozen = pack_module(module, spec)
    states_np = np.linspace(-0.5, 0.5, 8 * spec.obs_dim, dtype=np.float32).reshape(8, spec.obs_dim)
    return module, spec, flat, frozen, states_np


def _noise(spec, rows):
    rng = np.random.default_rng(1)
    if spec.kind == "univariate":
        return rng.normal(size=(rows, 3)).astype(np.float32)
    return rng.normal(size=(rows, 3, spec.action_dim)).astype(np.float32)


@pytest.mark.parametrize("cov", ["single", "fixed", "state"])
@pytest.mark.parametrize(
    "env, kind",
    [("Pendulum-v1", UnivariateGaussianPolicy), ("TestPointMass-v0", DiagonalGaussianPolicy)],
)
def test_shared_forward_matches_the_module_and_jax(make_pendulum_mc, cov, env, kind):
    """One definition: MLX module, MLX functional forward and JAX functional forward agree."""
    module, spec, flat, frozen, states_np = _pair(make_pendulum_mc, env, policy_cov_type=cov, policy_layer_norm=True)
    assert isinstance(module, kind)
    states = mx.array(states_np)
    theta = mx.array(flat)
    fixed = frozen if cov == "fixed" else None
    fixed_mx = None if fixed is None else mx.array(fixed)
    ref = np.array(module.forward(states))
    mlx_out = np.array(suff_stats(MLX, spec, theta, states, fixed_mx))
    fixed_j = None if fixed is None else jnp.asarray(fixed)
    jax_out = np.array(suff_stats(JAX, spec, jnp.asarray(flat), jnp.asarray(states_np), fixed_j))
    np.testing.assert_allclose(mlx_out, ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(jax_out, ref, atol=1e-5, rtol=1e-5)

    noise = _noise(spec, states_np.shape[0])
    ref_a = np.array(module.sample_actions(states=states, noise=mx.array(noise)))
    mlx_a = np.array(sample_from_stats(MLX, spec, mx.array(mlx_out), mx.array(noise)))
    jax_stats = jnp.asarray(jax_out)
    jax_a = np.array(sample_from_stats(JAX, spec, jax_stats, jnp.asarray(noise)))
    np.testing.assert_allclose(mlx_a, ref_a, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(jax_a, ref_a, atol=1e-5, rtol=1e-5)

    ref_lp = np.array(module.log_prob(mx.array(ref_a), suff_stats=mx.array(ref)))
    mlx_lp = np.array(log_prob_from_stats(MLX, spec, mx.array(mlx_out), mx.array(mlx_a)))
    # JAX log_prob sees the JAX sample, which matches, so it is compared to the module at that same action
    jax_lp = np.array(log_prob_from_stats(JAX, spec, jax_stats, jnp.asarray(ref_a)))
    np.testing.assert_allclose(mlx_lp, ref_lp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(jax_lp, ref_lp, atol=1e-4, rtol=1e-4)
    got_mode = np.array(mode_from_stats(MLX, spec, mx.array(ref)))
    ref_mode = np.array(module.mode(suff_stats=mx.array(ref)))
    np.testing.assert_allclose(got_mode, ref_mode, atol=1e-5)
