"""Initial standard deviation, std floor, and mean-head scale of the Gaussian policies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.misc.flat_params import l2_norm, ravel_module, weight_spec
from abstract_rl.policy.continuous.continuous_policy import (
    STD_HEAD_INIT_SCALE,
    build_gaussian_params,
    refresh_sigma_floors,
)
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.full_covariance_gaussian_policy import FullCovarianceGaussianPolicy
from abstract_rl.policy.continuous.gaussian_math import scheduled_sigma_floor, softplus_inverse, unpack_lower
from abstract_rl.scripts.hydra_conf import make_config

_POINT = "TestPointMass-v0"


def _params(make_pendulum_mc, key, **overrides):
    mc = make_pendulum_mc(env_name=_POINT, **overrides)
    with mc["conf"].ns("policy"):
        return build_gaussian_params(mc, key, univariate=False)


def _policy(make_pendulum_mc, cls=DiagonalGaussianPolicy, **overrides):
    mc = make_pendulum_mc(env_name=_POINT, **overrides)
    with mc["conf"].ns("policy"):
        policy = cls(mc)
        mc.add_syncable("policy", policy, 1)
        return policy


def test_scheduled_sigma_floor_is_linear_then_zero():
    assert scheduled_sigma_floor(0.3, 100, 0) == pytest.approx(0.3)
    assert scheduled_sigma_floor(0.3, 100, 50) == pytest.approx(0.15)
    assert scheduled_sigma_floor(0.3, 100, 100) == pytest.approx(0.0)
    assert scheduled_sigma_floor(0.3, 100, 150) == pytest.approx(0.0)
    assert scheduled_sigma_floor(0.3, 0, 50) == pytest.approx(0.3)


def test_init_std_sets_the_diagonal_bias_and_scales_the_std_head(make_pendulum_mc):
    key = jax.random.PRNGKey(0)
    init_std = 0.7
    base = _params(
        make_pendulum_mc,
        key,
        policy_cov_type="state",
        policy_mean_init_scale=1.0,
        policy_min_std=0.0,
        policy_init_std=None,
    )
    tuned = _params(
        make_pendulum_mc,
        key,
        policy_cov_type="state",
        policy_mean_init_scale=1.0,
        policy_init_std=init_std,
        policy_min_std=0.0,
    )
    for left, right in zip(base.network.layers[:-1], tuned.network.layers[:-1], strict=True):
        np.testing.assert_array_equal(np.asarray(left.weight), np.asarray(right.weight))
        np.testing.assert_array_equal(np.asarray(left.bias), np.asarray(right.bias))
    base_head, tuned_head = base.network.layers[-1], tuned.network.layers[-1]
    np.testing.assert_array_equal(np.asarray(base_head.weight[:2]), np.asarray(tuned_head.weight[:2]))
    np.testing.assert_allclose(
        np.asarray(tuned_head.weight[2:]), STD_HEAD_INIT_SCALE * np.asarray(base_head.weight[2:])
    )
    np.testing.assert_allclose(np.asarray(tuned_head.bias[2:]), softplus_inverse(init_std), rtol=0, atol=1e-6)
    states = np.zeros((4, 4), dtype=np.float32)
    sigma = np.asarray(tuned.network.forward(states)[:, 2:])
    # Zero hidden state: the std head outputs its bias, and softplus(bias) is init_std.
    np.testing.assert_allclose(jax.nn.softplus(sigma), init_std, rtol=0, atol=1e-5)
    random = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
    spread = np.asarray(jax.nn.softplus(tuned.network.forward(random)[:, 2:]))
    assert np.max(np.abs(spread - init_std)) < 0.05


def test_init_std_on_a_cholesky_head_is_diagonal_and_near_constant(make_pendulum_mc):
    key = jax.random.PRNGKey(1)
    init_std = 0.5
    base = _params(make_pendulum_mc, key, policy_cov_type="full", policy_mean_init_scale=1.0, policy_init_std=None)
    tuned = _params(make_pendulum_mc, key, policy_cov_type="full", policy_mean_init_scale=1.0, policy_init_std=init_std)
    base_head, tuned_head = base.network.layers[-1], tuned.network.layers[-1]
    expected = np.float32(STD_HEAD_INIT_SCALE) * np.asarray(base_head.weight[2:], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(tuned_head.weight[2:]), expected, rtol=1e-6, atol=0)
    # Packed layout for two actions is [L00, L10, L11]. Only the diagonal bias is init_std.
    bias = np.asarray(tuned_head.bias)
    np.testing.assert_allclose(bias[[2, 4]], softplus_inverse(init_std), rtol=0, atol=1e-6)
    assert bias[3] == pytest.approx(0.0)
    policy = _policy(
        make_pendulum_mc,
        FullCovarianceGaussianPolicy,
        policy_cov_type="full",
        policy_init_std=init_std,
        policy_min_std=0.0,
        policy_mean_init_scale=1.0,
    )
    zero = np.zeros((1, 4), dtype=np.float32)
    factor = np.asarray(unpack_lower(policy.forward(zero)[:, 2:], 2))
    np.testing.assert_allclose(factor[0].diagonal(), init_std, rtol=0, atol=1e-5)
    assert factor[0, 1, 0] == pytest.approx(0.0, abs=1e-6)
    random = np.random.default_rng(1).normal(size=(32, 4)).astype(np.float32)
    factors = np.asarray(unpack_lower(policy.forward(random)[:, 2:], 2))
    assert np.max(np.abs(factors.diagonal(axis1=-2, axis2=-1) - init_std)) < 0.05
    assert np.max(np.abs(factors[:, 1, 0])) < 0.05


def test_init_std_replaces_policy_sigma_for_a_state_independent_std(make_pendulum_mc):
    untouched = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_sigma=2.0, policy_min_std=0.0, policy_init_std=None
    )
    replaced = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_sigma=2.0, policy_init_std=0.4, policy_min_std=0.0
    )
    states = np.zeros((2, 4), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(untouched.forward(states)[:, 2:]), 2.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(replaced.forward(states)[:, 2:]), 0.4, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="policy_init_std"):
        _policy(make_pendulum_mc, policy_init_std=0.0)


def test_min_std_floors_the_diagonal_and_is_not_a_weight(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_min_std=0.3, policy_min_std_anneal_steps=0)
    layers = list(policy.network.layers)
    layer = layers[-1]
    layer = eqx.tree_at(lambda lin: lin.weight, layer, jnp.zeros_like(layer.weight))
    layer = eqx.tree_at(lambda lin: lin.bias, layer, jnp.asarray([0.0, 0.0, -40.0, -40.0], dtype=jnp.float32))
    layers[-1] = layer
    policy.network = eqx.tree_at(lambda net: net.layers, policy.network, tuple(layers))
    states = np.zeros((3, 4), dtype=np.float32)
    sigma = np.asarray(policy.forward(states)[:, 2:])
    np.testing.assert_allclose(sigma, 0.3, rtol=0, atol=1e-5)
    flat, _ = ravel_module(policy.params)
    penalty = np.asarray(l2_norm(policy.params))
    policy.params = eqx.tree_at(lambda params: params.sigma_floor, policy.params, jnp.asarray(0.9, dtype=jnp.float32))
    flat_after, _ = ravel_module(policy.params)
    np.testing.assert_array_equal(np.asarray(flat), np.asarray(flat_after))
    np.testing.assert_array_equal(penalty, np.asarray(l2_norm(policy.params)))
    assert not np.asarray(weight_spec(policy.params).sigma_floor)
    np.testing.assert_allclose(np.asarray(policy.forward(states)[:, 2:]), 0.9, rtol=0, atol=1e-5)

    wide = _policy(
        make_pendulum_mc, policy_cov_type="fixed", policy_sigma=2.0, policy_min_std=0.3, policy_init_std=None
    )
    np.testing.assert_allclose(np.asarray(wide.forward(states)[:, 2:]), 2.0, rtol=0, atol=1e-6)
    narrow = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_sigma=0.05, policy_min_std=0.3, policy_init_std=None
    )
    np.testing.assert_allclose(np.asarray(narrow.forward(states)[:, 2:]), 0.3, rtol=0, atol=1e-6)

    annealed = _policy(
        make_pendulum_mc,
        policy_cov_type="state",
        policy_min_std=0.3,
        policy_min_std_anneal_steps=160_000,
    )
    refresh_sigma_floors(annealed, 0)
    assert float(annealed.params.sigma_floor) == pytest.approx(0.3)
    assert float(annealed.mc.get("policy", True).params.sigma_floor) == pytest.approx(0.3)
    refresh_sigma_floors(annealed, 80_000)
    assert float(annealed.params.sigma_floor) == pytest.approx(0.15)
    assert float(annealed.mc.get("policy", True).params.sigma_floor) == pytest.approx(0.15)
    refresh_sigma_floors(annealed, 160_000)
    assert float(annealed.params.sigma_floor) == pytest.approx(0.0)
    mode, log_np = annealed.act(states, deterministic=True)
    log_jax = np.asarray(annealed.log_prob(mode, states=states))[:, 0]
    np.testing.assert_allclose(log_np, log_jax, rtol=1e-4, atol=1e-4)


def test_mean_init_scale_multiplies_only_the_final_mean_rows(make_pendulum_mc):
    key = jax.random.PRNGKey(2)
    plain = _params(make_pendulum_mc, key, policy_cov_type="state", policy_mean_init_scale=1.0, policy_init_std=None)
    scaled = _params(make_pendulum_mc, key, policy_cov_type="state", policy_mean_init_scale=0.01, policy_init_std=None)
    for left, right in zip(plain.network.layers[:-1], scaled.network.layers[:-1], strict=True):
        np.testing.assert_array_equal(np.asarray(left.weight), np.asarray(right.weight))
        np.testing.assert_array_equal(np.asarray(left.bias), np.asarray(right.bias))
    plain_head, scaled_head = plain.network.layers[-1], scaled.network.layers[-1]
    np.testing.assert_allclose(np.asarray(scaled_head.weight[:2]), 0.01 * np.asarray(plain_head.weight[:2]))
    np.testing.assert_array_equal(np.asarray(scaled_head.weight[2:]), np.asarray(plain_head.weight[2:]))
    np.testing.assert_array_equal(np.asarray(scaled_head.bias), 0.0)

    single = _params(make_pendulum_mc, key, policy_cov_type="single", policy_mean_init_scale=0.01, policy_init_std=None)
    single_plain = _params(
        make_pendulum_mc, key, policy_cov_type="single", policy_mean_init_scale=1.0, policy_init_std=None
    )
    np.testing.assert_allclose(
        np.asarray(single.network.layers[-1].weight),
        0.01 * np.asarray(single_plain.network.layers[-1].weight),
    )
    np.testing.assert_array_equal(np.asarray(single.network.layers[-1].bias), 0.0)


def test_mean_init_scale_defaults_and_mountaincar_floor():
    trpo = make_config("pendulum", "trpo")
    mpo = make_config("pendulum", "mpo")
    assert trpo["policy_mean_init_scale"] == 0.01 and type(trpo["policy_mean_init_scale"]) is float
    assert mpo["policy_mean_init_scale"] == 1.0 and type(mpo["policy_mean_init_scale"]) is float
    for conf in (trpo, mpo):
        assert conf["policy_init_std"] == 0.5 and type(conf["policy_init_std"]) is float
        assert conf["policy_min_std"] == 0.0 and type(conf["policy_min_std"]) is float
        assert conf["policy_min_std_anneal_steps"] == 0 and type(conf["policy_min_std_anneal_steps"]) is int
    for alg in ("trpo", "mpo"):
        mountain = make_config("mountaincar", alg)
        assert mountain["policy_min_std"] == 0.3
        assert mountain["policy_min_std_anneal_steps"] == 160000
        assert type(mountain["policy_min_std_anneal_steps"]) is int
        assert mountain["policy_init_std"] == 0.5
        assert mountain["policy_sigma"] == 2.0
    mountain_mpo = make_config("mountaincar", "mpo")
    assert mountain_mpo["policy_cov_type"] == "fixed"
    assert mountain_mpo["policy_type"] == "auto"
    assert make_config("halfcheetah", "mpo")["policy_sigma"] == 0.5
    assert make_config("halfcheetah", "mpo")["policy_init_std"] == 0.5
    assert make_config("halfcheetah", "trpo")["policy_min_std"] == 0.0
