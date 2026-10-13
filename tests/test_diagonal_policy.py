import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.policy.continuous.diagonal_gaussian_policy import (
    DiagonalGaussianPolicy,
    head_activations,
    sigma_is_softplus,
)
from abstract_rl.policy.continuous.gaussian_math import kl_divergence
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

LOW, HIGH = np.array([-1.0, -2.0]), np.array([1.0, 2.0])


class _Batch:
    def __init__(self, states):
        self.states = states


def _policy(make_pendulum_mc, env="TestPointMass-v0", cls=DiagonalGaussianPolicy, **overrides):
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        return cls(mc)


def _force_head(policy, mu_pre, sigma_pre):
    """Make the last layer output ``[mu_pre, sigma_pre]`` independent of the state."""
    layers = list(policy.network.layers)
    layer = layers[-1]
    layer = eqx.tree_at(lambda lin: lin.weight, layer, jnp.zeros_like(layer.weight))
    layer = eqx.tree_at(lambda lin: lin.bias, layer, jnp.asarray([*mu_pre, *sigma_pre], dtype=jnp.float32))
    layers[-1] = layer
    policy.network = eqx.tree_at(lambda net: net.layers, policy.network, tuple(layers))


def _normal_log_prob(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def test_select_policy_uses_the_diagonal_policy_for_multi_dimensional_actions(make_pendulum_mc):
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, "Pendulum-v1", select_policy), UnivariateGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_type="diagonal"), DiagonalGaussianPolicy)


def test_shapes(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state")
    states = np.random.default_rng(0).normal(size=(7, 4)).astype(np.float32)
    stats = policy.forward(states)
    assert stats.shape == (7, 4) == (7, policy.get_suff_stats_dim())
    actions = policy.sample_actions(suff_stats=stats, num_actions=5)
    assert actions.shape == (7, 5, 2)
    assert policy.log_prob(actions, suff_stats=stats).shape == (7, 5)
    assert policy.mode(suff_stats=stats).shape == (7, 2)
    assert policy.log_prob(policy.mode(suff_stats=stats), suff_stats=stats).shape == (7, 1)


def test_head_activations_are_expanded_per_output():
    act, sigma_acts = head_activations(["tanh", "tanh", ["none", "softplus"]], 3)
    assert act[-1] == ["none"] * 3 + ["softplus"] * 3
    assert sigma_acts == ["softplus"] * 3
    assert sigma_is_softplus(sigma_acts)
    assert head_activations(["tanh", "tanh", ["none", "softplus"]], 1)[0][-1] == ["none", "softplus"]
    assert not sigma_is_softplus(head_activations(["tanh", "tanh", "none"], 2)[1])
    with pytest.raises(ValueError):
        head_activations(["tanh", "tanh", ["none", "softplus", "none"]], 2)
    with pytest.raises(ValueError):
        sigma_is_softplus(["softplus", "none"])


def test_state_dependent_sigma_default_activation_applies_one_softplus(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", "none"])
    states = np.zeros((3, 4), dtype=np.float32)
    _force_head(policy, [0.0, 0.0], [-20.0, -20.0])
    sigma = np.asarray(policy.forward(states)[:, 2:])
    assert np.all(sigma > 0) and np.all(sigma < 1e-3)
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))]) + 1e-6
    assert np.allclose(np.asarray(policy.forward(states)[0, 2:]), expected, atol=1e-5)


def test_state_dependent_sigma_with_softplus_activation_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", ["none", "softplus"]])
    states = np.zeros((3, 4), dtype=np.float32)
    _force_head(policy, [0.0, 0.0], [-20.0, -1000.0])
    sigma = np.asarray(policy.forward(states)[:, 2:])
    assert np.all(sigma > 0) and np.all(sigma < 1e-3)
    policy.log_prob(policy.sample_actions(states=states), states=states)
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    assert np.allclose(np.asarray(policy.forward(states)[0, 2:]), expected, atol=1e-5)


def test_mean_is_not_activated_twice(make_pendulum_mc):
    policy = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_squash=True, policy_act_fn=["tanh", "tanh", "none"]
    )
    mu_pre = [0.5, -1.5]
    _force_head(policy, mu_pre, [])
    states = np.zeros((2, 4), dtype=np.float32)
    assert np.allclose(np.asarray(policy.forward(states)[0, :2]), np.asarray(mu_pre), atol=1e-6)
    center, scale = (HIGH + LOW) / 2, (HIGH - LOW) / 2
    expected = center + scale * np.tanh(mu_pre)
    assert np.allclose(np.asarray(policy.mode(states=states)[0]), expected, atol=1e-6)


def test_squashed_samples_lie_in_the_per_dimension_bounds(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True, policy_sigma=5.0)
    _force_head(policy, [3.0, -3.0], [])
    actions = policy.sample_actions(states=np.zeros((4, 4), dtype=np.float32), num_actions=2000)
    actions = np.asarray(actions)
    assert np.all(actions[..., 0] <= 1) and np.all(actions[..., 0] >= -1)
    assert np.all(actions[..., 1] <= 2) and np.all(actions[..., 1] >= -2)
    assert np.isfinite(np.asarray(policy.log_prob(actions, states=np.zeros((4, 4), dtype=np.float32)))).all()
    assert actions[..., 0].mean() > 0.3 and actions[..., 1].mean() < -0.6


@pytest.mark.parametrize("squash", [False, True])
def test_log_prob_is_a_density(make_pendulum_mc, squash):
    policy = _policy(make_pendulum_mc, policy_squash=squash, policy_cov_type="single", policy_sigma=0.5)
    stats = np.array([[0.3, -0.2, 0.5, 0.5]], dtype=np.float32)
    if squash:
        lo, hi, n = LOW + 1e-4, HIGH - 1e-4, 801
    else:
        lo, hi, n = np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 801
    g0, g1 = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    xx, yy = np.meshgrid(g0, g1, indexing="ij")
    grid = np.stack([xx, yy], axis=-1).reshape(1, -1, 2)
    density = np.exp(np.asarray(policy.log_prob(grid, suff_stats=stats))).reshape(n, n)
    integral = np.trapezoid(np.trapezoid(density, g1, axis=1), g0)
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_squashed_log_prob_matches_the_change_of_variables(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True)
    mu = np.array([[0.3, -0.7]], dtype=np.float32)
    sigma = np.array([[0.8, 1.1]], dtype=np.float32)
    u = np.array([[[0.2, -0.4]]], dtype=np.float32)
    scale = ((HIGH - LOW) / 2).astype(np.float32)
    center = ((HIGH + LOW) / 2).astype(np.float32)
    action = center + scale * np.tanh(u)
    expected = _normal_log_prob(u[0, 0], mu[0], sigma[0]).sum() - (np.log(scale) + np.log1p(-(np.tanh(u) ** 2))).sum()
    got = float(np.asarray(policy.log_prob(action, suff_stats=np.concatenate([mu, sigma], axis=1))).sum())
    assert np.isclose(got, expected, atol=1e-4)


def _perturb(module, key, scale=0.3):
    diff, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(diff)
    updated = []
    for leaf in leaves:
        key, sub = jax.random.split(key)
        updated.append(leaf + scale * jax.random.normal(sub, leaf.shape, dtype=leaf.dtype))
    return eqx.combine(jax.tree.unflatten(treedef, updated), static)


def test_kl_divergence_matches_the_per_dimension_sum_and_monte_carlo(make_pendulum_mc):
    mc = make_pendulum_mc(env_name="TestPointMass-v0", policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
        mc.add_syncable("policy", policy, 1)
    states = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    batch = _Batch(states)
    policy.params = _perturb(policy.params, jax.random.PRNGKey(0))
    stats = np.asarray(policy.forward(states))
    target = np.asarray(mc.get("policy", True).forward(states))
    mu, sigma, t_mu, t_sigma = stats[:, :2], stats[:, 2:], target[:, :2], target[:, 2:]

    def closed(left_mu, left_sigma, right_mu, right_sigma):
        ratio = (left_sigma**2 + (left_mu - right_mu) ** 2) / (2.0 * right_sigma**2)
        return (np.log(right_sigma / left_sigma) + ratio - 0.5).sum(axis=1).mean()

    kl = float(policy.kl_divergence(batch, mode="i-projection", est="mean"))
    expected = closed(mu, sigma, t_mu, t_sigma)
    assert np.isclose(kl, expected, atol=1e-5)
    assert np.isclose(float(policy.kl_divergence(batch, est="sum")), expected * 3, atol=1e-4)
    reverse = float(policy.kl_divergence(batch, mode="m-projection", est="mean"))
    assert np.isclose(reverse, closed(t_mu, t_sigma, mu, sigma), atol=1e-5)

    drawn = policy.sample_actions(suff_stats=stats, num_actions=20_000, key=jax.random.PRNGKey(1))
    current_lp = policy.log_prob(drawn, suff_stats=stats)
    target_lp = policy.log_prob(drawn, suff_stats=target)
    mc_kl = float(np.asarray(current_lp - target_lp).mean())
    assert mc_kl == pytest.approx(expected, rel=0.15, abs=0.05)

    target_params = mc.get("policy", True).params

    def loss(params):
        return kl_divergence(params, target_params, jnp.asarray(states), mode_name="i-projection", est="mean")

    grads = eqx.filter_grad(loss)(policy.params)
    flat = [g.reshape(-1) for g in jax.tree.leaves(grads) if eqx.is_array(g)]
    assert float(jnp.sum(jnp.abs(jnp.concatenate(flat)))) > 0


def test_one_dimensional_diagonal_policy_equals_the_univariate_policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        diag, uni = DiagonalGaussianPolicy(mc), UnivariateGaussianPolicy(mc)
    uni.network = jax.tree.map(lambda value: value, diag.network)
    states = np.random.default_rng(0).normal(size=(6, 3)).astype(np.float32)
    sd, su = diag.forward(states), uni.forward(states)
    assert np.allclose(np.asarray(sd), np.asarray(su), atol=1e-6)
    actions = uni.sample_actions(suff_stats=su, num_actions=4, key=jax.random.PRNGKey(2))
    diag_lp = diag.log_prob(np.asarray(actions)[..., None], suff_stats=sd)
    uni_lp = uni.log_prob(actions, suff_stats=su)
    assert np.allclose(np.asarray(diag_lp), np.asarray(uni_lp), atol=1e-5)
    assert np.allclose(np.asarray(diag.mode(suff_stats=sd)), np.asarray(uni.mode(suff_stats=su)), atol=1e-5)


def test_entropy_of_the_unsquashed_policy_and_refusal_for_squashed(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=False, policy_cov_type="single", policy_sigma=0.7)
    expected = 2 * 0.5 * math.log(2 * math.pi * math.e * 0.7**2)
    assert float(policy.entropy(_Batch(np.zeros((3, 4), np.float32)))) == pytest.approx(expected, abs=1e-5)
    with pytest.raises(NotImplementedError):
        _policy(make_pendulum_mc, policy_squash=True).entropy(_Batch(np.zeros((3, 4), np.float32)))
