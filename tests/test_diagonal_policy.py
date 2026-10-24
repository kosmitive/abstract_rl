import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from abstract_rl.policy.continuous.diagonal_gaussian_policy import (
    DiagonalGaussianPolicy,
    head_activations,
    sigma_is_softplus,
)
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
    layer = policy.network.layers[-1]
    layer.weight = mx.zeros_like(layer.weight)
    layer.bias = mx.array([*mu_pre, *sigma_pre], dtype=mx.float32)


def _normal_kl(p_mu, p_sigma, q_mu, q_sigma):
    """KL(N(p) || N(q)) elementwise, in numpy."""
    return np.log(q_sigma / p_sigma) + (p_sigma**2 + (p_mu - q_mu) ** 2) / (2 * q_sigma**2) - 0.5


def _normal_log_prob(x, mu, sigma):
    return -((x - mu) ** 2) / (2 * sigma**2) - np.log(sigma) - 0.5 * math.log(2 * math.pi)


def _perturb(policy, scale, seed):
    keys = iter(mx.random.split(mx.random.key(seed), 64))
    policy.update(tree_map(lambda p: p + scale * mx.random.normal(p.shape, key=next(keys)), policy.parameters()))


def test_select_policy_uses_the_diagonal_policy_for_multi_dimensional_actions(make_pendulum_mc):
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, "Pendulum-v1", select_policy), UnivariateGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_type="diagonal"), DiagonalGaussianPolicy)


@pytest.mark.parametrize("env", ["TestPointMass-v0", "Pendulum-v1"])
def test_full_covariance_is_refused(make_pendulum_mc, env):
    # MLX has no vjp for the Cholesky factorization, the full covariance is not ported
    with pytest.raises(NotImplementedError, match="full"):
        _policy(make_pendulum_mc, env, select_policy, policy_cov_type="full")


def test_shapes(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state")
    states = mx.random.normal((7, 4), key=mx.random.key(0))
    stats = policy.forward(states)
    assert stats.shape == (7, 4) == (7, policy.get_suff_stats_dim())
    actions = policy.sample_actions(suff_stats=stats, num_actions=5)
    assert actions.shape == (7, 5, 2)
    assert policy.log_prob(actions, suff_stats=stats).shape == (7, 5)
    assert policy.mode(suff_stats=stats).shape == (7, 2)
    # one action per state, e.g. the ones stored in a batch
    assert policy.log_prob(policy.mode(suff_stats=stats), suff_stats=stats).shape == (7, 1)


def test_head_activations_are_expanded_per_output():
    act, sigma_acts = head_activations(["tanh", "tanh", ["none", "softplus"]], 3)
    assert act[-1] == ["none"] * 3 + ["softplus"] * 3
    assert sigma_acts == ["softplus"] * 3
    assert sigma_is_softplus(sigma_acts)
    # one dimension keeps the original univariate configuration, a single name applies to all outputs
    assert head_activations(["tanh", "tanh", ["none", "softplus"]], 1)[0][-1] == ["none", "softplus"]
    assert not sigma_is_softplus(head_activations(["tanh", "tanh", "none"], 2)[1])
    with pytest.raises(ValueError):
        head_activations(["tanh", "tanh", ["none", "softplus", "none"]], 2)
    with pytest.raises(ValueError):
        sigma_is_softplus(["softplus", "none"])


def test_state_dependent_sigma_default_activation_applies_one_softplus(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", "none"])
    states = mx.zeros((3, 4))
    # a strongly negative raw output may give a tiny, but strictly positive sigma (a second softplus would floor it)
    _force_head(policy, [0.0, 0.0], [-20.0, -20.0])
    sigma = np.array(policy.forward(states)[:, 2:])
    assert np.all(sigma > 0) and np.all(sigma < 1e-3)
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(np.array(policy.forward(states)[0, 2:]), expected + 1e-6, atol=1e-5)


def test_state_dependent_sigma_with_softplus_activation_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", ["none", "softplus"]])
    states = mx.zeros((3, 4))
    _force_head(policy, [0.0, 0.0], [-20.0, -1000.0])
    sigma = np.array(policy.forward(states)[:, 2:])
    # the underflowing softplus is clamped, never zero and never floored at ln 2
    assert np.all(sigma > 0) and np.all(sigma < 1e-3)
    assert mx.all(mx.isfinite(policy.log_prob(policy.sample_actions(states=states), states=states))).item()
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(np.array(policy.forward(states)[0, 2:]), expected, atol=1e-5)


def test_mean_is_not_activated_twice(make_pendulum_mc):
    # the mean output is passed through unchanged by forward, the tanh squash is applied exactly once in mode / samples
    policy = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_squash=True, policy_act_fn=["tanh", "tanh", "none"]
    )
    mu_pre = [0.5, -1.5]
    _force_head(policy, mu_pre, [])
    states = mx.zeros((2, 4))
    np.testing.assert_allclose(np.array(policy.forward(states)[0, :2]), mu_pre, atol=1e-7)
    center, scale = (HIGH + LOW) / 2, (HIGH - LOW) / 2
    expected = center + scale * np.tanh(mu_pre)
    np.testing.assert_allclose(np.array(policy.mode(states=states)[0]), expected, atol=1e-6)


def test_squashed_samples_lie_in_the_per_dimension_bounds(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True, policy_sigma=5.0)
    _force_head(policy, [3.0, -3.0], [])
    mx.random.seed(0)
    a = policy.sample_actions(states=mx.zeros((4, 4)), num_actions=2000)
    an = np.array(a)
    assert np.all(an[..., 0] <= 1) and np.all(an[..., 0] >= -1)
    assert np.all(an[..., 1] <= 2) and np.all(an[..., 1] >= -2)
    # extreme mean and large sigma saturate the bounds but keep the log probability finite
    assert mx.all(mx.isfinite(policy.log_prob(a, states=mx.zeros((4, 4))))).item()
    assert an[..., 0].mean() > 0.3 and an[..., 1].mean() < -0.6


@pytest.mark.parametrize("squash", [False, True])
def test_log_prob_is_a_density(make_pendulum_mc, squash):
    policy = _policy(make_pendulum_mc, policy_squash=squash, policy_cov_type="single", policy_sigma=0.5)
    stats = mx.array([[0.3, -0.2, 0.5, 0.5]])
    if squash:
        lo, hi, n = LOW + 1e-4, HIGH - 1e-4, 801
    else:
        lo, hi, n = np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 801
    g0, g1 = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    grid = np.stack(np.meshgrid(g0, g1, indexing="ij"), axis=-1).reshape(1, -1, 2)
    density = np.exp(np.array(policy.log_prob(mx.array(grid, dtype=mx.float32), suff_stats=stats))).reshape(n, n)
    integral = np.trapezoid(np.trapezoid(density, g1, axis=1), g0)
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_squashed_log_prob_matches_the_change_of_variables(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True)
    mu, sigma = np.array([[0.3, -0.7]]), np.array([[0.8, 1.1]])
    u = np.array([[[0.2, -0.4]]])
    a = (HIGH + LOW) / 2 + (HIGH - LOW) / 2 * np.tanh(u)
    expected = (
        _normal_log_prob(u[:, 0], mu, sigma).sum() - (np.log((HIGH - LOW) / 2) + np.log1p(-(np.tanh(u) ** 2))).sum()
    )
    stats = mx.array(np.concatenate([mu, sigma], 1), dtype=mx.float32)
    got = policy.log_prob(mx.array(a, dtype=mx.float32), suff_stats=stats).sum().item()
    assert got == pytest.approx(expected, abs=1e-4)


def test_kl_divergence_matches_the_per_dimension_sum_and_monte_carlo(make_pendulum_mc):
    mc = make_pendulum_mc(env_name="TestPointMass-v0", policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
        mc.add_syncable("policy", policy, 1)
    states = mx.random.normal((3, 4), key=mx.random.key(1))
    batch = _Batch(np.array(states))
    # move the policy away from its (synced) target
    _perturb(policy, 0.1, 2)
    stats = policy.forward(states)
    target = mc.get("policy", True).forward(states)
    s, t = np.array(stats, dtype=np.float64), np.array(target, dtype=np.float64)
    mu, sigma, t_mu, t_sigma = s[:, :2], s[:, 2:], t[:, :2], t[:, 2:]

    kl = policy.kl_divergence(batch, mode="i-projection", est="mean")
    expected = _normal_kl(mu, sigma, t_mu, t_sigma).sum(axis=1).mean()
    assert kl.item() == pytest.approx(expected, abs=1e-6)
    assert policy.kl_divergence(batch, est="sum").item() == pytest.approx(expected * 3, abs=1e-5)
    reverse = policy.kl_divergence(batch, mode="m-projection", est="mean")
    expected_rev = _normal_kl(t_mu, t_sigma, mu, sigma).sum(axis=1).mean()
    assert reverse.item() == pytest.approx(expected_rev, abs=1e-6)

    # Monte Carlo estimate of KL(current || target) from the squashed log probabilities (the squashing is a bijection)
    mx.random.seed(0)
    a = policy.sample_actions(suff_stats=stats, num_actions=100_000)
    mc_kl = (policy.log_prob(a, suff_stats=stats) - policy.log_prob(a, suff_stats=target)).mean()
    assert mc_kl.item() == pytest.approx(expected, rel=0.1, abs=0.02)

    # the gradient reaches the policy parameters
    def loss(params):
        policy.update(params)
        return policy.kl_divergence(batch, mode="i-projection", est="mean")

    grads = mx.grad(loss)(policy.trainable_parameters())
    assert any(mx.abs(g).sum().item() > 0 for _, g in tree_flatten(grads))


def test_one_dimensional_diagonal_policy_equals_the_univariate_policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        diag, uni = DiagonalGaussianPolicy(mc), UnivariateGaussianPolicy(mc)
    uni.network.update(diag.network.parameters())
    states = mx.random.normal((6, 3), key=mx.random.key(3))
    sd, su = diag.forward(states), uni.forward(states)
    assert mx.allclose(sd, su).item()
    actions = uni.sample_actions(suff_stats=su, num_actions=4)
    assert mx.allclose(
        diag.log_prob(mx.expand_dims(actions, -1), suff_stats=sd), uni.log_prob(actions, suff_stats=su), atol=1e-5
    ).item()
    assert mx.allclose(diag.mode(suff_stats=sd), uni.mode(suff_stats=su)).item()


def test_state_covariance_starts_at_policy_sigma(make_pendulum_mc):
    sigma = 0.4
    state = _policy(make_pendulum_mc, policy_cov_type="state", policy_sigma=sigma)
    np.testing.assert_allclose(np.array(state.forward(mx.zeros((1, 4)))[0, 2:]), [sigma, sigma], atol=1e-5)
    uni = _policy(
        make_pendulum_mc, "Pendulum-v1", UnivariateGaussianPolicy, policy_cov_type="state", policy_sigma=sigma
    )
    np.testing.assert_allclose(np.array(uni.forward(mx.zeros((1, 3)))[:, 1]), [sigma], atol=1e-5)


def test_entropy_of_the_unsquashed_policy_and_refusal_for_squashed(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=False, policy_cov_type="single", policy_sigma=0.7)
    expected = 2 * 0.5 * math.log(2 * math.pi * math.e * 0.7**2)
    assert policy.entropy(_Batch(np.zeros((3, 4), np.float32))).item() == pytest.approx(expected, abs=1e-5)
    with pytest.raises(NotImplementedError):
        _policy(make_pendulum_mc, policy_squash=True).entropy(_Batch(np.zeros((3, 4), np.float32)))


def test_policy_parameters_are_the_network_and_the_free_covariance(make_pendulum_mc):
    # the constants of the squashing and a fixed covariance are no trainable parameters
    fixed = _policy(make_pendulum_mc, policy_cov_type="fixed")
    names = {k for k, _ in tree_flatten(fixed.trainable_parameters())}
    assert names and all(k.startswith("network.") for k in names)
    single = _policy(make_pendulum_mc, policy_cov_type="single")
    assert {k for k, _ in tree_flatten(single.trainable_parameters())} - names == {"cov"}
    assert isinstance(fixed.network.layers[0], nn.Linear)
