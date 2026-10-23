import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.policy.continuous.diagonal_gaussian_policy import (
    DiagonalGaussianPolicy,
    head_activations,
    sigma_is_softplus,
)
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

LOW, HIGH = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
KEY = jax.random.key(0)


class _Batch:
    def __init__(self, states):
        self.states = states
        self.mask = None


def _policy(make_pendulum_mc, env="TestPointMass-v0", cls=DiagonalGaussianPolicy, **overrides):
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        return cls(mc, KEY)


def _force_head(policy, mu_pre, sigma_pre):
    """Make the last layer output ``[mu_pre, sigma_pre]`` independent of the state."""
    last = policy.network.layers[-1]
    return eqx.tree_at(
        lambda p: (p.network.layers[-1].weight, p.network.layers[-1].bias),
        policy,
        (jnp.zeros_like(last.weight), jnp.asarray([*mu_pre, *sigma_pre], jnp.float32)),
    )


def _perturb(module, key, scale):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    leaves = [leaf + scale * jax.random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys, strict=True)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _unpack_tril(packed, action_dim):
    rows, cols = np.tril_indices(action_dim)
    return jnp.zeros((packed.shape[0], action_dim, action_dim)).at[:, rows, cols].set(packed)


def _kl_normal(m0, s0, m1, s1):
    return 0.5 * (s0**2 / s1**2 + (m1 - m0) ** 2 / s1**2 - 1 + 2 * np.log(s1 / s0))


def _kl_mvn(m0, c0, m1, c1):
    k = m0.shape[-1]
    inv = np.linalg.inv(c1)
    d = m1 - m0
    return 0.5 * (np.trace(inv @ c0) + d @ inv @ d - k + np.log(np.linalg.det(c1) / np.linalg.det(c0)))


def test_select_policy_uses_the_diagonal_policy_for_multi_dimensional_actions(make_pendulum_mc):
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, "Pendulum-v1", select_policy), UnivariateGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_type="diagonal"), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_cov_type="full"), DiagonalGaussianPolicy)
    assert isinstance(
        _policy(make_pendulum_mc, "Pendulum-v1", select_policy, policy_cov_type="full"), DiagonalGaussianPolicy
    )


def test_shapes(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state")
    states = jax.random.normal(KEY, (7, 4))
    stats = policy.forward(states)
    assert stats.shape == (7, 4) == (7, policy.get_suff_stats_dim())
    actions = policy.sample_actions(KEY, suff_stats=stats, num_actions=5)
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
    states = jnp.zeros((3, 4))
    # a strongly negative raw output may give a tiny, but strictly positive sigma (a second softplus would floor it)
    sigma = _force_head(policy, [0.0, 0.0], [-20.0, -20.0]).forward(states)[:, 2:]
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(
        _force_head(policy, [0.0, 0.0], [3.0, -3.0]).forward(states)[0, 2:], expected + 1e-6, atol=1e-5
    )


def test_state_dependent_sigma_with_softplus_activation_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", ["none", "softplus"]])
    states = jnp.zeros((3, 4))
    p = _force_head(policy, [0.0, 0.0], [-20.0, -1000.0])
    sigma = p.forward(states)[:, 2:]
    # the underflowing softplus is clamped, never zero and never floored at ln 2
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    assert jnp.all(jnp.isfinite(p.log_prob(p.sample_actions(KEY, states=states), states=states)))
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(_force_head(policy, [0.0, 0.0], [3.0, -3.0]).forward(states)[0, 2:], expected, atol=1e-5)


def test_mean_is_not_activated_twice(make_pendulum_mc):
    # the mean output is passed through unchanged by forward, the tanh squash is applied exactly once in mode / samples
    policy = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_squash=True, policy_act_fn=["tanh", "tanh", "none"]
    )
    mu_pre = [0.5, -1.5]
    policy = _force_head(policy, mu_pre, [])
    states = jnp.zeros((2, 4))
    np.testing.assert_allclose(policy.forward(states)[0, :2], mu_pre, atol=1e-7)
    center, scale = (HIGH + LOW) / 2, (HIGH - LOW) / 2
    np.testing.assert_allclose(policy.mode(states=states)[0], center + scale * np.tanh(mu_pre), atol=1e-6)


def test_squashed_samples_lie_in_the_per_dimension_bounds(make_pendulum_mc):
    policy = _force_head(_policy(make_pendulum_mc, policy_squash=True, policy_sigma=5.0), [3.0, -3.0], [])
    a = policy.sample_actions(KEY, states=jnp.zeros((4, 4)), num_actions=2000)
    assert jnp.all(a[..., 0] <= 1) and jnp.all(a[..., 0] >= -1)
    assert jnp.all(a[..., 1] <= 2) and jnp.all(a[..., 1] >= -2)
    # extreme mean and large sigma saturate the bounds but keep the log probability finite
    assert jnp.isfinite(policy.log_prob(a, states=jnp.zeros((4, 4)))).all()
    assert a[..., 0].mean() > 0.3 and a[..., 1].mean() < -0.6


@pytest.mark.parametrize("squash", [False, True])
def test_log_prob_is_a_density(make_pendulum_mc, squash):
    policy = _policy(make_pendulum_mc, policy_squash=squash, policy_cov_type="single", policy_sigma=0.5)
    stats = jnp.array([[0.3, -0.2, 0.5, 0.5]])
    if squash:
        lo, hi, n = LOW + 1e-4, HIGH - 1e-4, 801
    else:
        lo, hi, n = np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 801
    g0, g1 = jnp.linspace(lo[0], hi[0], n), jnp.linspace(lo[1], hi[1], n)
    grid = jnp.stack(jnp.meshgrid(g0, g1, indexing="ij"), axis=-1).reshape(1, -1, 2)
    density = jnp.exp(policy.log_prob(grid, suff_stats=stats)).reshape(n, n)
    integral = float(jnp.trapezoid(jnp.trapezoid(density, g1, axis=1), g0))
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_squashed_log_prob_matches_the_change_of_variables(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True)
    mu, sigma = np.array([[0.3, -0.7]]), np.array([[0.8, 1.1]])
    u = np.array([[[0.2, -0.4]]])
    scale = (HIGH - LOW) / 2
    a = (HIGH + LOW) / 2 + scale * np.tanh(u)
    expected = norm.logpdf(u[:, 0], mu, sigma).sum() - (np.log(scale) + np.log1p(-(np.tanh(u) ** 2))).sum()
    got = policy.log_prob(jnp.asarray(a), suff_stats=jnp.asarray(np.concatenate([mu, sigma], 1))).sum()
    assert float(got) == pytest.approx(float(expected), abs=1e-4)


def test_kl_divergence_matches_the_per_dimension_sum_and_monte_carlo(make_pendulum_mc):
    target = _policy(make_pendulum_mc, policy_cov_type="state", policy_squash=True)
    # move the policy away from its (synced) target
    policy = _perturb(target, jax.random.key(1), 0.1)
    state = ModelState.create({}, {"policy": (policy, target)})
    states = jax.random.normal(jax.random.key(2), (3, 4))
    batch = _Batch(states)
    stats, t_stats = np.asarray(policy.forward(states)), np.asarray(target.forward(states))
    mu, sigma, t_mu, t_sigma = stats[:, :2], stats[:, 2:], t_stats[:, :2], t_stats[:, 2:]

    kl = policy.kl_divergence(state, batch, mode="i-projection", est="mean")
    expected = _kl_normal(mu, sigma, t_mu, t_sigma).sum(axis=1).mean()
    assert float(kl) == pytest.approx(expected, abs=1e-5)
    assert float(policy.kl_divergence(state, batch, est="sum")) == pytest.approx(expected * 3, abs=1e-4)
    reverse = policy.kl_divergence(state, batch, mode="m-projection", est="mean")
    assert float(reverse) == pytest.approx(_kl_normal(t_mu, t_sigma, mu, sigma).sum(axis=1).mean(), abs=1e-5)

    # Monte Carlo estimate of KL(current || target) from the squashed log probabilities (the squashing is a bijection)
    a = policy.sample_actions(jax.random.key(3), suff_stats=jnp.asarray(stats), num_actions=100_000)
    mc_kl = float(
        jnp.mean(
            policy.log_prob(a, suff_stats=jnp.asarray(stats)) - policy.log_prob(a, suff_stats=jnp.asarray(t_stats))
        )
    )
    assert mc_kl == pytest.approx(float(expected), rel=0.1, abs=0.02)
    # the gradient reaches the policy parameters
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="i-projection", est="mean"))(policy)
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))


def test_one_dimensional_diagonal_policy_equals_the_univariate_policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        diag, uni = DiagonalGaussianPolicy(mc, KEY), UnivariateGaussianPolicy(mc, jax.random.key(1))
    uni = eqx.tree_at(lambda u: u.network, uni, diag.network)
    states = jax.random.normal(KEY, (6, 3))
    sd, su = diag.forward(states), uni.forward(states)
    np.testing.assert_allclose(sd, su, atol=1e-7)
    actions = uni.sample_actions(KEY, suff_stats=su, num_actions=4)
    np.testing.assert_allclose(
        diag.log_prob(actions[..., None], suff_stats=sd), uni.log_prob(actions, suff_stats=su), atol=1e-5
    )
    np.testing.assert_allclose(diag.mode(suff_stats=sd), uni.mode(suff_stats=su), atol=1e-7)


def test_full_covariance_log_prob_matches_multivariate_normal(make_pendulum_mc):
    # packed row-major triangle for two actions: L00, L10, L11
    policy = _force_head(
        _policy(make_pendulum_mc, policy_cov_type="full", policy_squash=False), [0.2, -0.4], [0.0, 0.5, -0.3]
    )
    states = jnp.zeros((3, 4))
    stats = policy.forward(states)
    diagonal = jax.nn.softplus(jnp.array([0.0, -0.3])) + 1e-6
    np.testing.assert_allclose(stats[0, :2], [0.2, -0.4], atol=1e-7)
    np.testing.assert_allclose(stats[0, 2:], [diagonal[0], 0.5, diagonal[1]], atol=1e-6)
    actions = jnp.array([[[0.1, -0.2], [0.0, 0.3]], [[-0.4, 0.2], [0.2, -0.1]], [[0.3, 0.1], [-0.2, 0.0]]])
    factor = _unpack_tril(stats[:, 2:], 2)
    cov = factor @ jnp.swapaxes(factor, -1, -2)
    expected = jnp.stack(
        [jax.vmap(multivariate_normal.logpdf)(actions[:, k], stats[:, :2], cov) for k in range(actions.shape[1])],
        axis=1,
    )
    np.testing.assert_allclose(policy.log_prob(actions, suff_stats=stats), expected, atol=1e-5)
    # one action per state, the shape stored on a batch
    np.testing.assert_allclose(policy.log_prob(actions[:, 0], suff_stats=stats)[:, 0], expected[:, 0], atol=1e-5)


def test_full_covariance_kl_of_a_policy_with_itself_is_zero(make_pendulum_mc):
    target = _policy(make_pendulum_mc, policy_cov_type="full", policy_squash=True)
    batch = _Batch(jnp.asarray(np.random.default_rng(0).normal(size=(6, 4)), jnp.float32))
    same = ModelState.create({}, {"policy": (target, target)})
    # TRPO uses the m-projection mean, MPO the same total KL
    assert float(target.kl_divergence(same, batch, mode="i-projection", est="mean")) == pytest.approx(0.0, abs=1e-6)
    assert float(target.kl_divergence(same, batch, mode="m-projection", est="sum")) == pytest.approx(0.0, abs=1e-5)

    policy = _perturb(target, jax.random.key(4), 0.2)
    state = ModelState.create({}, {"policy": (policy, target)})
    stats, t_stats = np.asarray(policy.forward(batch.states)), np.asarray(target.forward(batch.states))
    f, tf = (
        np.asarray(_unpack_tril(jnp.asarray(stats[:, 2:]), 2)),
        np.asarray(_unpack_tril(jnp.asarray(t_stats[:, 2:]), 2)),
    )
    expected = np.mean([_kl_mvn(t_stats[i, :2], tf[i] @ tf[i].T, stats[i, :2], f[i] @ f[i].T) for i in range(6)])
    kl = policy.kl_divergence(state, batch, mode="m-projection", est="mean")
    assert float(kl) == pytest.approx(expected, abs=1e-4)
    assert float(kl) > 0
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))


def test_full_covariance_is_isotropic_at_init(make_pendulum_mc):
    sigma = 0.4
    policy = _policy(make_pendulum_mc, policy_cov_type="full", policy_sigma=sigma)
    states = jax.random.normal(KEY, (16, 4))
    factor = _unpack_tril(policy.forward(states)[:, policy.action_dim :], policy.action_dim)
    cov = factor @ jnp.swapaxes(factor, -1, -2)
    np.testing.assert_allclose(cov, np.broadcast_to((sigma**2) * np.eye(2), cov.shape), atol=1e-5)


def test_state_and_full_covariance_start_at_policy_sigma(make_pendulum_mc):
    sigma = 0.4
    state = _policy(make_pendulum_mc, policy_cov_type="state", policy_sigma=sigma)
    np.testing.assert_allclose(state.forward(jnp.zeros((1, 4)))[:, 2:], np.full((1, 2), sigma), atol=1e-5)
    full = _policy(make_pendulum_mc, policy_cov_type="full", policy_sigma=sigma)
    np.testing.assert_allclose(full.forward(jnp.zeros((1, 4)))[0, jnp.array([2, 4])], np.full(2, sigma), atol=1e-5)
    uni = _policy(
        make_pendulum_mc, "Pendulum-v1", UnivariateGaussianPolicy, policy_cov_type="state", policy_sigma=sigma
    )
    np.testing.assert_allclose(uni.forward(jnp.zeros((1, 3)))[:, 1], [sigma], atol=1e-5)


def test_entropy_of_the_unsquashed_policy_and_refusal_for_squashed(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=False, policy_cov_type="single", policy_sigma=0.7)
    expected = 2 * 0.5 * math.log(2 * math.pi * math.e * 0.7**2)
    assert float(policy.entropy(None, _Batch(jnp.zeros((3, 4))))) == pytest.approx(expected, abs=1e-5)
    with pytest.raises(NotImplementedError):
        _policy(make_pendulum_mc, policy_squash=True).entropy(None, _Batch(jnp.zeros((3, 4))))
