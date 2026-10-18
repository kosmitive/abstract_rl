import math

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

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
    with torch.no_grad():
        layer = policy.network.layers[-1]
        layer.weight.zero_()
        layer.bias[:] = torch.tensor([*mu_pre, *sigma_pre])


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
    states = torch.randn(7, 4)
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
    states = torch.zeros(3, 4)
    # a strongly negative raw output may give a tiny, but strictly positive sigma (a second softplus would floor it)
    _force_head(policy, [0.0, 0.0], [-20.0, -20.0])
    sigma = policy.forward(states)[:, 2:]
    assert torch.all(sigma > 0) and torch.all(sigma < 1e-3)
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = torch.tensor([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    assert torch.allclose(policy.forward(states)[0, 2:], expected + 1e-6, atol=1e-5)


def test_state_dependent_sigma_with_softplus_activation_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", ["none", "softplus"]])
    states = torch.zeros(3, 4)
    _force_head(policy, [0.0, 0.0], [-20.0, -1000.0])
    sigma = policy.forward(states)[:, 2:]
    # the underflowing softplus is clamped, never zero and never floored at ln 2
    assert torch.all(sigma > 0) and torch.all(sigma < 1e-3)
    policy.log_prob(policy.sample_actions(states=states), states=states)
    _force_head(policy, [0.0, 0.0], [3.0, -3.0])
    expected = torch.tensor([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    assert torch.allclose(policy.forward(states)[0, 2:], expected, atol=1e-5)


def test_mean_is_not_activated_twice(make_pendulum_mc):
    # the mean output is passed through unchanged by forward, the tanh squash is applied exactly once in mode / samples
    policy = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_squash=True, policy_act_fn=["tanh", "tanh", "none"]
    )
    mu_pre = [0.5, -1.5]
    _force_head(policy, mu_pre, [])
    states = torch.zeros(2, 4)
    assert torch.allclose(policy.forward(states)[0, :2], torch.tensor(mu_pre))
    center, scale = (HIGH + LOW) / 2, (HIGH - LOW) / 2
    expected = torch.as_tensor(center + scale * np.tanh(mu_pre), dtype=torch.float32)
    assert torch.allclose(policy.mode(states=states)[0], expected, atol=1e-6)


def test_squashed_samples_lie_in_the_per_dimension_bounds(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True, policy_sigma=5.0)
    _force_head(policy, [3.0, -3.0], [])
    torch.manual_seed(0)
    a = policy.sample_actions(states=torch.zeros(4, 4), num_actions=2000)
    assert torch.all(a[..., 0] <= 1) and torch.all(a[..., 0] >= -1)
    assert torch.all(a[..., 1] <= 2) and torch.all(a[..., 1] >= -2)
    # extreme mean and large sigma saturate the bounds but keep the log probability finite
    assert torch.isfinite(policy.log_prob(a, states=torch.zeros(4, 4))).all()
    assert a[..., 0].mean() > 0.3 and a[..., 1].mean() < -0.6


@pytest.mark.parametrize("squash", [False, True])
def test_log_prob_is_a_density(make_pendulum_mc, squash):
    policy = _policy(make_pendulum_mc, policy_squash=squash, policy_cov_type="single", policy_sigma=0.5)
    stats = torch.tensor([[0.3, -0.2, 0.5, 0.5]])
    if squash:
        lo, hi, n = LOW + 1e-4, HIGH - 1e-4, 801
    else:
        lo, hi, n = np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 801
    g0, g1 = torch.linspace(lo[0], hi[0], n), torch.linspace(lo[1], hi[1], n)
    grid = torch.stack(torch.meshgrid(g0, g1, indexing="ij"), dim=-1).reshape(1, -1, 2)
    density = policy.log_prob(grid, suff_stats=stats).exp().reshape(n, n)
    integral = torch.trapezoid(torch.trapezoid(density, g1, dim=1), g0).item()
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_squashed_log_prob_matches_the_change_of_variables(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True)
    mu, sigma = torch.tensor([[0.3, -0.7]]), torch.tensor([[0.8, 1.1]])
    u = torch.tensor([[[0.2, -0.4]]])
    a = torch.as_tensor((HIGH + LOW) / 2, dtype=torch.float32) + torch.as_tensor(
        (HIGH - LOW) / 2, dtype=torch.float32
    ) * torch.tanh(u)
    expected = (
        Normal(mu, sigma).log_prob(u[:, 0]).sum()
        - (torch.log(torch.as_tensor((HIGH - LOW) / 2, dtype=torch.float32)) + torch.log1p(-torch.tanh(u).pow(2))).sum()
    )
    assert torch.allclose(policy.log_prob(a, suff_stats=torch.cat([mu, sigma], 1)).sum(), expected, atol=1e-4)


def test_kl_divergence_matches_the_per_dimension_sum_and_monte_carlo(make_pendulum_mc):
    mc = make_pendulum_mc(env_name="TestPointMass-v0", policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
        mc.add_syncable("policy", policy, 1)
    states = torch.randn(3, 4)
    batch = _Batch(states.numpy())
    # move the policy away from its (synced) target
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.3 * torch.randn_like(p))
    stats = policy.forward(states)
    target = mc.get("policy", True).forward(states)
    mu, sigma, t_mu, t_sigma = stats[:, :2], stats[:, 2:], target[:, :2], target[:, 2:]

    kl = policy.kl_divergence(batch, mode="i-projection", est="mean")
    expected = torch.distributions.kl.kl_divergence(Normal(mu, sigma), Normal(t_mu, t_sigma)).sum(dim=1).mean()
    assert torch.allclose(kl, expected, atol=1e-6)
    assert torch.allclose(policy.kl_divergence(batch, est="sum"), expected * 3, atol=1e-5)
    reverse = policy.kl_divergence(batch, mode="m-projection", est="mean")
    expected_rev = torch.distributions.kl.kl_divergence(Normal(t_mu, t_sigma), Normal(mu, sigma)).sum(dim=1).mean()
    assert torch.allclose(reverse, expected_rev, atol=1e-6)

    # Monte Carlo estimate of KL(current || target) from the squashed log probabilities (the squashing is a bijection)
    torch.manual_seed(0)
    with torch.no_grad():
        a = policy.sample_actions(suff_stats=stats, num_actions=100_000)
        mc_kl = (policy.log_prob(a, suff_stats=stats) - policy.log_prob(a, suff_stats=target)).mean()
    assert mc_kl.item() == pytest.approx(expected.item(), rel=0.1, abs=0.02)
    # the gradient reaches the policy parameters
    kl.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_one_dimensional_diagonal_policy_equals_the_univariate_policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        diag, uni = DiagonalGaussianPolicy(mc), UnivariateGaussianPolicy(mc)
    uni.network.load_state_dict(diag.network.state_dict())
    states = torch.randn(6, 3)
    sd, su = diag.forward(states), uni.forward(states)
    assert torch.allclose(sd, su)
    actions = uni.sample_actions(suff_stats=su, num_actions=4)
    assert torch.allclose(
        diag.log_prob(actions.unsqueeze(-1), suff_stats=sd), uni.log_prob(actions, suff_stats=su), atol=1e-5
    )
    assert torch.allclose(diag.mode(suff_stats=sd), uni.mode(suff_stats=su))


def _unpack_tril(packed: torch.Tensor, action_dim: int) -> torch.Tensor:
    rows, cols = torch.tril_indices(action_dim, action_dim)
    factor = packed.new_zeros(packed.shape[0], action_dim, action_dim)
    factor[:, rows, cols] = packed
    return factor


def test_full_covariance_log_prob_matches_multivariate_normal(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="full", policy_squash=False)
    # packed row-major triangle for two actions: L00, L10, L11
    _force_head(policy, [0.2, -0.4], [0.0, 0.5, -0.3])
    states = torch.zeros(3, 4)
    stats = policy.forward(states)
    diagonal = torch.nn.functional.softplus(torch.tensor([0.0, -0.3])) + 1e-6
    assert torch.allclose(stats[0, :2], torch.tensor([0.2, -0.4]))
    assert torch.allclose(stats[0, 2:], torch.tensor([diagonal[0], 0.5, diagonal[1]]), atol=1e-6)
    actions = torch.tensor(
        [[[0.1, -0.2], [0.0, 0.3]], [[-0.4, 0.2], [0.2, -0.1]], [[0.3, 0.1], [-0.2, 0.0]]], dtype=torch.float32
    )
    factor = _unpack_tril(stats[:, 2:], 2)
    dist = MultivariateNormal(stats[:, :2], scale_tril=factor)
    expected = torch.stack([dist.log_prob(actions[:, k]) for k in range(actions.shape[1])], dim=1)
    assert torch.allclose(policy.log_prob(actions, suff_stats=stats), expected, atol=1e-5)
    # one action per state, the shape stored on a batch
    assert torch.allclose(policy.log_prob(actions[:, 0], suff_stats=stats).squeeze(1), expected[:, 0], atol=1e-5)


def test_full_covariance_kl_of_a_policy_with_itself_is_zero(make_pendulum_mc):
    mc = make_pendulum_mc(env_name="TestPointMass-v0", policy_cov_type="full", policy_squash=True)
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
        mc.add_syncable("policy", policy, 1)
    batch = _Batch(np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32))
    # TRPO uses the m-projection mean, MPO the same total KL
    assert policy.kl_divergence(batch, mode="i-projection", est="mean").item() == pytest.approx(0.0, abs=1e-6)
    assert policy.kl_divergence(batch, mode="m-projection", est="sum").item() == pytest.approx(0.0, abs=1e-5)

    torch.manual_seed(0)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.2 * torch.randn_like(p))
    states = torch.as_tensor(batch.states)
    with torch.no_grad():
        stats, target = policy.forward(states), mc.get("policy", True).forward(states)
    current = MultivariateNormal(stats[:, :2], scale_tril=_unpack_tril(stats[:, 2:], 2))
    old = MultivariateNormal(target[:, :2], scale_tril=_unpack_tril(target[:, 2:], 2))
    kl = policy.kl_divergence(batch, mode="m-projection", est="mean")
    expected = torch.distributions.kl.kl_divergence(old, current).mean()
    assert torch.allclose(kl, expected, atol=1e-5)
    assert kl.detach().item() > 0
    kl.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in policy.parameters())


def test_state_and_full_covariance_start_at_policy_sigma(make_pendulum_mc):
    sigma = 0.4
    state = _policy(make_pendulum_mc, policy_cov_type="state", policy_sigma=sigma)
    assert torch.allclose(state.forward(torch.zeros(1, 4))[:, 2:], torch.full((2,), sigma), atol=1e-5)
    full = _policy(make_pendulum_mc, policy_cov_type="full", policy_sigma=sigma)
    assert torch.allclose(full.forward(torch.zeros(1, 4))[0, [2, 4]], torch.full((2,), sigma), atol=1e-5)
    uni = _policy(
        make_pendulum_mc, "Pendulum-v1", UnivariateGaussianPolicy, policy_cov_type="state", policy_sigma=sigma
    )
    assert torch.allclose(uni.forward(torch.zeros(1, 3))[:, 1], torch.tensor([sigma]), atol=1e-5)


def test_entropy_of_the_unsquashed_policy_and_refusal_for_squashed(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=False, policy_cov_type="single", policy_sigma=0.7)
    expected = 2 * 0.5 * math.log(2 * math.pi * math.e * 0.7**2)
    assert policy.entropy(_Batch(np.zeros((3, 4), np.float32))).item() == pytest.approx(expected, abs=1e-5)
    with pytest.raises(NotImplementedError):
        _policy(make_pendulum_mc, policy_squash=True).entropy(_Batch(np.zeros((3, 4), np.float32)))
