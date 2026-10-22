import math

import pytest
import torch

from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

# the torch policies / value nets build the ported (Equinox) MLP;
# re-enabled when their port lands (plan 1 tasks 8-10 / plan 2)
pytestmark = pytest.mark.skip(reason="porting to JAX: torch modules on the ported MLP")


def _policy(make_pendulum_mc, cls, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        return cls(mc)


def test_state_dependent_sigma_is_positive(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    torch.manual_seed(0)
    states = torch.randn(512, policy.mc.env.observation_dim) * 10
    sigma = policy.forward(states)[:, 1]
    assert torch.all(sigma > 0)
    policy.log_prob(policy.sample_actions(states=states), states=states)  # Normal(...) validates sigma > 0


def _force_sigma_preactivation(policy, value):
    """Make the last layer output ``[0, value]`` independent of the state."""
    with torch.no_grad():
        policy.network.layers[-1].weight.zero_()
        policy.network.layers[-1].bias[:] = torch.tensor([0.0, value])


def test_state_dependent_sigma_can_be_small_with_default_act_fn(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(torch.zeros(4, policy.mc.env.observation_dim))[:, 1]
    assert torch.all(sigma > 0) and torch.all(sigma < 1e-3)


def test_state_dependent_sigma_is_not_softplused_twice(make_pendulum_mc):
    # the original configurations put a softplus on the sigma head through the activation function
    policy = _policy(
        make_pendulum_mc,
        UnivariateGaussianPolicy,
        policy_cov_type="state",
        policy_act_fn=["tanh", "tanh", ["none", "softplus"]],
    )
    states = torch.zeros(4, policy.mc.env.observation_dim)
    _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(states)[:, 1]
    # a second softplus would floor sigma at ln 2 = 0.693
    assert torch.all(sigma > 0) and torch.all(sigma < 1e-3)
    # a strongly negative input underflows the softplus to zero, sigma must stay strictly positive
    _force_sigma_preactivation(policy, -1000.0)
    sigma = policy.forward(states)[:, 1]
    assert torch.all(sigma > 0)
    policy.log_prob(policy.sample_actions(states=states), states=states)
    # positive pre-activations are passed through a single softplus: softplus(3) = log(1 + e^3)
    _force_sigma_preactivation(policy, 3.0)
    assert torch.allclose(policy.forward(states)[:, 1], torch.full((4,), math.log1p(math.exp(3.0))), atol=1e-5)


def test_entropy_refuses_squashed_policy(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    assert policy.squash
    with pytest.raises(NotImplementedError):
        policy.entropy(None)
