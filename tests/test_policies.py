import math

import mlx.core as mx
import pytest

from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def _policy(make_pendulum_mc, cls, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        return cls(mc)


def test_state_dependent_sigma_is_positive(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    states = mx.random.normal((512, policy.mc.env.observation_dim), key=mx.random.key(0)) * 10
    sigma = policy.forward(states)[:, 1]
    assert mx.all(sigma > 0).item()
    ll = policy.log_prob(policy.sample_actions(states=states), states=states)
    assert mx.all(mx.isfinite(ll)).item()


def _force_sigma_preactivation(policy, value):
    """Make the last layer output ``[0, value]`` independent of the state."""
    last = policy.network.layers[-1]
    last.weight = mx.zeros_like(last.weight)
    last.bias = mx.array([0.0, value])


def test_state_dependent_sigma_can_be_small_with_default_act_fn(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(mx.zeros((4, policy.mc.env.observation_dim)))[:, 1]
    assert mx.all(sigma > 0).item() and mx.all(sigma < 1e-3).item()


def test_state_dependent_sigma_is_not_softplused_twice(make_pendulum_mc):
    # the original configurations put a softplus on the sigma head through the activation function
    policy = _policy(
        make_pendulum_mc,
        UnivariateGaussianPolicy,
        policy_cov_type="state",
        policy_act_fn=["tanh", "tanh", ["none", "softplus"]],
    )
    states = mx.zeros((4, policy.mc.env.observation_dim))
    _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(states)[:, 1]
    # a second softplus would floor sigma at ln 2 = 0.693
    assert mx.all(sigma > 0).item() and mx.all(sigma < 1e-3).item()
    # a strongly negative input underflows the softplus to zero, sigma must stay strictly positive
    _force_sigma_preactivation(policy, -1000.0)
    sigma = policy.forward(states)[:, 1]
    assert mx.all(sigma > 0).item()
    ll = policy.log_prob(policy.sample_actions(states=states), states=states)
    assert mx.all(mx.isfinite(ll)).item()
    # positive pre-activations are passed through a single softplus: softplus(3) = log(1 + e^3)
    _force_sigma_preactivation(policy, 3.0)
    expected = mx.full((4,), math.log1p(math.exp(3.0)))
    assert mx.allclose(policy.forward(states)[:, 1], expected, atol=1e-5).item()


def test_entropy_refuses_squashed_policy(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    assert policy.squash
    with pytest.raises(NotImplementedError):
        policy.entropy(None)
