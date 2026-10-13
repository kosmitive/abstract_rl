import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def _policy(make_pendulum_mc, cls, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        return cls(mc)


def _force_last_bias(policy, bias):
    """Make the last layer output ``bias`` independent of the state."""
    layers = list(policy.network.layers)
    layer = layers[-1]
    layer = eqx.tree_at(lambda lin: lin.weight, layer, jnp.zeros_like(layer.weight))
    layer = eqx.tree_at(lambda lin: lin.bias, layer, jnp.asarray(bias, dtype=jnp.float32))
    layers[-1] = layer
    policy.network = eqx.tree_at(lambda net: net.layers, policy.network, tuple(layers))


def test_state_dependent_sigma_is_positive(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    states = np.random.default_rng(0).normal(size=(512, policy.mc.env.observation_dim)) * 10
    sigma = policy.forward(states)[:, 1]
    assert np.all(np.asarray(sigma) > 0)
    policy.log_prob(policy.sample_actions(states=states), states=states)


def test_state_dependent_sigma_can_be_small_with_default_act_fn(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    _force_last_bias(policy, [0.0, -20.0])
    sigma = policy.forward(np.zeros((4, policy.mc.env.observation_dim)))[:, 1]
    assert np.all(np.asarray(sigma) > 0) and np.all(np.asarray(sigma) < 1e-3)


def test_state_dependent_sigma_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(
        make_pendulum_mc,
        UnivariateGaussianPolicy,
        policy_cov_type="state",
        policy_act_fn=["tanh", "tanh", ["none", "softplus"]],
    )
    states = np.zeros((4, policy.mc.env.observation_dim))
    _force_last_bias(policy, [0.0, -20.0])
    sigma = np.asarray(policy.forward(states)[:, 1])
    assert np.all(sigma > 0) and np.all(sigma < 1e-3)
    _force_last_bias(policy, [0.0, -1000.0])
    sigma = np.asarray(policy.forward(states)[:, 1])
    assert np.all(sigma > 0)
    policy.log_prob(policy.sample_actions(states=states), states=states)
    _force_last_bias(policy, [0.0, 3.0])
    expected = math.log1p(math.exp(3.0))
    assert np.allclose(np.asarray(policy.forward(states)[:, 1]), expected, atol=1e-5)


def test_entropy_refuses_squashed_policy(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    assert policy.squash
    with pytest.raises(NotImplementedError):
        policy.entropy(None)


def test_numpy_rollout_matches_the_jax_density(make_pendulum_mc):
    """The environment loop evaluates the policy in NumPy. The stored log density must match JAX."""
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    states = np.random.default_rng(0).normal(size=(8, policy.mc.env.observation_dim)).astype(np.float32)
    actions, log_density = policy.act(states, deterministic=True)
    expected = np.asarray(policy.mode(states))
    assert np.allclose(actions, expected, atol=1e-5)
    jax_log = np.asarray(policy.log_prob(actions, states=states)).reshape(-1)
    assert np.allclose(log_density, jax_log, atol=1e-5)
    sampled, sampled_log = policy.act(states, deterministic=False)
    jax_sampled = np.asarray(policy.log_prob(sampled, states=states)).reshape(-1)
    assert np.allclose(sampled_log, jax_sampled, atol=1e-5)
    assert sampled.shape == (8, 1)
