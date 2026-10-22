import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

KEY = jax.random.key(0)


def _policy(make_pendulum_mc, cls, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        return cls(mc, KEY)


def _force_sigma_preactivation(policy, value):
    """Make the last layer output ``[0, value]`` independent of the state."""
    last = policy.network.layers[-1]
    return eqx.tree_at(
        lambda p: (p.network.layers[-1].weight, p.network.layers[-1].bias),
        policy,
        (jnp.zeros_like(last.weight), jnp.array([0.0, value], jnp.float32)),
    )


def test_state_dependent_sigma_is_positive(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    states = jax.random.normal(jax.random.key(1), (512, 3)) * 10
    sigma = policy.forward(states)[:, 1]
    assert jnp.all(sigma > 0)
    assert jnp.all(jnp.isfinite(policy.log_prob(policy.sample_actions(KEY, states=states), states=states)))


def test_state_dependent_sigma_can_be_small_with_default_act_fn(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    policy = _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(jnp.zeros((4, 3)))[:, 1]
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)


def test_state_dependent_sigma_is_not_softplused_twice(make_pendulum_mc):
    # the original configurations put a softplus on the sigma head through the activation function
    policy = _policy(
        make_pendulum_mc,
        UnivariateGaussianPolicy,
        policy_cov_type="state",
        policy_act_fn=["tanh", "tanh", ["none", "softplus"]],
    )
    states = jnp.zeros((4, 3))
    sigma = _force_sigma_preactivation(policy, -20.0).forward(states)[:, 1]
    # a second softplus would floor sigma at ln 2 = 0.693
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    # a strongly negative input underflows the softplus to zero, sigma must stay strictly positive
    p = _force_sigma_preactivation(policy, -1000.0)
    assert jnp.all(p.forward(states)[:, 1] > 0)
    assert jnp.all(jnp.isfinite(p.log_prob(p.sample_actions(KEY, states=states), states=states)))
    # positive pre-activations are passed through a single softplus: softplus(3) = log(1 + e^3)
    p = _force_sigma_preactivation(policy, 3.0)
    np.testing.assert_allclose(p.forward(states)[:, 1], np.full(4, math.log1p(math.exp(3.0))), atol=1e-5)


def test_entropy_refuses_squashed_policy(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    assert policy.squash
    with pytest.raises(NotImplementedError):
        policy.entropy(ModelState.create({}, {"policy": (policy, policy)}), Batch(*[None] * 6, {}))


def test_float64_states_give_float32_stats(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    stats = policy.forward(np.random.default_rng(0).normal(size=(5, 3)))  # numpy float64, e.g. Pendulum observations
    assert stats.dtype == jnp.float32


def test_two_policies_have_equal_static_structure(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("policy"):
        a, b = UnivariateGaussianPolicy(mc, KEY), UnivariateGaussianPolicy(mc, jax.random.key(9))
    assert jax.tree.structure(a) == jax.tree.structure(b)


def test_policy_methods_jit(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    s = jnp.zeros((6, 3))
    out = jax.jit(lambda p, k: p.log_prob(p.sample_actions(k, states=s, num_actions=4), states=s))(policy, KEY)
    assert out.shape == (6, 4)
