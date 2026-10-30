"""The NumPy host actor of the rollout samples exactly the actions, log likelihoods and random keys of policy.act."""

import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.host_actor import BLOCK, normal_from_bits, random_bits, split_chain
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="the host actor uses Accelerate")


def _policy(make_pendulum_mc, env, cls, **overrides):
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        policy = cls(mc)
    keys = iter(mx.random.split(mx.random.key(11), 64))
    policy.update(tree_map(lambda p: p + 0.3 * mx.random.normal(p.shape, key=next(keys)), policy.parameters()))
    return mc, policy


def test_threefry_matches_mlx():
    key = mx.random.key(1234)
    subs, states = split_chain(tuple(int(k) for k in np.array(key)), 3)
    k = key
    for i in range(3):
        new, sub = mx.random.split(k)
        assert np.array_equal(np.array(new), states[i]) and np.array_equal(np.array(sub), subs[i])
        k = new
    normal = mx.compile(lambda k, d: mx.random.normal((d,), key=k), shapeless=False)
    for d in (1, 2, 5, 6):
        expected = np.stack([np.array(normal(mx.array(s), d)) for s in subs])
        assert np.array_equal(normal_from_bits(random_bits(subs, d)), expected)


CASES = [
    ("TestPointMass-v0", DiagonalGaussianPolicy, {"policy_squash": True, "policy_cov_type": "single"}),
    ("TestPointMass-v0", DiagonalGaussianPolicy, {"policy_squash": False, "policy_cov_type": "single"}),
    ("TestPointMass-v0", DiagonalGaussianPolicy, {"policy_squash": True, "policy_cov_type": "fixed"}),
    ("HalfCheetah-v5", DiagonalGaussianPolicy, {"policy_squash": True, "policy_structure": [64, 64]}),
    ("Pendulum-v1", UnivariateGaussianPolicy, {"policy_squash": True, "policy_cov_type": "single"}),
    ("Pendulum-v1", UnivariateGaussianPolicy, {"policy_squash": False, "policy_cov_type": "single"}),
    ("MountainCarContinuous-v0", UnivariateGaussianPolicy, {"policy_squash": True, "policy_cov_type": "fixed"}),
    ("InvertedDoublePendulum-v5", UnivariateGaussianPolicy, {"policy_squash": True, "policy_cov_type": "single"}),
]


@darwin
@pytest.mark.parametrize("env,cls,overrides", CASES)
def test_host_actor_matches_act(make_pendulum_mc, env, cls, overrides):
    mc, policy = _policy(make_pendulum_mc, env, cls, policy_act_fn=["tanh", "tanh", "none"], **overrides)
    rng = np.random.default_rng(0)
    states = (2 * rng.standard_normal((BLOCK + 37, mc.env.observation_dim))).astype(np.float32)
    mx.random.seed(5)
    actions, lls = [], []
    for s in states:
        a, ll = policy.act(mx.array(s[None]), False)
        actions.append(np.array(a).reshape(-1))
        lls.append(np.array(ll).reshape(-1)[0])
    key = np.array(mx.random.state[0])

    mx.random.seed(5)
    actor = policy.host_actor()
    assert actor is not None
    host = [actor(s[None]) for s in states]
    host_lls = actor.finish()
    assert np.array_equal(np.stack(host), np.stack(actions))
    assert np.array_equal(host_lls, np.array(lls))
    assert np.array_equal(np.array(mx.random.state[0]), key)


@darwin
def test_unsupported_configurations_use_act(make_pendulum_mc):
    _, state_cov = _policy(make_pendulum_mc, "TestPointMass-v0", DiagonalGaussianPolicy, policy_cov_type="state")
    assert state_cov.host_actor() is None
    _, elu = _policy(make_pendulum_mc, "TestPointMass-v0", DiagonalGaussianPolicy, policy_act_fn=["elu", "elu", "none"])
    assert elu.host_actor() is None
    _, univariate = _policy(make_pendulum_mc, "Pendulum-v1", UnivariateGaussianPolicy, policy_cov_type="state")
    assert univariate.host_actor() is None
    if mx.metal.is_available():
        _, policy = _policy(make_pendulum_mc, "TestPointMass-v0", DiagonalGaussianPolicy)
        with mx.stream(mx.gpu):
            assert policy.host_actor() is None


@darwin
@pytest.mark.parametrize(
    "env,cls", [("TestPointMass-v0", DiagonalGaussianPolicy), ("Pendulum-v1", UnivariateGaussianPolicy)]
)
def test_rollout_with_and_without_host_actor(make_pendulum_mc, env, cls):
    trajectories = []
    for host in (True, False):
        mx.random.seed(2)
        mc, policy = _policy(
            make_pendulum_mc,
            env,
            cls,
            policy_squash=True,
            policy_act_fn=["tanh", "tanh", "none"],
            env_host_policy=host,
        )
        mx.random.seed(3)
        tc = mc.env.execute_policy(policy, 40, 100)
        trajectories.append((tc.trajectories(), np.array(mx.random.state[0])))
    (a, key_a), (b, key_b) = trajectories
    assert np.array_equal(key_a, key_b)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.actions, y.actions)
        assert np.array_equal(x.log_likelihoods, y.log_likelihoods)
        assert np.array_equal(x.states, y.states)
