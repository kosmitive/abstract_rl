import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet

N, PAD = 6, 3
CASES = {"univariate": ("Pendulum-v1", 3), "diagonal": ("TestPointMass-v0", 4)}


def _perturb(module, key, scale):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    leaves = [leaf + scale * jax.random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys, strict=True)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _pad(x, fill):
    return jnp.concatenate([x, jnp.full((PAD, *x.shape[1:]), fill, x.dtype)])


def _pair(states, actions, ll, a, extra=None):
    """The same data as an unpadded batch and as a batch padded with NaN/inf rows plus a mask."""
    extra = extra or {}
    z = jnp.zeros((N, 1))
    clean = Batch(states, actions, ll, z, states, z, {"a": a, **extra})
    mask = jnp.arange(N + PAD) < N
    fills = [jnp.nan, jnp.inf, -jnp.inf]
    padded = Batch(
        _pad(states, jnp.nan),
        _pad(actions, jnp.inf),
        _pad(ll, jnp.nan),
        _pad(z, jnp.nan),
        _pad(states, -jnp.inf),
        _pad(z, jnp.nan),
        {"a": _pad(a, fills[0]), **{k: _pad(v, fills[1]) for k, v in extra.items()}},
        mask=mask,
    )
    return clean, padded


def _setup(make_pendulum_mc, case):
    env, dim = CASES[case]
    mc = make_pendulum_mc(env_name=env, policy_cov_type="single", policy_squash=False)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    target = _perturb(policy, jax.random.key(1), 0.1)
    states = jax.random.normal(jax.random.key(2), (N, dim))
    actions = target.sample_actions(jax.random.key(3), states=states)
    ll = target.log_prob(actions, states=states).reshape(-1)
    a = jax.random.normal(jax.random.key(4), (N,))
    return policy, target, *_pair(states, actions, ll, a)


def _same(f, policy, state_of, clean, padded):
    """Values and policy gradients of ``f(state, batch)`` agree between the clean and the padded batch."""
    outs = []
    for batch in (clean, padded):
        v, g = eqx.filter_value_and_grad(lambda p, b=batch: f(state_of(p), b))(policy)
        outs.append((v, g))
    (v0, g0), (v1, g1) = outs
    assert np.isfinite(float(v1))
    np.testing.assert_allclose(v1, v0, rtol=1e-5, atol=1e-6)
    for x, y in zip(jax.tree.leaves(g0), jax.tree.leaves(g1), strict=True):
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_padded_rows_do_not_reach_likelihood_ratio(make_pendulum_mc, case):
    policy, target, clean, padded = _setup(make_pendulum_mc, case)
    state_of = lambda p: ModelState.create({}, {"policy": (policy, target)}).set("policy", p)  # noqa: E731
    _same(lambda st, b: st.get("policy", False).likelihood_ratio(st, b), policy, state_of, clean, padded)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", ["i-projection", "m-projection"])
@pytest.mark.parametrize("est", ["sum", "mean"])
def test_padded_rows_do_not_reach_kl_divergence(make_pendulum_mc, case, mode, est):
    policy, target, clean, padded = _setup(make_pendulum_mc, case)
    state_of = lambda p: ModelState.create({}, {"policy": (policy, target)}).set("policy", p)  # noqa: E731
    _same(
        lambda st, b: st.get("policy", False).kl_divergence(st, b, mode=mode, est=est), policy, state_of, clean, padded
    )


@pytest.mark.parametrize("case", CASES)
def test_padded_rows_do_not_reach_entropy(make_pendulum_mc, case):
    policy, target, clean, padded = _setup(make_pendulum_mc, case)
    state_of = lambda p: ModelState.create({}, {"policy": (policy, target)}).set("policy", p)  # noqa: E731
    _same(lambda st, b: st.get("policy", False).entropy(st, b), policy, state_of, clean, padded)


def test_padded_rows_do_not_reach_v_and_q_objectives(make_pendulum_mc):
    mc = make_pendulum_mc()
    states = jax.random.normal(jax.random.key(2), (N, 3))
    actions = jax.random.normal(jax.random.key(3), (N, 1))
    ll = jnp.zeros((N,))
    a = jnp.zeros((N,))
    tv, tq = jnp.sin(states[:, 0]), jnp.cos(states[:, 0])
    clean, padded = _pair(states, actions, ll, a, {"tv": tv, "tq": tq})
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc, jax.random.key(0))
        q = FullyQNeuralNet(mc, jax.random.key(0))
    for net in (v, q):
        _same(lambda st, b: st.tr_obj(b), net, lambda p: p, clean, padded)
