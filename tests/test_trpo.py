import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy
from golden import golden, load_named, named_parameters, sub


def _setup(make_pendulum_mc, case, env):
    mc = make_pendulum_mc(env_name=env)
    with mc["conf"].ns("policy"):
        policy = load_named(select_policy(mc, jax.random.key(0)), sub(golden(case), "before/params"))
    with mc["conf"].ns("trpo"):
        alg = TRPOAlgorithm(mc)
    g = golden(case)
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(
        jnp.asarray(g["states"]),
        jnp.asarray(g["actions"]),
        jnp.asarray(g["lls"]),
        z,
        jnp.asarray(g["states"]),
        z,
        {"a": jnp.asarray(g["a"])},
    )
    return alg, ModelState.create({}, {"policy": (policy, policy)}), batch, g


@pytest.mark.parametrize("case, env", [("trpo_uni", "Pendulum-v1"), ("trpo_diag", "TestPointMass-v0")])
def test_trpo_step_matches_torch(make_pendulum_mc, case, env):
    alg, state, batch, g = _setup(make_pendulum_mc, case, env)
    new_state, m = jax.jit(lambda s, b: alg.perform_tr_sgd(s, 0, b, b))(state, batch)
    got = named_parameters(new_state.get("policy"))
    before, after = sub(g, "before/params"), sub(g, "after/params")
    torch_accepted = any(not np.allclose(before[k], after[k]) for k in after)
    assert bool(m["accepted"]) == torch_accepted
    for k, expected in after.items():
        np.testing.assert_allclose(got[k], expected, rtol=1e-3, atol=1e-5, err_msg=k)
    # the target is untouched
    np.testing.assert_allclose(
        named_parameters(new_state.get("policy", True))["network.layer_0.weight"], before["network.layer_0.weight"]
    )


def _assert_parameters_unchanged(new_state, state):
    got, before = named_parameters(new_state.get("policy")), named_parameters(state.get("policy"))
    assert got.keys() == before.keys()
    for k in before:
        np.testing.assert_allclose(got[k], before[k], err_msg=k)


def test_line_search_with_nan_ratio_keeps_the_parameters(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    # zero advantages give a zero gradient and zero step (the expected improvement is 0), so the ratio is NaN and
    # every step is rejected
    zero = Batch(
        batch.states,
        batch.actions,
        batch.log_likelihoods,
        batch.rewards,
        batch.next_states,
        batch.dones,
        {"a": jnp.zeros_like(batch["a"])},
    )
    new_state, m = alg.perform_tr_sgd(state, 0, zero, zero)
    assert not bool(m["accepted"])
    _assert_parameters_unchanged(new_state, state)


def test_line_search_rejects_steps_outside_the_trust_region(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    original = alg.tr_distance
    # a distance far above 1.5 * tr_eps rejects every step with finite ratios
    alg.tr_distance = lambda s, b: original(s, b) + 100.0 * alg.tr_eps
    new_state, m = alg.perform_tr_sgd(state, 0, batch, batch)
    assert not bool(m["accepted"])
    assert int(m["step"]) == 10
    rows = np.asarray(m["rows"])
    assert rows.shape[0] == 10 and np.all(np.isfinite(rows))
    _assert_parameters_unchanged(new_state, state)


def test_objective_gradient_flows_through_the_state(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    import equinox as eqx

    grads = eqx.filter_grad(lambda p: alg.tr_obj(state.set("policy", p), batch))(state.get("policy"))
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree.leaves(grads))
