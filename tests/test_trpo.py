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


def test_failed_line_search_keeps_the_parameters(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    # zero advantages give no expected improvement, every step is rejected
    zero = Batch(
        batch.states,
        batch.actions,
        batch.log_likelihoods,
        batch.rewards,
        batch.next_states,
        batch.dones,
        {"a": jnp.zeros_like(batch["a"]) + jnp.arange(len(batch)) * 0.0},
    )
    new_state, m = alg.perform_tr_sgd(state, 0, zero, zero)
    assert not bool(m["accepted"])
    np.testing.assert_allclose(
        named_parameters(new_state.get("policy"))["cov"], named_parameters(state.get("policy"))["cov"]
    )


def test_objective_gradient_flows_through_the_state(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    import equinox as eqx

    grads = eqx.filter_grad(lambda p: alg.tr_obj(state.set("policy", p), batch))(state.get("policy"))
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree.leaves(grads))
