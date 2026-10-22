import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from golden import golden, load_named, sub

MPO_VAL = {"val_struct": [256, 256], "val_act_fn": ["elu", "elu", "none"]}


def _mpo_mc(make_pendulum_mc, **overrides):
    from abstract_rl.scripts.mpo_pendulum import run_config

    keys = (
        "res_add_acts",
        "ret_lambda",
        "ret_add_acts",
        "mem_bootstrap_steps",
        "rnk_eps",
        "rnk_lr",
        "rnk_init_eta",
        "rnk_max_steps",
        "env_discount",
    )
    return make_pendulum_mc(alg="mpo", **MPO_VAL, **{k: run_config[k] for k in keys}, **overrides)


def _models(mc):
    with mc["conf"].ns("policy"):
        policy = load_named(select_policy(mc, jax.random.key(0)), sub(golden("retrace_policy"), "params"))
    with mc["conf"].ns("val"):
        q = load_named(FullyQNeuralNet(mc, jax.random.key(0)), sub(golden("retrace_q"), "params"))
    return ModelState.create({}, {"policy": (policy, policy), "q_network": (q, q)})


def _lane(g):
    s = jnp.asarray(g["all_states"], jnp.float32)
    n = len(s) - 1
    ends = jnp.zeros(n, bool).at[-1].set(True)
    return Trajectory(
        s[:-1],
        s[1:],
        jnp.asarray(g["actions"], jnp.float32),
        jnp.asarray(g["rewards"], jnp.float32),
        jnp.asarray(g["rewards"][:, 0], jnp.float32),
        jnp.asarray(g["dones"], jnp.float32),
        jnp.asarray(g["log_likelihoods"], jnp.float32),
        ends,
        jnp.ones(n, bool),
    )


@pytest.mark.parametrize("case", ["truncated", "terminated"])
def test_resample_and_retrace_match_torch(make_pendulum_mc, monkeypatch, case):
    # the torch fixture replaced rsample by the mean; zero noise reproduces it
    monkeypatch.setattr(jax.random, "normal", lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype))
    mc = _mpo_mc(make_pendulum_mc)
    state = _models(mc)
    with mc["conf"].ns("res"):
        res = ResampleOperator(mc)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    g = golden(f"retrace_{case}")
    t = ret.transform(state, res.transform(state, _lane(g), jax.random.key(0)), jax.random.key(1))
    np.testing.assert_allclose(t["add_act"], g["add_act"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["add_q"], g["add_q"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(t["add_pi"], g["add_pi"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t["q"], g["q"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(t["tq"], g["tq"], rtol=1e-4, atol=1e-4)


def test_retrace_does_not_cross_episodes(make_pendulum_mc, monkeypatch):
    monkeypatch.setattr(jax.random, "normal", lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype))
    mc = _mpo_mc(make_pendulum_mc)
    state = _models(mc)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    g1, g2 = golden("retrace_terminated"), golden("retrace_truncated")
    both = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), _lane(g1), _lane(g2))
    t = ret.transform(state, both, jax.random.key(1))
    n1 = len(g1["rewards"])
    np.testing.assert_allclose(t["tq"][:n1], g1["tq"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t["tq"][n1:], g2["tq"], rtol=1e-4, atol=1e-4)


def test_eta_ignores_invalid_memory_rows(make_pendulum_mc):
    mc = _mpo_mc(make_pendulum_mc)
    with mc["conf"].ns("rnk"):
        rnk = VariationalRankingOperator(mc)
    q = jax.random.normal(jax.random.key(0), (1, 64, 20)) * 5
    memory = TrajectoryCollection.memory(None, 80, 3, 1).with_annotations(
        {"add_q": jax.ShapeDtypeStruct((20,), jnp.float32)}
    )
    t = memory.trajectories()
    t = t.set("add_q", t["add_q"].at[:, :64].set(q).at[:, 64:].set(1e6)).replace(mask=t.mask.at[:, :64].set(True))
    memory = memory.with_trajectories(t)
    state = ModelState.create(rnk.b.init_state(), {})
    eta = rnk.hook(state, memory).get("eta")
    from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature

    np.testing.assert_allclose(
        eta,
        find_optimal_temperature(q[0], rnk.b.non_parametric_variational_eps, 1.0, rnk.b.non_parametric_variational_lr),
        rtol=1e-5,
    )
    ranked = rnk.transform_all(ModelState.create({"eta": eta}, {}), memory).trajectories()["grad_ranking"]
    np.testing.assert_allclose(jnp.exp(ranked[0, :64]).sum(axis=1), 1.0, rtol=1e-5)
