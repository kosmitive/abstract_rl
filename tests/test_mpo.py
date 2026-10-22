import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.mpo_pendulum import run_config
from golden import golden, load_named, named_parameters, sub

MPO_KEYS = [k for k in run_config if k.startswith("mpo_")]
CASES = {
    "mpo_uni": {},
    "mpo_split": {
        "mpo_coordinate_ascent_objective": True,
        "mpo_coordinate_ascent_constraint": True,
        "mpo_trust_regions": [0.01, 0.02],
    },
    "mpo_diag": {"env_name": "TestPointMass-v0"},
}


@pytest.mark.parametrize("case", CASES)
def test_three_mpo_steps_match_torch(make_pendulum_mc, case):
    mc = make_pendulum_mc(alg="mpo", **{**{k: run_config[k] for k in MPO_KEYS}, **CASES[case]})
    g = golden(case)
    with mc["conf"].ns("policy"):
        fresh = select_policy(mc, jax.random.key(0))
    mc.add_syncable("policy", fresh, 1)
    policy, target = load_named(fresh, sub(g, "step0/params")), load_named(fresh, sub(g, "target/params"))
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    state = ModelState.create(alg.init_state(policy), {"policy": (policy, target)})
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(
        jnp.asarray(g["states"]),
        z,
        z,
        z,
        jnp.asarray(g["states"]),
        z,
        {
            "add_act": jnp.asarray(g["add_act"]),
            "grad_ranking": jnp.asarray(g["grad_ranking"]),
            "add_pi": jnp.zeros((n, 20)),
        },
    )
    np.testing.assert_allclose(state.get("lagrange_mults"), g["step0/lagrange"].reshape(-1))
    v = alg.sample_v(jax.random.key(0))
    step = jax.jit(lambda s, i: alg.perform_step(s, i, batch, v))
    for s in range(3):
        state, _ = step(state, s)
        got = named_parameters(state.get("policy"))
        for k, expected in sub(g, f"step{s + 1}/params").items():
            np.testing.assert_allclose(got[k], expected, rtol=1e-4, atol=1e-5, err_msg=f"{case} step {s + 1} {k}")
        np.testing.assert_allclose(
            state.get("lagrange_mults"), g[f"step{s + 1}/lagrange"].reshape(-1), rtol=1e-4, atol=1e-6
        )


def test_list_trust_regions_are_sampled_per_key(make_pendulum_mc):
    mc = make_pendulum_mc(alg="mpo", **{**{k: run_config[k] for k in MPO_KEYS}, "mpo_trust_regions": [[0.01, 0.05]]})
    with mc["conf"].ns("policy"):
        mc.add_syncable("policy", select_policy(mc, jax.random.key(0)), 1)
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    a, b = alg.sample_v(jax.random.key(0))[0], alg.sample_v(jax.random.key(1))[0]
    assert 0.01 <= float(a) <= 0.05 and float(a) != float(b)
