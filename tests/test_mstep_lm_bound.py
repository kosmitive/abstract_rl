"""The opt-in bound on the M-step Lagrange multiplier holds when the KL overshoots, the default is exp(lm)."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MULTIPLIER_EPS,
    MPOAlgorithm,
)
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.hydra_conf import load, run_dict
from abstract_rl.scripts.mpo_pendulum import run_config

MPO_KEYS = [k for k in run_config if k.startswith("mpo_")]


def _memory() -> TrajectoryCollection:
    """One valid episode of 40 steps, with the annotations the M-step samples."""
    memory = TrajectoryCollection.memory(None, 64, 3, 1).with_annotations(
        {k: jax.ShapeDtypeStruct((20,), jnp.float32) for k in ("add_act", "grad_ranking", "add_pi")}
    )
    lane = memory.trajectories()
    k1, k2, k3 = jax.random.split(jax.random.key(3), 3)
    lane = (
        lane.set("add_act", jax.random.normal(k1, lane["add_act"].shape))
        .set("grad_ranking", jax.nn.log_softmax(jax.random.normal(k2, lane["grad_ranking"].shape), axis=-1))
        .replace(
            states=jax.random.normal(k3, lane.states.shape),
            mask=lane.mask.at[:, :40].set(True),
            ends=lane.ends.at[:, 39].set(True),
        )
    )
    return memory.with_trajectories(lane)


def _alg(mc, lm_bound, trust_region):
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    mc.add_syncable("policy", policy, 1)
    mc["conf"]["mpo_lm_bound"] = lm_bound
    mc["conf"]["mpo_trust_regions"] = [trust_region]
    mc["conf"]["mpo_inner_lr"] = 0.1
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    state = ModelState.create(alg.init_state(policy), {"policy": (policy, policy)})
    return alg, state


def _lm_after_steps(mc, lm_bound, trust_region, overshoot):
    alg, state = _alg(mc, lm_bound, trust_region)
    if overshoot:
        # move the target policy far away, so the KL stays far above the trust region during the steps
        target = state.get("policy", target=True)
        shifted = jax.tree.map(lambda x: x + 1.0 if eqx.is_inexact_array(x) else x, target)
        state = state.set("policy", shifted, target=True)
    state, _ = jax.jit(alg.perform_mult_steps, static_argnums=(2, 4))(state, jax.random.key(0), 100, _memory(), 64)
    lm = state.get("lagrange_mults")
    return float(lm[0]), float(alg.multiplier(lm)[0])


def test_default_is_unbounded_exp(make_pendulum_mc):
    alg, state = _alg(make_pendulum_mc(alg="mpo", **{k: run_config[k] for k in MPO_KEYS}), None, 0.01)
    assert alg.lm_bound is None
    lm = jnp.array([0.0, 1.0, 47.0], jnp.float32)
    np.testing.assert_array_equal(np.asarray(alg.multiplier(lm)), np.asarray(jnp.exp(lm)))
    assert run_dict(load(["env=pendulum", "alg=mpo"]))["mpo_lm_bound"] is None
    assert math.isfinite(float(state.get("lagrange_mults")[0]))


def test_missing_bound_matches_explicit_null_bit_for_bit(make_pendulum_mc):
    """The default (key absent or null) stays on exp(lm): two runs from one init agree exactly."""
    keys = {k: run_config[k] for k in MPO_KEYS}
    bare, _ = _alg(make_pendulum_mc(alg="mpo", **keys), None, 0.01)
    # rebuild the bare algorithm without ever writing the key, from the same policy init
    mc = make_pendulum_mc(alg="mpo", **{k: v for k, v in keys.items() if k != "mpo_lm_bound"})
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    mc.add_syncable("policy", policy, 1)
    with mc["conf"].ns("mpo"):
        omitted = MPOAlgorithm(mc)
    assert omitted.lm_bound is None and bare.lm_bound is None
    lm = jnp.array([-4.0, 0.0, 12.0], jnp.float32)
    np.testing.assert_array_equal(np.asarray(omitted.multiplier(lm)), np.asarray(bare.multiplier(lm)))
    np.testing.assert_array_equal(np.asarray(omitted.multiplier(lm)), np.asarray(jnp.exp(lm)))


def test_bounded_multiplier_is_softplus():
    lm = jnp.array([-30.0, 0.0, 18.0], jnp.float32)
    alg = MPOAlgorithm.__new__(MPOAlgorithm)
    alg.lm_bound = (-18.0, 18.0)
    expected = [math.log1p(math.exp(v)) + MULTIPLIER_EPS for v in (-30.0, 0.0, 18.0)]
    np.testing.assert_allclose(np.asarray(alg.multiplier(lm)), expected, rtol=1e-5, atol=1e-6)


def test_bound_holds_under_a_kl_overshoot(make_pendulum_mc):
    keys = {k: run_config[k] for k in MPO_KEYS}
    free_lm, _ = _lm_after_steps(make_pendulum_mc(alg="mpo", **keys), None, 1e-6, True)
    lm, mult = _lm_after_steps(make_pendulum_mc(alg="mpo", **keys), [-18.0, 2.0], 1e-6, True)
    # the overshoot pushes the unbounded lm past the bound, the bounded one stops at it
    assert free_lm > 2.0
    assert lm == 2.0
    assert math.isclose(mult, math.log1p(math.exp(2.0)) + MULTIPLIER_EPS, rel_tol=1e-5)


def test_bound_holds_from_below(make_pendulum_mc):
    keys = {k: run_config[k] for k in MPO_KEYS}
    # a trust region far above the KL drives lm down
    free_lm, _ = _lm_after_steps(make_pendulum_mc(alg="mpo", **keys), None, 100.0, False)
    lm, mult = _lm_after_steps(make_pendulum_mc(alg="mpo", **keys), [0.5, 18.0], 100.0, False)
    assert free_lm < 0.5
    assert lm == 0.5
    assert mult > 0
