"""The opt-in bound on the M-step Lagrange multiplier holds when the KL overshoots, the default is exp(lm)."""

import math

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MULTIPLIER_EPS,
    MPOAlgorithm,
)
from abstract_rl.scripts.hydra_conf import load, run_dict
from test_torch_parity import build_mpo


def _alg(root, lm_bound, trust_region):
    mc = build_mpo(root)
    mx.random.seed(5)
    np.random.seed(5)  # noqa: NPY002
    tc = mc.env.execute_policy(mc.get("policy", target=False), 20, 300)
    rng = np.random.default_rng(1)
    for traj in tc:
        traj["add_act"] = np.tanh(rng.standard_normal((len(traj), 20))).astype(np.float32)
        traj["grad_ranking"] = np.log(rng.dirichlet(np.ones(20), len(traj))).astype(np.float32)
    conf = mc["conf"]
    conf["mpo_lm_bound"] = lm_bound
    conf["mpo_trust_regions"] = [trust_region]
    conf["mpo_inner_lr"] = 0.1
    with conf.ns("mpo"):
        alg = MPOAlgorithm(mc)
    return mc, alg, tc


def _lm_after_steps(root, lm_bound, trust_region, overshoot):
    mc, alg, tc = _alg(root, lm_bound, trust_region)
    if overshoot:
        # move the target policy far away, so the KL stays far above the trust region during all steps
        target = mc.get("policy", target=True)
        target.update(tree_map(lambda p: p + 1.0, target.parameters()))
    np.random.seed(0)  # noqa: NPY002
    alg.perform_mult_steps(100, tc, 64)
    lm = alg.lagrange_mults[0]
    mult = alg.multiplier(lm)
    mc.env.close()
    return float(lm.item()), float(mult.item())


def test_default_is_unbounded_exp(tmp_path):
    _, alg, _ = _alg(tmp_path, None, 0.01)
    assert alg.lm_bound is None
    lm = mx.array([50.0])
    assert alg.multiplier(lm).item() == mx.exp(lm).item()
    assert run_dict(load(["env=pendulum", "alg=mpo"]))["mpo_lm_bound"] is None


def test_bounded_multiplier_is_softplus():
    lm = mx.array([-30.0, 0.0, 18.0])
    alg = MPOAlgorithm.__new__(MPOAlgorithm)
    alg.lm_bound = (-18.0, 18.0)
    expected = [math.log1p(math.exp(v)) + MULTIPLIER_EPS for v in (-30.0, 0.0, 18.0)]
    np.testing.assert_allclose(np.array(alg.multiplier(lm)), expected, rtol=1e-6)


def test_bound_holds_under_a_kl_overshoot(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    free_lm, _ = _lm_after_steps(tmp_path / "a", None, 1e-6, True)
    lm, mult = _lm_after_steps(tmp_path / "b", [-18.0, 2.0], 1e-6, True)
    # the overshoot pushes the unbounded lm past the bound, the bounded one stops at it
    assert free_lm > 2.0
    assert lm == 2.0
    assert math.isclose(mult, math.log1p(math.exp(2.0)) + MULTIPLIER_EPS, rel_tol=1e-6)


def test_bound_holds_from_below(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    # a trust region far above the KL drives lm down
    free_lm, _ = _lm_after_steps(tmp_path / "a", None, 100.0, False)
    lm, mult = _lm_after_steps(tmp_path / "b", [0.5, 18.0], 100.0, False)
    assert free_lm < 0.5
    assert lm == 0.5
    assert mult > 0
