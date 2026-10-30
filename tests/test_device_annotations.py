"""Annotations, the temperature dual and the ranking stay on the device; checked against the numpy version."""

import mlx.core as mx
import numpy as np
import pytest

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.mixed import set_device, to_mx
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from test_batched_operators import _trajectories
from test_torch_parity import build_mpo

# float32 ranking on the device against the numpy one: the largest deviation measured was 4.8e-7 on the CPU and on Metal
RANKING_ATOL = 1e-5


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_annotations_dual_and_ranking_on_the_device(tmp_path, device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("needs MLX with Metal (Apple silicon)")
    set_device(device)
    try:
        _check(tmp_path)
    finally:
        set_device("cpu")


def _check(tmp_path):
    mc = build_mpo(tmp_path)
    tc = TrajectoryCollection(mc.env, 10_000)
    tc.extend(_trajectories(mc.env, np.random.default_rng(0)))
    mc.add_main("tc", tc)
    conf = mc["conf"]
    with conf.ns("res"):
        res = ResampleOperator(mc)
    with conf.ns("rnk"):
        rnk = VariationalRankingOperator(mc)
    eta0 = rnk.b.eta[0]

    mx.random.seed(7)
    res.transform_all(tc)
    trajectories = tc.trajectories()
    for k in ["add_act", "add_q", "add_pi"]:
        # the stored rows are views of the flat array, which the collection returns without a copy
        assert isinstance(trajectories[0][k], mx.array)
        assert tc.annotation(k) is tc.annotation(k)
        np.testing.assert_array_equal(
            np.array(tc.annotation(k)), np.concatenate([np.array(t[k]) for t in trajectories])
        )

    # the dual sees the same q values as with the numpy concatenation
    q_np = np.concatenate([np.array(t["add_q"]) for t in trajectories])
    rnk.hook()
    expected_eta = np.array(
        find_optimal_temperature(
            to_mx(q_np), rnk.b.non_parametric_variational_eps, eta0, rnk.b.non_parametric_variational_lr
        )
    )
    np.testing.assert_array_equal(rnk.b.eta, expected_eta)

    # the ranking against the numpy log-sum-exp of the earlier version
    rnk.transform_all(tc)
    samples = q_np / rnk.b.eta
    max_s = np.max(samples, axis=1, keepdims=True)
    expected = samples - (np.log(np.sum(np.exp(samples - max_s), axis=1, keepdims=True)) + max_s)
    got = np.array(tc.annotation("grad_ranking"))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=0, atol=RANKING_ATOL)

    # the M-step replay uses the device arrays, equal to the copy of the per-trajectory rows
    replay = tc.flatten(["add_act", "grad_ranking"])
    for k in ["add_act", "grad_ranking"]:
        np.testing.assert_array_equal(
            np.array(replay.tree["annotations"][k]), np.concatenate([np.array(t[k]) for t in trajectories])
        )

    # a trajectory whose annotation is replaced invalidates the flat array
    trajectories[1]["add_q"] = np.zeros_like(np.array(trajectories[1]["add_q"]))
    n0 = len(trajectories[0])
    assert np.all(np.array(tc.annotation("add_q"))[n0 : n0 + len(trajectories[1])] == 0)
    mc.env.close()
