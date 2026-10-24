"""The M-step in compiled chunks gives the parameters and the log lines of one compiled step at a time."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    M_STEPS_PER_CALL,
    MPOAlgorithm,
)
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.mixed import set_device
from test_torch_parity import build_mpo


def _run(root, fused, capsys):
    mc = build_mpo(root)
    mx.random.seed(5)
    np.random.seed(5)  # noqa: NPY002
    tc = mc.env.execute_policy(mc.get("policy", target=False), 40, 300)
    rng = np.random.default_rng(1)
    for traj in tc:
        traj["add_act"] = np.tanh(rng.standard_normal((len(traj), 20))).astype(np.float32)
        traj["grad_ranking"] = np.log(rng.dirichlet(np.ones(20), len(traj))).astype(np.float32)
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    capsys.readouterr()
    num_steps = 2 * M_STEPS_PER_CALL + 7
    np.random.seed(0)  # noqa: NPY002
    if fused:
        alg.perform_mult_steps(num_steps, tc, 64)
    else:
        alg.sample_v()
        alg.cli.empty().print(
            alg.titlestr % ("k", "objective", *[f"lm_{i} x constr_{i}" for i in range(len(alg.lagrange_mults))])
        )
        replay = tc.flatten(["add_act", "grad_ranking"])
        for s in range(num_steps):
            alg.perform_step(s, Batch.from_tree(replay.sample(64)))
    log = capsys.readouterr().out
    params = {k: np.array(v) for k, v in tree_flatten(mc.get("policy", target=False).parameters())}
    lms = [np.array(v) for v in alg.lagrange_mults]
    mc.env.close()
    return params, lms, log


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_chunked_m_step_matches_one_step_at_a_time(tmp_path, capsys, device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("needs MLX with Metal (Apple silicon)")
    set_device(device)
    try:
        _check(tmp_path, capsys)
    finally:
        set_device("cpu")


def _check(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    fused = _run(tmp_path / "a", True, capsys)
    stepped = _run(tmp_path / "b", False, capsys)
    assert fused[0].keys() == stepped[0].keys()
    for k in fused[0]:
        assert np.array_equal(fused[0][k], stepped[0][k]), k
    assert all(np.array_equal(a, b) for a, b in zip(fused[1], stepped[1], strict=True))
    assert fused[2] == stepped[2]
    # the title and the steps 0, 10 and 20
    assert fused[2].count("\n") >= 4
