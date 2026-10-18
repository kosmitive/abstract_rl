"""Eager and ``torch_compile`` training agree on a Pendulum-v1 smoke run."""

import warnings

import numpy as np

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.run.run_trpo import run_trpo


def _returns(runner, conf, root, compile):
    run_conf = {**conf, "torch_compile": compile, "data_root": str(root)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        returns = runner(run_conf)
    if compile:
        failed = [str(item.message) for item in caught if "torch.compile failed" in str(item.message)]
        assert not failed, failed
    return returns


def test_torch_compile_matches_eager_on_pendulum(trpo_conf, mpo_conf, tmp_path):
    for name, runner, conf in (
        ("trpo", run_trpo, trpo_conf),
        ("mpo", run_mpo_non_parametric, mpo_conf),
    ):
        eager = _returns(runner, conf, tmp_path / name / "eager", False)
        compiled = _returns(runner, conf, tmp_path / name / "compiled", True)
        assert eager.shape == compiled.shape
        assert np.allclose(eager, compiled, rtol=1e-4, atol=1e-5), name
