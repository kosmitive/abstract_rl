import json

import numpy as np

from abstract_rl.run import run_mpo_non_parametric as module
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric


def _conf(mpo_conf, **overrides):
    return {**mpo_conf, "env_name": "TestPointMass-v0", "max_steps": 30, "env_sample": 120, "val_opt_steps": 5,
            "policy_opt_steps": 5, "tc_size": 600, **overrides}


def test_returns_per_seed_and_epoch_and_eta_log(mpo_conf, tmp_path):
    returns = run_mpo_non_parametric(_conf(mpo_conf, seeds=[1, 2], num_epochs=3))
    assert returns.shape == (2, 3, 2) and np.all(np.isfinite(returns))
    runs = sorted(tmp_path.glob("TestPointMass-v0/mpo/*"))
    assert sorted(json.loads((r / "run.conf").read_text())["seed"] for r in runs) == [1, 2]
    for r in runs:
        assert (r / "model/last/policy.eqx").exists() and (r / "model/last/q_network.eqx").exists()
        etas = np.loadtxt(r / "data/eta.txt")
        assert etas.shape == (3,) and np.all(etas > 0)


def test_epoch_is_traced_once(mpo_conf):
    module.EPOCH_TRACES.clear()
    run_mpo_non_parametric(_conf(mpo_conf, num_epochs=3))
    assert module.EPOCH_TRACES == [120]


def test_memory_respects_tc_size(mpo_conf):
    returns = run_mpo_non_parametric(_conf(mpo_conf, num_epochs=8, tc_size=200))
    assert returns.shape == (1, 8, 2)
    assert module.LAST_MEMORY_SIZES and max(module.LAST_MEMORY_SIZES) <= 200 + 30
