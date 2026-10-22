import json

import numpy as np

from abstract_rl.run import run_trpo as run_trpo_module
from abstract_rl.run.run_trpo import run_trpo


def _conf(trpo_conf, **overrides):
    return {
        **trpo_conf,
        "env_name": "TestPointMass-v0",
        "max_steps": 30,
        "policy_batch_size": 120,
        "val_opt_steps": 5,
        **overrides,
    }


def test_returns_per_seed_and_epoch(trpo_conf, tmp_path):
    returns = run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=3))
    assert returns.shape == (2, 3, 2) and np.all(np.isfinite(returns))
    runs = sorted(tmp_path.glob("TestPointMass-v0/trpo/*"))
    assert len(runs) == 2
    assert sorted(json.loads((r / "run.conf").read_text())["seed"] for r in runs) == [1, 2]
    for r in runs:
        assert (r / "model/last/policy.eqx").exists() and (r / "model/best/policy.eqx").exists()
        assert len((r / "data/sreward.txt").read_text().split()) == 3 * 2


def test_epoch_is_traced_once(trpo_conf):
    run_trpo_module.EPOCH_TRACES.clear()
    run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=3))
    assert run_trpo_module.EPOCH_TRACES == [120]


def test_vmapped_seeds_match_single_seed_runs(trpo_conf):
    both = run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=2))
    alone = run_trpo(_conf(trpo_conf, seeds=[2], num_epochs=2))
    np.testing.assert_allclose(both[1], alone[0], rtol=1e-3, atol=1e-3)


def test_env_step_budget_shortens_the_last_epoch(trpo_conf):
    returns = run_trpo(_conf(trpo_conf, num_epochs=10, max_env_steps=300))
    assert returns.shape[1] == 3  # 120 + 120 + a shortened third epoch reach the 300 step budget


def test_epochs_per_call(trpo_conf):
    returns = run_trpo(_conf(trpo_conf, num_epochs=4, epochs_per_call=2))
    assert returns.shape == (1, 4, 2) and np.all(np.isfinite(returns))
