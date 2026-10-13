import numpy as np

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.run.run_trpo import run_trpo


def _check_returns(returns, num_epochs):
    assert returns.shape == (num_epochs, 2)
    assert np.all(np.isfinite(returns))
    # Pendulum-v1 rewards are in [-16.3, 0] and episodes last 200 steps
    assert np.all(returns[:, 0] <= 0)
    assert np.all(returns[:, 0] >= -16.3 * 200)


def test_trpo_pendulum_smoke(trpo_conf, tmp_path):
    returns = run_trpo(trpo_conf)
    _check_returns(returns, trpo_conf["num_epochs"])
    assert list(tmp_path.glob("Pendulum-v1/trpo/*/model/last/policy.data"))


def test_mpo_pendulum_smoke(mpo_conf, tmp_path):
    returns = run_mpo_non_parametric(mpo_conf)
    _check_returns(returns, mpo_conf["num_epochs"])
    assert list(tmp_path.glob("Pendulum-v1/mpo/*/model/last/policy.data"))


def test_eval_balance_runs_on_trained_policy(trpo_conf, tmp_path):
    from abstract_rl.scripts import eval_balance

    run_trpo(trpo_conf)
    (run_dir,) = tmp_path.glob("Pendulum-v1/trpo/*")
    out = tmp_path / "eval"
    code = eval_balance.main([str(run_dir), "--episodes", "1", "--min-balanced", "0", "--out", str(out), "--gif"])
    assert code == 0
    assert list(out.glob("*_theta.png"))
    assert list(out.glob("*.gif"))


def _check_multi_dimensional(returns, num_epochs):
    assert returns.shape == (num_epochs, 2)
    assert np.all(np.isfinite(returns))


def test_trpo_with_two_dimensional_actions_smoke(trpo_conf, tmp_path):
    trpo_conf.update({"env_name": "TestPointMass-v0", "max_steps": 30, "policy_batch_size": 300})
    _check_multi_dimensional(run_trpo(trpo_conf), trpo_conf["num_epochs"])
    assert list(tmp_path.glob("TestPointMass-v0/trpo/*/model/last/policy.data"))


def test_mpo_with_two_dimensional_actions_smoke(mpo_conf, tmp_path):
    mpo_conf.update({"env_name": "TestPointMass-v0", "max_steps": 30, "env_sample": 300})
    _check_multi_dimensional(run_mpo_non_parametric(mpo_conf), mpo_conf["num_epochs"])
    assert list(tmp_path.glob("TestPointMass-v0/mpo/*/model/last/policy.data"))
