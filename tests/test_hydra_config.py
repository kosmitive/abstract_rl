import gymnasium as gym
import pytest

from abstract_rl.scripts.hydra_conf import CONFIG_DIR, load, run_dict

ENVS = sorted(path.stem for path in (CONFIG_DIR / "env").glob("*.yaml"))
ALGS = ("mpo", "trpo")


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("alg", ALGS)
def test_composed_config_names_a_registered_environment(env, alg):
    conf = run_dict(load([f"env={env}", f"alg={alg}"]))
    assert gym.spec(conf["env_name"]).id == conf["env_name"]


def test_deliberate_environment_overrides():
    mountain_mpo = run_dict(load(["env=mountaincar", "alg=mpo"]))
    assert mountain_mpo["policy_cov_type"] == "fixed"
    assert mountain_mpo["policy_sigma"] == 2.0
    assert mountain_mpo["max_env_steps"] == 400_000

    assert run_dict(load(["env=pendulum", "alg=trpo"]))["env_discount"] == 0.95
    assert run_dict(load(["env=pendulum", "alg=mpo"]))["rnk_eps"] == 0.05


def test_trpo_budgets_for_halfcheetah_and_inverted_double_pendulum():
    halfcheetah = run_dict(load(["env=halfcheetah", "alg=trpo"]))
    assert halfcheetah["obs_norm"] is True
    assert halfcheetah["num_epochs"] == 300
    assert halfcheetah["policy_batch_size"] == 4000
    assert halfcheetah["trpo_tr_eps"] == 0.02
    assert halfcheetah["val_l2_reg"] == 0.0

    inverted = run_dict(load(["env=inverted_double_pendulum", "alg=trpo"]))
    assert inverted["num_epochs"] == 75
    assert inverted["policy_batch_size"] == 8000
    assert inverted["trpo_tr_eps"] == 0.005
    assert inverted["val_l2_reg"] == 0.0
    assert inverted["obs_norm"] is False


def test_command_line_overrides_beat_environment_defaults():
    conf = run_dict(load(["env=mountaincar", "alg=mpo", "seed=3", "alg.policy_sigma=1.5", "data_root=/tmp/x"]))
    assert conf["policy_sigma"] == 1.5 and conf["policy_cov_type"] == "fixed"
    assert conf["seed"] == 3 and conf["data_root"] == "/tmp/x" and conf["max_env_steps"] == 400_000
    assert "max_env_steps" not in run_dict(load(["env=pendulum", "alg=trpo"]))
