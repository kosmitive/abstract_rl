import json
from pathlib import Path

import pytest

from abstract_rl.scripts import hydra_conf

# snapshot of make_config(env, alg, 1) and the SUCCESS / STEPS tables of the former scripts/configs.py
LEGACY = json.loads((Path(__file__).parent / "fixtures" / "legacy_configs.json").read_text())


@pytest.mark.parametrize("key", sorted(LEGACY["cfg"]))
def test_hydra_configs_equal_legacy_tables(key):
    env, alg, seed = key.split("|")
    conf = hydra_conf.make_config(env, alg, int(seed))
    assert conf.pop("data_root") == "data"
    assert conf["steps"] == hydra_conf.steps()[env]
    expected = dict(LEGACY["cfg"][key])
    assert set(conf) == set(expected)
    for k, v in expected.items():
        assert conf[k] == v and type(conf[k]) is type(v), k


def test_tables():
    assert hydra_conf.success() == LEGACY["SUCCESS"]
    assert hydra_conf.steps() == LEGACY["STEPS"]
    assert hydra_conf.environments() == LEGACY["ENV"]
    assert sorted(hydra_conf.ALGORITHMS) == sorted(LEGACY["ALG"])


def test_command_line_overrides_beat_environment_table():
    conf = hydra_conf.make_config("mountaincar", "mpo", 3, ["alg.policy_init_std=1.5", "data_root=/tmp/x"])
    assert conf["policy_init_std"] == 1.5 and conf["policy_cov_type"] == "fixed"
    assert conf["seed"] == 3 and conf["data_root"] == "/tmp/x" and conf["steps"] == 400_000
    assert hydra_conf.make_config("Pendulum-v1", "trpo")["steps"] == 800_000
    assert hydra_conf.make_config("Pendulum-v1", "mpo")["steps"] == 800_000
    overridden = hydra_conf.make_config("Pendulum-v1", "trpo", overrides=["env.steps=1234"])
    assert overridden["steps"] == 1234 and type(overridden["steps"]) is int
