from abstract_rl.data_structures.abstract_conf.hierarchical_conf import SharedConf
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration


def test_namespaces():
    conf = SharedConf({"policy_lr": 0.1, "seed": 3})
    assert conf["seed"] == 3
    with conf.ns("policy"):
        assert conf["lr"] == 0.1
        assert "lr" in conf
        assert conf.get("missing", 5) == 5
        assert conf.get_root("seed") == 3
    assert conf["policy_missing"] == 5


def test_model_configuration_creates_run_dirs(tmp_path):
    mc = ModelConfiguration({"env_name": "Pendulum-v1", "alg": "trpo", "data_root": str(tmp_path)})
    assert mc["conf"]["reload"] is True
    for d in (mc.run_root, mc.model_root, mc.plot_root, mc.data_dir):
        assert (tmp_path / d.removeprefix(str(tmp_path) + "/")).is_dir()


def test_concurrent_runs_tolerate_existing_directories(tmp_path, monkeypatch):
    # emulate two runs racing for the same directories: the existence checks of the second run happen "before" the first
    # run creates them, and both draw the same time stamp
    from datetime import datetime

    from abstract_rl.data_structures.abstract_conf import model_configuration

    stamps = iter(
        [datetime(2026, 1, 1, 12, 0, 0, 1), datetime(2026, 1, 1, 12, 0, 0, 1), datetime(2026, 1, 1, 12, 0, 0, 2)]
    )

    class FakeDatetime:
        @staticmethod
        def now():
            return next(stamps)

    monkeypatch.setattr(model_configuration, "exists", lambda path: False)
    monkeypatch.setattr(model_configuration, "datetime", FakeDatetime)
    conf = {"env_name": "Pendulum-v1", "alg": "trpo", "data_root": str(tmp_path)}
    first = ModelConfiguration(dict(conf)).run_root
    second = ModelConfiguration(dict(conf)).run_root
    assert first != second
