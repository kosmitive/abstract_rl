"""SB3 TD3 baseline: zoo hyperparameters and a short Pendulum run."""

import json
from pathlib import Path

import numpy as np
import pytest

from abstract_rl.scripts import run_baselines
from abstract_rl.scripts.run_baselines import ZOO, parse_hp, resolve_hyper

MUJOCO_TD3 = (
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Pusher-v5",
)


def test_algorithms_drop_ppo() -> None:
    """The SB3 baseline list is TD3 and TRPO."""
    assert run_baselines.ALGORITHMS_SB3 == ["td3", "trpo"]
    assert all("ppo" not in algorithms for algorithms in ZOO.values())


def test_td3_zoo_entries() -> None:
    """TD3 hyperparameters match the rl-baselines3-zoo td3.yml entries used by this benchmark."""
    pendulum = ZOO["Pendulum-v1"]["td3"]
    assert pendulum["gamma"] == 0.98
    assert pendulum["buffer_size"] == 200_000
    assert pendulum["learning_starts"] == 10_000
    assert pendulum["noise_type"] == "normal"
    assert pendulum["noise_std"] == 0.1
    assert pendulum["learning_rate"] == 1e-3
    assert pendulum["policy_kwargs"]["net_arch"] == [400, 300]
    assert ZOO["Pendulum-v1"]["trpo"]["gamma"] == 0.9

    mountain = ZOO["MountainCarContinuous-v0"]["td3"]
    assert mountain["noise_type"] == "ornstein-uhlenbeck"
    assert mountain["noise_std"] == 0.5
    assert mountain["learning_rate"] == 1e-3
    assert mountain["batch_size"] == 256
    assert mountain["policy_kwargs"]["net_arch"] == [400, 300]

    for env_id in MUJOCO_TD3:
        hyper = ZOO[env_id]["td3"]
        assert hyper["learning_starts"] == 10_000
        assert hyper["noise_type"] == "normal"
        assert hyper["noise_std"] == 0.1
        assert hyper["learning_rate"] == 1e-3
        assert hyper["batch_size"] == 256
        assert hyper["policy_kwargs"]["net_arch"] == [400, 300]
        assert "gamma" not in hyper

    swimmer = ZOO["Swimmer-v5"]["td3"]
    assert swimmer["gamma"] == 0.9999
    assert swimmer["batch_size"] == 256
    assert swimmer["noise_type"] == "normal"


def test_hp_overrides_are_applied_on_top_of_the_zoo_entry() -> None:
    """``--hp`` parses Python literals and replaces zoo keys without dropping the rest."""
    assert parse_hp(["learning_rate=1e-4", "noise_type=normal", "net_arch=[64, 64]"]) == {
        "learning_rate": 1e-4,
        "noise_type": "normal",
        "net_arch": [64, 64],
    }
    hyper = resolve_hyper(
        "Pendulum-v1",
        "td3",
        zoo=False,
        hp=["learning_starts=10", "policy_kwargs={'net_arch': [32, 32]}", "use_sde=False"],
        hyper_json="{}",
    )
    assert hyper["learning_starts"] == 10
    assert hyper["gamma"] == 0.98
    assert hyper["noise_std"] == 0.1
    assert hyper["policy_kwargs"] == {"net_arch": [32, 32]}
    assert hyper["use_sde"] is False
    # TRPO keeps library defaults unless --zoo is set.
    assert resolve_hyper("Pendulum-v1", "trpo", zoo=False, hp=["n_steps=128"], hyper_json="{}") == {"n_steps": 128}
    assert resolve_hyper("Pendulum-v1", "trpo", zoo=True, hp=["n_steps=128"], hyper_json="{}")["gamma"] == 0.9


def test_action_noise_is_sized_to_the_action_dim() -> None:
    """``normal`` and ``ornstein-uhlenbeck`` become SB3 noise objects of the action dimension."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    from abstract_rl.scripts.run_baselines import make_action_noise

    kept = {"learning_rate": 1e-3, "noise_type": "normal", "noise_std": 0.1}
    normal = make_action_noise(kept, 3)
    assert isinstance(normal, NormalActionNoise)
    assert np.allclose(normal._sigma, [0.1, 0.1, 0.1])
    assert "noise_type" not in kept and "noise_std" not in kept
    assert kept["learning_rate"] == 1e-3

    ou = make_action_noise({"noise_type": "ornstein-uhlenbeck", "noise_std": [0.5]}, 1)
    assert isinstance(ou, OrnsteinUhlenbeckActionNoise)
    assert ou._sigma.shape == (1,)
    assert np.allclose(ou._sigma, 0.5)
    assert make_action_noise({"learning_rate": 1e-3}, 2) is None


def test_td3_pendulum_writes_run_json(tmp_path: Path) -> None:
    """A few hundred TD3 steps on Pendulum-v1, with a small ``learning_starts``, write ``run.json``."""
    pytest.importorskip("stable_baselines3")
    run_baselines.main(
        [
            "Pendulum-v1",
            "td3",
            "--seed",
            "1",
            "--steps",
            "300",
            "--data-root",
            str(tmp_path),
            "--hp",
            "learning_starts=10",
            "--hp",
            "buffer_size=500",
            "--hp",
            "batch_size=32",
            "--hp",
            "policy_kwargs={'net_arch': [32, 32]}",
        ]
    )
    (path,) = tmp_path.glob("Pendulum-v1/sb3_td3/*/run.json")
    result = json.loads(path.read_text())
    assert result["env"] == "Pendulum-v1"
    assert result["alg"] == "sb3_td3"
    assert result["seed"] == 1
    assert result["steps"] == 300
    assert result["hyper"]["learning_starts"] == 10
    assert result["hyper"]["noise_type"] == "normal"
    assert result["hyper"]["noise_std"] == 0.1
    assert len(result["returns"]) == 20
    assert np.all(np.isfinite(result["returns"]))
    assert path.with_name("monitor.csv").is_file()
