"""Hydra configuration of the benchmark runs (``abstract_rl/conf``, shipped inside the package).

``conf/alg`` and ``conf/env`` are config groups. Shared environment overrides (``conf/tune_all``) and
per-algorithm overrides (``conf/tune``) are selected by defaults-list interpolation, ``${env}`` and
``${env}_${alg}``, and merged into the algorithm node. Command line ``alg.*`` overrides are applied by Hydra
on top of that. The training loops receive the algorithm node as a flat dictionary, with the environment id,
seed and data root filled in.
"""

from functools import cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"


def load(overrides: list[str] | None = None) -> DictConfig:
    """Compose ``config.yaml`` with Hydra overrides, e.g. ``["env=mountaincar", "alg=mpo"]``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides or []))


def run_dict(cfg: DictConfig) -> dict[str, Any]:
    """Flat run configuration of a composed config."""
    conf: dict[str, Any] = OmegaConf.to_container(cfg.alg, resolve=True)  # type: ignore[assignment]
    conf.pop("name", None)
    # a set top-level budget wins over the one merged in from the environment tune file
    if cfg.max_env_steps is not None:
        conf["max_env_steps"] = cfg.max_env_steps
    conf["env_name"] = cfg.env.id
    conf["seed"] = cfg.seed
    conf["data_root"] = cfg.data_root
    return conf


@cache
def _envs() -> dict[str, DictConfig]:
    names = sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))
    return {name: load([f"env={name}"]).env for name in names}


def environments() -> list[str]:
    """Gymnasium ids in the benchmark order stored in ``config.yaml``."""
    return [str(item) for item in load().env_order]


def success() -> dict[str, float]:
    """Return threshold from which an environment counts as solved, by Gymnasium id."""
    return {str(conf.id): float(conf.success) for conf in _envs().values()}


def steps() -> dict[str, int]:
    """Env-step budget per Gymnasium id (SB3 baselines, cut-off of the comparison)."""
    return {str(conf.id): int(conf.steps) for conf in _envs().values()}
