"""Hydra configuration of the benchmark runs.

``config/alg/<alg>.yaml`` holds the Pendulum-v1 defaults of TRPO and MPO, ``config/env/<env>.yaml`` the Gymnasium id,
the success threshold, the env-step budget and the overrides of the algorithm defaults (``all``, ``trpo``, ``mpo``).
``run_config`` merges them into the flat dictionary the training loops expect: algorithm defaults, then environment
overrides, then explicit ``alg.<key>=...`` command line overrides.

The directory is the copy installed next to this package (``abstract_rl/config``, included in the wheel) when that
file is present, and the repository ``config/`` when running from a checkout.
"""

import functools
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _resolve_config_dir() -> Path:
    packaged = Path(__file__).resolve().parents[1] / "config"
    if (packaged / "config.yaml").is_file():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "config"
    if (checkout / "config.yaml").is_file():
        return checkout
    raise FileNotFoundError("abstract_rl Hydra config was not found in the package or the repository checkout")


CONFIG_DIR = _resolve_config_dir()
ALGORITHMS = sorted(p.stem for p in (CONFIG_DIR / "alg").glob("*.yaml"))
ALGORITHMS_SB3 = ["ppo", "trpo"]


def load(overrides: list[str] | None = None) -> DictConfig:
    """Compose ``config.yaml`` with Hydra command line style overrides, e.g. ``["env=mountaincar", "alg=mpo"]``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides or []))


def run_config(cfg: DictConfig, overrides: list[str] | None = None) -> dict[str, Any]:
    """Flat run configuration of a composed config; ``overrides`` are the command line overrides of ``cfg``."""
    alg: dict[str, Any] = OmegaConf.to_container(cfg.alg, resolve=True)  # type: ignore[assignment]
    name = str(alg.pop("name"))
    # mpo_paper is the paper preset of MPO, so it inherits that environment's MPO overrides.
    env_key = "mpo" if name == "mpo_paper" else name
    spec: dict[str, Any] = OmegaConf.to_container(cfg.env.overrides, resolve=True)  # type: ignore[assignment]
    conf: dict[str, Any] = {**alg, **(spec.get("all") or {}), **(spec.get(env_key) or {})}
    # explicit alg.<key>=value overrides of the command line take precedence over the environment table
    for item in overrides or []:
        key = item.lstrip("+~").partition("=")[0]
        if key.startswith("alg.") and key.count(".") == 1:
            conf[key[4:]] = alg[key[4:]]
    # null at the top level means the environment `steps` budget. An environment override or a
    # command-line `max_env_steps=` replaces it. The training loop stops at this count.
    if conf.get("max_env_steps") is None:
        conf["max_env_steps"] = int(cfg.env.steps)
    if cfg.max_env_steps is not None:
        conf["max_env_steps"] = int(cfg.max_env_steps)
    conf.update({"env_name": cfg.env.id, "seed": cfg.seed, "data_root": cfg.data_root})
    return conf


def make_config(env: str, alg: str, seed: int = 1, overrides: list[str] | None = None) -> dict[str, Any]:
    """Run configuration of an environment (group name or Gymnasium id) and algorithm."""
    ov = [f"env={env_group(env)}", f"alg={alg}", f"seed={seed}", *(overrides or [])]
    return run_config(load(ov), ov)


@functools.cache
def _envs() -> dict[str, DictConfig]:
    groups = sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))
    return {g: load([f"env={g}"]).env for g in groups}


def env_group(env: str) -> str:
    """Config group name of an environment given by group name or Gymnasium id."""
    for group, conf in _envs().items():
        if env in (group, conf.id):
            return group
    raise KeyError(f"unknown environment {env}, available: {', '.join(_envs())}")


_ORDER = [
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "HalfCheetah-v5",
    "Humanoid-v5",
]


def environments() -> list[str]:
    """Gymnasium ids of all configured environments (benchmark order first)."""
    ids = [c.id for c in _envs().values()]
    return [e for e in _ORDER if e in ids] + sorted(e for e in ids if e not in _ORDER)


def success() -> dict[str, float]:
    """Return threshold from which an environment counts as solved, by Gymnasium id."""
    return {c.id: float(c.success) for c in _envs().values()}


def steps() -> dict[str, int]:
    """Env-step budget per Gymnasium id (SB3 baselines, cut-off of the comparison)."""
    return {c.id: int(c.steps) for c in _envs().values()}
