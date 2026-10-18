"""Train TRPO or MPO on one of the benchmark environments (Hydra entry point, configuration in ``config/``).

Usage::

    uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
    uv run --extra mujoco python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=2 alg.num_epochs=50
    uv run python -m abstract_rl.scripts.run --cfg job env=mountaincar alg=mpo   # print the composed config

Environments: pendulum, mountaincar, inverted_pendulum, inverted_double_pendulum, halfcheetah. ``alg.<key>=value``
overrides beat the environment tune files. Runs are written to ``<data_root>/<env id>/<alg>/<timestamp>/``.
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from abstract_rl.scripts.hydra_conf import CONFIG_DIR, run_dict


def train(alg: str, conf: dict[str, Any]) -> Any:
    """Run one training. Returns the logged ``[mean, std]`` return of every epoch."""
    if alg == "mpo":
        from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric

        return run_mpo_non_parametric(conf)
    from abstract_rl.run.run_trpo import run_trpo

    return run_trpo(conf)


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(str(cfg.alg.name), run_dict(cfg))


if __name__ == "__main__":
    main()
