"""Train TRPO on Pendulum-v1: ``uv run python -m abstract_rl.scripts.trpo_pendulum``.

Equivalent to ``python -m abstract_rl.scripts.run env=pendulum alg=trpo``, the configuration lives in ``config/``.
"""

from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts.hydra_conf import make_config

run_config = make_config("pendulum", "trpo")

if __name__ == "__main__":
    run_trpo(run_config)
