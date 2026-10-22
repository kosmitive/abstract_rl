"""Train TRPO on Pendulum-v1: ``uv run python -m abstract_rl.scripts.trpo_pendulum``.

Equivalent to ``python -m abstract_rl.scripts.run env=pendulum alg=trpo``, the configuration lives in
``abstract_rl/conf``.
"""

from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts.hydra_conf import load, run_dict

run_config = run_dict(load(["env=pendulum", "alg=trpo"]))

if __name__ == "__main__":
    run_trpo(run_config)
