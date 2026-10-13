"""Train MPO (non-parametric E-step, retrace critic) on Pendulum-v1.

Run with ``uv run python -m abstract_rl.scripts.mpo_pendulum``. Equivalent to
``python -m abstract_rl.scripts.run env=pendulum alg=mpo``, the configuration lives in ``config/``.
"""

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.scripts.hydra_conf import make_config

run_config = make_config("pendulum", "mpo")

if __name__ == "__main__":
    run_mpo_non_parametric(run_config)
