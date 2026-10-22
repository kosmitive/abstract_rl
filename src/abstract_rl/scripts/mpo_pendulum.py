"""Train MPO (non-parametric E-step, retrace critic) on Pendulum-v1.

Run with ``uv run python -m abstract_rl.scripts.mpo_pendulum``. Equivalent to
``python -m abstract_rl.scripts.run env=pendulum alg=mpo``, the configuration lives in ``abstract_rl/conf``.
"""

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.scripts.hydra_conf import load, run_dict

run_config = run_dict(load(["env=pendulum", "alg=mpo"]))

if __name__ == "__main__":
    run_mpo_non_parametric(run_config)
