"""Modular TRPO and MPO on JAX."""

import os
from importlib.metadata import PackageNotFoundError, version

# Set before any import that loads JAX, so the first session does not preallocate.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    __version__ = version("abstract-rl")
except PackageNotFoundError:
    __version__ = "0.1.0"

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts.hydra_conf import make_config

__all__ = ["__version__", "make_config", "run_mpo_non_parametric", "run_trpo"]
