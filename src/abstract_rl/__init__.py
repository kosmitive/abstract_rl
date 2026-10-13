"""Modular TRPO and MPO on JAX."""

import os

# Keep JAX from reserving most of the machine's RAM up front. CPU is the default
# platform via the jax[cpu] wheel; a CUDA install can still take over when present.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
