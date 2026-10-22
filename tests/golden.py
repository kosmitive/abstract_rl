"""Access to the torch reference outputs (tests/fixtures/torch_golden.npz) and torch <-> Equinox parameter naming."""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

_GOLDEN = np.load(Path(__file__).parent / "fixtures" / "torch_golden.npz")
MLP_ATTRS = ("network", "net", "fc_net")


def golden(case: str) -> dict[str, np.ndarray]:
    """All arrays of one case, keyed by the name after ``<case>/``."""
    p = case + "/"
    return {k[len(p) :]: _GOLDEN[k] for k in _GOLDEN.files if k.startswith(p)}


def sub(d: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    p = prefix + "/"
    return {k[len(p) :]: v for k, v in d.items() if k.startswith(p)}


def named_parameters(module: Any) -> dict[str, np.ndarray]:
    """torch ``named_parameters()`` names of an Equinox model (or of its gradient pytree)."""
    out: dict[str, Any] = {}
    if getattr(module, "cov", None) is not None:
        out["cov"] = module.cov
    mlps = [(f"{a}.", getattr(module, a)) for a in MLP_ATTRS if getattr(module, a, None) is not None]
    if hasattr(module, "layers"):
        mlps = [("", module)]
    for prefix, mlp in mlps:
        for i, layer in enumerate(mlp.layers):
            out[f"{prefix}layer_{i}.weight"] = layer.weight
            out[f"{prefix}layer_{i}.bias"] = layer.bias
    return {k: np.asarray(v) for k, v in out.items()}


def load_named(module: Any, params: dict[str, np.ndarray]) -> Any:
    """Copy torch-named parameters into an Equinox model."""
    for name, value in params.items():
        parts = name.split(".")
        if parts == ["cov"]:
            module = eqx.tree_at(lambda m: m.cov, module, jnp.asarray(value))
            continue
        attr = None if parts[0].startswith("layer_") else parts[0]
        layer_name, field = parts[-2], parts[-1]
        i = int(layer_name.split("_")[1])

        def where(m, attr=attr, i=i, field=field):
            mlp = m if attr is None else getattr(m, attr)
            return getattr(mlp.layers[i], field)

        module = eqx.tree_at(where, module, jnp.asarray(value))
    return module
