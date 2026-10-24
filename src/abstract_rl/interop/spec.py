"""Layout of one flat Gaussian policy.

Weights are stored row-major ``[out, in]`` then bias, layer by layer, which is
how ``mlx.nn.Linear`` multiplies (``y = x @ W.T + b``). A state-independent
trainable ``log_sigma`` sits at the end of the vector. A frozen ``log_sigma``
is not in the vector: training must not move it, and the rollout receives it
as its own argument.

The flat order is not MLX's parameter-tree order (that tree lists ``cov``
first). Initialization copies tensors by name, and checkpoints are written
back out under the names ``select_policy`` saves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from abstract_rl.policy.continuous.diagonal_gaussian_policy import head_activations, sigma_is_softplus
from abstract_rl.policy.continuous.univariate_gaussian_policy import SIGMA_MIN, _sigma_head_is_softplus

# checkpoints and the JAX forward agree on these names
WEIGHT = "network.layers.{i}.weight"
BIAS = "network.layers.{i}.bias"


@dataclass(frozen=True)
class LayerSpec:
    """One linear layer inside the flat vector."""

    in_dim: int
    out_dim: int
    weight_offset: int
    bias_offset: int
    activation: tuple[str, ...]
    layer_norm: bool

    @property
    def n_weight(self) -> int:
        """Number of weight entries."""
        return self.in_dim * self.out_dim

    @property
    def n_bias(self) -> int:
        """Number of bias entries."""
        return self.out_dim


@dataclass(frozen=True)
class NamedTensor:
    """A checkpoint tensor sliced out of the flat vector, or a frozen constant."""

    name: str
    offset: int | None
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GaussianSpec:
    """Shapes, activations and offsets of a univariate or diagonal Gaussian policy."""

    obs_dim: int
    action_dim: int
    kind: str
    cov_type: str
    squash: bool
    layers: tuple[LayerSpec, ...]
    n_params: int
    log_sigma_offset: int | None
    sigma_is_softplus: bool
    action_center: tuple[float, ...]
    action_scale: tuple[float, ...]
    named: tuple[NamedTensor, ...]
    sigma_min: float = SIGMA_MIN


def _activation_names(act_fn: list[Any], n_layers: int, action_dim: int, kind: str) -> tuple[list[str], bool]:
    """Per-layer activation names and whether the network already softpluses sigma."""
    if kind == "diagonal":
        expanded, sigma_acts = head_activations(act_fn, action_dim)
        softplus = sigma_is_softplus(sigma_acts)
        names = list(expanded)
    else:
        names = list(act_fn)
        softplus = _sigma_head_is_softplus(names)
    if len(names) == 1:
        names = names * n_layers
    if len(names) != n_layers:
        raise ValueError(f"expected 1 or {n_layers} activations, got {len(act_fn)}")
    return names, softplus


def _as_names(entry: Any, out_dim: int) -> tuple[str, ...]:
    """One activation name, or one name per output."""
    if isinstance(entry, list | tuple):
        names = tuple(str(a) for a in entry)
        if len(names) != out_dim:
            raise ValueError(f"per-output activation has {len(names)} names for {out_dim} outputs")
        return names
    return (str(entry),)


def make_gaussian_spec(
    *,
    obs_dim: int,
    action_dim: int,
    kind: str,
    cov_type: str,
    structure: list[int],
    act_fn: list[Any],
    layer_norm: bool,
    squash: bool,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> GaussianSpec:
    """Build the flat layout for a policy configuration.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        kind: ``univariate`` or ``diagonal``.
        cov_type: ``state``, ``single`` or ``fixed``.
        structure: Hidden widths, excluding the input and the head.
        act_fn: The per-layer activation list from the config.
        layer_norm: Whether the first layer is normalized, as in ``MultilayerPerceptron``.
        squash: Whether samples are squashed with tanh into the action bounds.
        action_low: Lower bound of the action space.
        action_high: Upper bound of the action space.

    Returns:
        The spec. Frozen ``log_sigma`` is not allocated; the caller stores it beside the vector.
    """
    match cov_type:
        case "state":
            out_dim = 2 * action_dim if kind == "diagonal" else 2
        case "single" | "fixed":
            out_dim = action_dim
        case "full":
            raise NotImplementedError(
                "cov_type full is not supported: the MLX port has no Cholesky gradient, "
                "use policy_cov_type=state, single or fixed"
            )
        case _:
            raise NotImplementedError(f"unknown covariance type {cov_type}")
    if kind == "univariate" and action_dim != 1:
        raise ValueError("a univariate policy needs one action dimension")
    match kind:
        case "univariate" | "diagonal":
            pass
        case _:
            raise NotImplementedError(f"unknown policy kind {kind}")

    dims = [obs_dim, *structure, out_dim]
    n_layers = len(dims) - 1
    act_names, softplus = _activation_names(list(act_fn), n_layers, action_dim, kind)

    layers: list[LayerSpec] = []
    named: list[NamedTensor] = []
    cursor = 0
    for i in range(n_layers):
        inn, out = int(dims[i]), int(dims[i + 1])
        weight_offset = cursor
        cursor += out * inn
        bias_offset = cursor
        cursor += out
        layer = LayerSpec(
            in_dim=inn,
            out_dim=out,
            weight_offset=weight_offset,
            bias_offset=bias_offset,
            activation=_as_names(act_names[i], out),
            layer_norm=bool(layer_norm) and i == 0,
        )
        layers.append(layer)
        named.append(NamedTensor(WEIGHT.format(i=i), weight_offset, (out, inn)))
        named.append(NamedTensor(BIAS.format(i=i), bias_offset, (out,)))

    log_sigma_offset = None
    match cov_type:
        case "single":
            log_sigma_offset = cursor
            cursor += action_dim
            named.append(NamedTensor("cov", log_sigma_offset, (1, action_dim)))
        case "fixed":
            named.append(NamedTensor("cov", None, (1, action_dim)))
        case "state":
            pass
        case _:
            raise NotImplementedError(cov_type)

    low = np.asarray(action_low, dtype=np.float64).reshape(-1)
    high = np.asarray(action_high, dtype=np.float64).reshape(-1)
    if kind == "univariate":
        center = np.zeros(1, dtype=np.float64)
        scale = np.array([float(np.max(np.abs(high)))], dtype=np.float64)
    elif squash:
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("squashing needs a bounded action space")
        center = (high + low) / 2
        scale = (high - low) / 2
    else:
        center = np.zeros(action_dim, dtype=np.float64)
        scale = np.ones(action_dim, dtype=np.float64)

    return GaussianSpec(
        obs_dim=obs_dim,
        action_dim=action_dim,
        kind=kind,
        cov_type=cov_type,
        squash=squash,
        layers=tuple(layers),
        n_params=cursor,
        log_sigma_offset=log_sigma_offset,
        sigma_is_softplus=softplus,
        action_center=tuple(float(v) for v in center),
        action_scale=tuple(float(v) for v in scale),
        named=tuple(named),
    )


def pack_module(module: Any, spec: GaussianSpec) -> tuple[np.ndarray, np.ndarray]:
    """Copy a ``select_policy`` module into the flat layout, by parameter name.

    Args:
        module: An initialized univariate or diagonal Gaussian policy.
        spec: The layout to pack into.

    Returns:
        The flat float32 vector and, for ``cov_type=fixed``, the frozen ``log_sigma`` of shape ``(A,)``.
        Otherwise the second array is empty.
    """
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(module.parameters()))
    flat = np.zeros(spec.n_params, dtype=np.float32)
    for tensor in spec.named:
        if tensor.offset is None:
            continue
        value = np.asarray(params[tensor.name], dtype=np.float32).reshape(tensor.shape)
        flat[tensor.offset : tensor.offset + value.size] = value.reshape(-1)
    if spec.cov_type == "fixed":
        frozen = np.asarray(params["cov"], dtype=np.float32).reshape(-1)
        return flat, frozen
    return flat, np.zeros((0,), dtype=np.float32)


def unpack_vector(
    flat: np.ndarray, spec: GaussianSpec, frozen_log_sigma: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Slice the flat vector into the checkpoint names of ``select_policy``.

    Args:
        flat: The trainable vector.
        spec: The layout.
        frozen_log_sigma: The frozen standard-deviation parameter when ``cov_type`` is ``fixed``.

    Returns:
        Arrays keyed by ``network.layers.i.weight``, ``bias`` and, when present, ``cov``.
    """
    out: dict[str, np.ndarray] = {}
    for tensor in spec.named:
        if tensor.offset is None:
            if frozen_log_sigma is None:
                raise ValueError("fixed covariance needs the frozen log_sigma")
            out[tensor.name] = np.asarray(frozen_log_sigma, dtype=np.float32).reshape(tensor.shape)
            continue
        out[tensor.name] = np.asarray(flat[tensor.offset : tensor.offset + int(np.prod(tensor.shape))]).reshape(
            tensor.shape
        )
    return out
