"""The array operations of the shared Gaussian forward.

A JAX rollout and an MLX training step call the same formulas. Only this module
knows which framework evaluates them.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class ArrayNS(Protocol):
    """Matmul, activations and reductions used by the policy forward."""

    def tanh(self, x: Any) -> Any:
        """Hyperbolic tangent."""

    def exp(self, x: Any) -> Any:
        """Exponential."""

    def log(self, x: Any) -> Any:
        """Natural logarithm."""

    def arctanh(self, x: Any) -> Any:
        """Inverse hyperbolic tangent."""

    def elu(self, x: Any, alpha: float = 1.0) -> Any:
        """ELU with a finite gradient for large negative inputs."""

    def relu(self, x: Any) -> Any:
        """Rectified linear unit."""

    def sigmoid(self, x: Any) -> Any:
        """Logistic sigmoid."""

    def softplus(self, x: Any) -> Any:
        """``log(1 + exp(x))`` as ``logaddexp(x, 0)``."""

    def leaky_relu(self, x: Any, negative_slope: float = 1e-2) -> Any:
        """Leaky ReLU, default slope matching ``mlx.nn.LeakyReLU``."""

    def layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """Layer norm over the last axis, no affine parameters."""

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        """Select from ``a`` where ``cond`` else ``b``."""

    def maximum(self, a: Any, b: Any) -> Any:
        """Elementwise maximum."""

    def minimum(self, a: Any, b: Any) -> Any:
        """Elementwise minimum."""

    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        """Clamp ``x`` to ``[lo, hi]``."""

    def concatenate(self, parts: list[Any], axis: int) -> Any:
        """Concatenate along ``axis``."""

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Reshape, keeping the same buffer when the framework allows it."""

    def broadcast_to(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Broadcast ``x`` to ``shape``."""

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum, optionally along one axis."""

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Mean, optionally along one axis."""

    def const(self, values: tuple[float, ...] | list[float]) -> Any:
        """A 1-d float32 vector of Python floats."""

    def transpose(self, x: Any) -> Any:
        """Transpose a matrix."""


def _layer_norm(ns: ArrayNS, x: Any, eps: float) -> Any:
    """Layer norm matching ``mx.fast.layer_norm`` without an affine transform."""
    mean = ns.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = ns.mean(centered * centered, axis=-1, keepdims=True)
    return centered / (var + eps) ** 0.5


class MlxNS:
    """Array namespace backed by MLX."""

    def tanh(self, x: Any) -> Any:
        """Hyperbolic tangent."""
        import mlx.core as mx

        return mx.tanh(x)

    def exp(self, x: Any) -> Any:
        """Exponential."""
        import mlx.core as mx

        return mx.exp(x)

    def log(self, x: Any) -> Any:
        """Natural logarithm."""
        import mlx.core as mx

        return mx.log(x)

    def arctanh(self, x: Any) -> Any:
        """Inverse hyperbolic tangent."""
        import mlx.core as mx

        return mx.arctanh(x)

    def elu(self, x: Any, alpha: float = 1.0) -> Any:
        """ELU. ``mlx.nn.elu`` NaNs its gradient above about 88, so the negative branch clamps the exp."""
        import mlx.core as mx

        return mx.where(x > 0, x, alpha * (mx.exp(mx.minimum(x, 0)) - 1))

    def relu(self, x: Any) -> Any:
        """Rectified linear unit."""
        import mlx.core as mx

        return mx.maximum(x, 0)

    def sigmoid(self, x: Any) -> Any:
        """Logistic sigmoid."""
        import mlx.core as mx

        return mx.sigmoid(x)

    def softplus(self, x: Any) -> Any:
        """Softplus, the same ``logaddexp`` as ``mlx.nn.softplus``."""
        import mlx.core as mx

        return mx.logaddexp(x, 0)

    def leaky_relu(self, x: Any, negative_slope: float = 1e-2) -> Any:
        """Leaky ReLU."""
        import mlx.core as mx

        return mx.maximum(negative_slope * x, x)

    def layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """Layer norm, the MLX primitive the existing MLP uses."""
        import mlx.core as mx

        return mx.fast.layer_norm(x, None, None, eps)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        """Select from ``a`` where ``cond`` else ``b``."""
        import mlx.core as mx

        return mx.where(cond, a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        """Elementwise maximum."""
        import mlx.core as mx

        return mx.maximum(a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        """Elementwise minimum."""
        import mlx.core as mx

        return mx.minimum(a, b)

    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        """Clamp ``x`` to ``[lo, hi]``."""
        import mlx.core as mx

        return mx.clip(x, lo, hi)

    def concatenate(self, parts: list[Any], axis: int) -> Any:
        """Concatenate along ``axis``."""
        import mlx.core as mx

        return mx.concatenate(parts, axis=axis)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Reshape."""
        import mlx.core as mx

        return mx.reshape(x, shape)

    def broadcast_to(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Broadcast ``x`` to ``shape``."""
        import mlx.core as mx

        return mx.broadcast_to(x, shape)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum."""
        import mlx.core as mx

        return mx.sum(x) if axis is None else mx.sum(x, axis=axis)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Mean."""
        import mlx.core as mx

        if axis is None:
            return mx.mean(x)
        return mx.mean(x, axis=axis, keepdims=keepdims)

    def const(self, values: tuple[float, ...] | list[float]) -> Any:
        """A 1-d float32 vector."""
        import mlx.core as mx

        return mx.array([float(v) for v in values], dtype=mx.float32)

    def transpose(self, x: Any) -> Any:
        """Transpose a matrix."""
        return x.T


class JaxNS:
    """Array namespace backed by JAX NumPy."""

    def tanh(self, x: Any) -> Any:
        """Hyperbolic tangent."""
        import jax.numpy as jnp

        return jnp.tanh(x)

    def exp(self, x: Any) -> Any:
        """Exponential."""
        import jax.numpy as jnp

        return jnp.exp(x)

    def log(self, x: Any) -> Any:
        """Natural logarithm."""
        import jax.numpy as jnp

        return jnp.log(x)

    def arctanh(self, x: Any) -> Any:
        """Inverse hyperbolic tangent."""
        import jax.numpy as jnp

        return jnp.arctanh(x)

    def elu(self, x: Any, alpha: float = 1.0) -> Any:
        """ELU with the same clamped negative branch as the MLX policy."""
        import jax.numpy as jnp

        return jnp.where(x > 0, x, alpha * (jnp.exp(jnp.minimum(x, 0)) - 1))

    def relu(self, x: Any) -> Any:
        """Rectified linear unit."""
        import jax.numpy as jnp

        return jnp.maximum(x, 0)

    def sigmoid(self, x: Any) -> Any:
        """Logistic sigmoid."""
        return jax_sigmoid(x)

    def softplus(self, x: Any) -> Any:
        """Softplus via ``logaddexp``, matching MLX."""
        import jax.numpy as jnp

        return jnp.logaddexp(x, jnp.zeros((), dtype=x.dtype))

    def leaky_relu(self, x: Any, negative_slope: float = 1e-2) -> Any:
        """Leaky ReLU."""
        import jax.numpy as jnp

        return jnp.maximum(negative_slope * x, x)

    def layer_norm(self, x: Any, eps: float = 1e-5) -> Any:
        """Layer norm over the last axis, no affine parameters."""
        return _layer_norm(self, x, eps)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        """Select from ``a`` where ``cond`` else ``b``."""
        import jax.numpy as jnp

        return jnp.where(cond, a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        """Elementwise maximum."""
        import jax.numpy as jnp

        return jnp.maximum(a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        """Elementwise minimum."""
        import jax.numpy as jnp

        return jnp.minimum(a, b)

    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        """Clamp ``x`` to ``[lo, hi]``."""
        import jax.numpy as jnp

        return jnp.clip(x, lo, hi)

    def concatenate(self, parts: list[Any], axis: int) -> Any:
        """Concatenate along ``axis``."""
        import jax.numpy as jnp

        return jnp.concatenate(parts, axis=axis)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Reshape."""
        import jax.numpy as jnp

        return jnp.reshape(x, shape)

    def broadcast_to(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Broadcast ``x`` to ``shape``."""
        import jax.numpy as jnp

        return jnp.broadcast_to(x, shape)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum."""
        import jax.numpy as jnp

        return jnp.sum(x) if axis is None else jnp.sum(x, axis=axis)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Mean."""
        import jax.numpy as jnp

        if axis is None:
            return jnp.mean(x)
        return jnp.mean(x, axis=axis, keepdims=keepdims)

    def const(self, values: tuple[float, ...] | list[float]) -> Any:
        """A 1-d float32 vector."""
        import jax.numpy as jnp

        return jnp.asarray(tuple(float(v) for v in values), dtype=jnp.float32)

    def transpose(self, x: Any) -> Any:
        """Transpose a matrix."""
        import jax.numpy as jnp

        return jnp.transpose(x)


def jax_sigmoid(x: Any) -> Any:
    """Logistic sigmoid. ``jnp.sigmoid`` is filled in lazily, so it is looked up by name."""
    import jax.numpy as jnp

    return jnp.sigmoid(x)  # type: ignore[attr-defined]


MLX = MlxNS()
JAX = JaxNS()
