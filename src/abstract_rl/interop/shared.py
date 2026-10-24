"""One parameter buffer, viewed by MLX and by every JAX host device.

MLX allocates the vector. A slice whose data pointer is a multiple of 64 bytes
is the live parameter: JAX's CPU ``device_put`` of that memoryview is a view,
including a replicated sharding across host devices. MLX's functional updates
return a new array, so the new bytes are copied back into this buffer and the
module is rebound to the view. That copy is one parameter vector. It happens
on the eager update (TRPO writes the flat vector again after the compiled
step) and, if a compiled optimizer left a fresh array, once at the start of
the next rollout.

Do not close over the JAX array inside ``jit``. Pass it as an argument, and do
not donate it: donation did not alias this buffer on JAX 0.12 CPU, and a
donated buffer would dangle the MLX view.

Ordering: call ``mx.eval`` on a training result before JAX reads the buffer
(the copy does). Call ``block_until_ready`` on a JAX result before MLX reads
a buffer JAX just wrote. The parameter buffer is only written from the MLX
side. Keep this object alive for as long as either view is used.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

ALIGN = 64


def buffer_address(array: Any) -> int:
    """Data pointer of an MLX array, a NumPy view, or a JAX CPU array.

    The Linux CPU build of MLX has no ``unsafe_buffer_pointer``. The address
    then comes from the memoryview. JAX CPU arrays expose
    ``unsafe_buffer_pointer``.
    """
    unsafe = getattr(array, "unsafe_buffer_pointer", None)
    if callable(unsafe):
        try:
            return int(unsafe())
        except (TypeError, ValueError, RuntimeError):
            pass
    view = np.array(memoryview(array), copy=False)
    return int(view.__array_interface__["data"][0])


def _aligned_slice(n: int) -> tuple[mx.array, mx.array, np.ndarray]:
    """An MLX vector of ``n`` float32s whose data pointer is 64-byte aligned.

    Returns:
        The allocation that must stay alive, the aligned view, and a NumPy view of it.
    """
    item = np.dtype(np.float32).itemsize
    extra = ALIGN // item
    base = mx.zeros((n + extra,), dtype=mx.float32)
    base_ptr = buffer_address(base)
    misaligned = base_ptr % ALIGN
    offset = 0 if misaligned == 0 else (ALIGN - misaligned) // item
    view = base[offset : offset + n]
    numpy_view = np.array(memoryview(view), copy=False).reshape(n)
    if buffer_address(view) % ALIGN != 0:
        raise RuntimeError("the shared parameter buffer is not 64-byte aligned")
    if buffer_address(view) != int(numpy_view.__array_interface__["data"][0]):
        raise RuntimeError("the NumPy view of the parameter buffer is a copy")
    return base, view, numpy_view


class SharedParams:
    """Flat float32 parameters owned by one MLX allocation."""

    def __init__(self, n: int) -> None:
        """Allocate an aligned buffer of ``n`` float32s, initially zero.

        Args:
            n: Length of the vector.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = int(n)
        self._base, self._view, self._np = _aligned_slice(self.n)

    def as_mlx(self) -> mx.array:
        """The MLX view. Writes through this view are the buffer."""
        return self._view

    def numpy_view(self) -> np.ndarray:
        """A writable NumPy view of the same bytes."""
        return self._np

    def write_numpy(self, values: np.ndarray) -> None:
        """Copy ``values`` into the buffer. The destination address does not change."""
        self._np[:] = np.asarray(values, dtype=np.float32).reshape(self.n)

    def write_from_mlx(self, new: mx.array) -> bool:
        """Copy ``new`` into the buffer when it is a different allocation.

        Args:
            new: The array a functional update just installed.

        Returns:
            Whether a copy was made. ``False`` when ``new`` already is this buffer.

        Raises:
            RuntimeError: When called while MLX is tracing. Callers skip the
                write-back during ``mx.compile`` and run it on the eager update.
        """
        # Evaluating a tracer is the signal that this update is inside a compiled
        # graph. MLX raises ValueError (the message says the eval is not allowed
        # during a transformation). The compiled result is installed afterwards.
        try:
            mx.eval(new)
        except (RuntimeError, ValueError) as exc:
            if "transform" not in str(exc):
                raise
            raise RuntimeError("write_from_mlx was called while MLX was tracing") from exc
        if buffer_address(new) == buffer_address(self._view):
            return False
        # ``np.array`` copies first, so assigning a view of this buffer onto itself is safe.
        self._np[:] = np.array(new, copy=True).reshape(self.n)
        return True

    def as_jax(self, device: Any = None) -> Any:
        """A JAX CPU array viewing this buffer.

        Args:
            device: A JAX device. The first CPU device when omitted.

        Returns:
            A JAX array with the same data pointer. Pass it into ``jit`` as an
            argument; do not close over it and do not donate it.
        """
        import jax

        target = jax.devices("cpu")[0] if device is None else device
        placed = jax.device_put(self._np, target)
        placed.block_until_ready()
        return placed

    def replicated(self, devices: list[Any]) -> Any:
        """One replicated JAX array, a zero-copy view on each host device.

        Args:
            devices: The host devices that will read the vector together.

        Returns:
            A globally replicated array. On JAX 0.12 CPU this kept the MLX
            data pointer on every device. Callers that observe a different
            pointer are holding a copy and should say so in the benchmark.
        """
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        mesh = Mesh(np.array(devices), ("i",))
        placed = jax.device_put(self._np, NamedSharding(mesh, PartitionSpec()))
        placed.block_until_ready()
        return placed
