from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


class TrajectoryOperator:
    """
    A simple evaluation operator interface.
    """

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        raise NotImplementedError

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator. The annotations, which are
        computed lazily as MLX arrays, are evaluated at once and stored as numpy arrays.
        :param trajectories: trajectories to transform.
        """

        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()

        for trajectory in trajectories:
            self.transform(trajectory)
        materialize(trajectories)


def materialize(trajectories: list[Trajectory]) -> None:
    """Evaluate all MLX annotations of the trajectories in one go and replace them by numpy arrays."""
    lazy = [(t, k, v) for t in trajectories for k, v in t._annotations.items() if isinstance(v, mx.array)]
    mx.eval([v for _, _, v in lazy])
    for t, k, v in lazy:
        t[k] = np.array(v)


# states per call of the operators that run over the whole collection (annotation, Retrace), per device. On Metal
# 2048 is as fast as 4096 (616 vs 607 ms per full-memory epoch) at a 0.5 GB lower peak. On the CPU 512 is faster than
# 2048 (2.60 vs 2.85 s) and holds a quarter of the activations.
CHUNK_STATES = {"gpu": 2048, "cpu": 512}


def chunk_states() -> int:
    """The number of states per call on the default device."""
    return CHUNK_STATES["gpu" if mx.default_device() == mx.gpu else "cpu"]


def compile_for_device(fn: Callable[..., Any], inputs: list[Any]) -> Callable[..., Any]:
    """
    fn compiled with the given inputs on Metal. On the CPU fn stays uncompiled: the fused CPU kernels of mx.compile
    are slower than the separate (multithreaded) ops at these sizes, e.g. 1.55 s instead of 0.87 s for the annotation
    of a full memory.
    """
    if mx.default_device() == mx.gpu:
        return mx.compile(fn, inputs=inputs)
    return fn


def padded_size(n: int) -> int:
    """
    The number of rows n is padded to: a multiple of chunk_states(), or of 256 below one chunk, so the compiled calls
    see few distinct shapes while the memory fills.
    """
    chunk = chunk_states()
    step = chunk if n > chunk else 256
    return -(-n // step) * step


def pad_rows(x: mx.array, n: int) -> mx.array:
    """x with zero rows appended up to n rows."""
    if x.shape[0] == n:
        return x
    return mx.concatenate([x, mx.zeros((n - x.shape[0], *x.shape[1:]), dtype=x.dtype)])


def stack_field(trajectories: list[Trajectory], field: str, n: int) -> mx.array:
    """A field of all trajectories (e.g. states) concatenated, as float32 MLX array padded with zero rows to n rows."""
    x = np.concatenate([getattr(t, field) for t in trajectories])
    out = np.zeros((n, *x.shape[1:]), dtype=np.float32)
    out[: len(x)] = x
    return mx.array(out)


def run_in_chunks(fn: Callable[..., tuple[mx.array, ...]], *arrays: mx.array) -> list[mx.array]:
    """Run fn on consecutive row chunks of the (padded) arrays and concatenate each of its outputs."""
    n = arrays[0].shape[0]
    chunk = min(n, chunk_states())
    outs = [fn(*(a[i : i + chunk] for a in arrays)) for i in range(0, n, chunk)]
    return [mx.concatenate(list(o)) if len(outs) > 1 else o[0] for o in zip(*outs, strict=True)]


def store_rows(trajectories: list[Trajectory], annotations: dict[str, mx.array]) -> None:
    """Evaluate row-aligned annotations of the concatenated trajectories and store each trajectory's rows (numpy)."""
    mx.eval(list(annotations.values()))
    flat = {k: np.array(v) for k, v in annotations.items()}
    offset = 0
    for t in trajectories:
        n = len(t)
        for k, v in flat.items():
            t[k] = v[offset : offset + n]
        offset += n
