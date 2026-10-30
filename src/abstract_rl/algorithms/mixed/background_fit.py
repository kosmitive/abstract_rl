"""
Fit a network in a second process while the main process goes on, e.g. with the next rollout.

Encoding the many small GPU kernels of a value fit keeps one CPU core busy, and so does the rollout. A thread does not
help, both need the GIL. The worker is a separate Python process that holds the network and its Adam moments. For
every fit the main process copies the replay and the drawn batch indices into shared memory and sends the current
parameters, the worker runs the same compiled Adam steps as the main process would and sends the parameters and the
log values back. The result is bit identical to fitting in the main process. optimizer_state and
set_optimizer_state read and write the moments of Adam in the worker, e.g. for a checkpoint.
"""

import importlib
import os
import subprocess
import sys
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# The worker runs one small graph per Adam step, so the encoding of command
# buffers dominates. Larger buffers cut the fit time by about 15 % on an M3 Max
# and do not change any result. A value set by the caller takes precedence.
WORKER_OPS_PER_BUFFER = "2000"

# (key, shape, dtype, offset in bytes) of every array in the shared memory
Layout = list[tuple[str, tuple[int, ...], str, int]]


def _layout(arrays: list[tuple[str, np.ndarray]]) -> tuple[Layout, int]:
    layout: Layout = []
    offset = 0
    for key, a in arrays:
        layout.append((key, tuple(a.shape), a.dtype.str, offset))
        # keep every array 64 byte aligned
        offset += -(-a.nbytes // 64) * 64
    return layout, offset


class BackgroundFit:
    """
    The main process side of a fit worker. submit starts a fit and returns, result waits for it.
    """

    def __init__(self, builder: str, spec: dict[str, Any]) -> None:
        """
        Start the worker process.
        :param builder: ``module:function``, called with spec in the worker. It returns the fit, a function of the
            parameters, the replay tree and the indices that returns the new parameters and the log values.
        :param spec: The (picklable) arguments of the builder.
        """
        r_main, w_worker = os.pipe()
        r_worker, w_main = os.pipe()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", __name__, str(r_worker), str(w_worker)],
            pass_fds=(r_worker, w_worker),
            stdout=subprocess.DEVNULL,
            env={"MLX_MAX_OPS_PER_BUFFER": WORKER_OPS_PER_BUFFER, **os.environ},
        )
        os.close(r_worker)
        os.close(w_worker)
        self._conn_in = Connection(r_main, writable=False)
        self._conn_out = Connection(w_main, readable=False)
        # the worker uses the device of the main process
        self._conn_out.send((builder, spec, "gpu" if mx.default_device() == mx.gpu else "cpu"))
        self._shm: SharedMemory | None = None
        self._pending = False

    def submit(self, params: dict[str, Any], tree: dict[str, Any], indices: mx.array) -> None:
        """
        Start a fit, the arrays are copied before this returns.
        :param params: The current parameters of the model.
        :param tree: The replay the rows of indices refer to.
        :param indices: One row of batch indices per Adam step.
        """
        assert not self._pending, "the last fit was not collected"
        arrays = [(k, np.asarray(v)) for k, v in cast(list[tuple[str, Any]], tree_flatten(tree))]
        arrays.append(("__indices__", np.asarray(indices)))
        layout, size = _layout(arrays)
        if self._shm is None or self._shm.size < size:
            self._close_shm()
            # some head room, the replay grows until it is full
            self._shm = SharedMemory(create=True, size=2 * size)
        for (_, a), (_, shape, dtype, offset) in zip(arrays, layout, strict=True):
            np.ndarray(shape, dtype, self._shm.buf, offset)[...] = a
        flat = [(k, np.array(v)) for k, v in cast(list[tuple[str, Any]], tree_flatten(params))]
        self._conn_out.send(("fit", self._shm.name, layout, flat))
        self._pending = True

    def result(self) -> tuple[dict[str, Any], list[tuple[int, float, float]]]:
        """
        Wait for the fit.
        :return: The new parameters (MLX arrays) and the log values (step, objective before, objective after).
        """
        assert self._pending, "no fit was submitted"
        try:
            msg = self._conn_in.recv()
        except EOFError:
            raise RuntimeError(f"the fit worker exited with {self._proc.wait()}") from None
        self._pending = False
        if isinstance(msg, str):
            raise RuntimeError(f"the fit worker failed:\n{msg}")
        flat, logs = msg
        return tree_unflatten([(k, mx.array(v)) for k, v in flat]), logs

    @property
    def pending(self) -> bool:
        return self._pending

    def _request(self, msg: tuple[Any, ...]) -> Any:
        assert not self._pending, "the last fit was not collected"
        self._conn_out.send(msg)
        try:
            reply = self._conn_in.recv()
        except EOFError:
            raise RuntimeError(f"the fit worker exited with {self._proc.wait()}") from None
        if isinstance(reply, str):
            raise RuntimeError(f"the fit worker failed:\n{reply}")
        return reply

    def optimizer_state(self) -> list[tuple[str, np.ndarray]]:
        """The flattened state of the optimizer in the worker (the moments of Adam), as NumPy arrays."""
        return cast(list[tuple[str, np.ndarray]], self._request(("get_state",)))

    def set_optimizer_state(self, flat: list[tuple[str, np.ndarray]]) -> None:
        """Set the state of the optimizer in the worker, flattened as returned by optimizer_state."""
        self._request(("set_state", flat))

    def _close_shm(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def close(self) -> None:
        """Stop the worker."""
        self._conn_out.close()
        self._conn_in.close()
        self._proc.wait()
        self._close_shm()


def worker_main(r_fd: int, w_fd: int) -> None:
    """The loop of the worker process, ends when the main process closes the pipe."""
    import traceback

    from abstract_rl.misc.mixed import default_settings

    conn_in, conn_out = Connection(r_fd, writable=False), Connection(w_fd, readable=False)
    builder, spec, device = conn_in.recv()
    default_settings(device)
    module, fn = builder.split(":")
    fit = getattr(importlib.import_module(module), fn)(spec)
    while True:
        try:
            msg = conn_in.recv()
        except EOFError:
            return
        try:
            if msg[0] == "get_state":
                conn_out.send(fit.optimizer_state())
                continue
            if msg[0] == "set_state":
                fit.set_optimizer_state(msg[1])
                conn_out.send(())
                continue
            _, name, layout, flat = msg
            shm = SharedMemory(name=name, track=False)
            arrays = [(k, mx.array(np.ndarray(shape, dtype, shm.buf, offset))) for k, shape, dtype, offset in layout]
            mx.eval([a for _, a in arrays])
            shm.close()
            indices = arrays.pop()[1]
            params = tree_unflatten([(k, mx.array(v)) for k, v in flat])
            new_params, logs = fit(params, tree_unflatten(arrays), indices)
            conn_out.send(([(k, np.array(v)) for k, v in cast(list[tuple[str, Any]], tree_flatten(new_params))], logs))
        except BrokenPipeError:
            # the main process is gone
            return
        except Exception:
            try:
                conn_out.send(traceback.format_exc())
            except BrokenPipeError:
                return


if __name__ == "__main__":
    worker_main(int(sys.argv[1]), int(sys.argv[2]))
