import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

_compile_warned = False


def default_settings(num_threads: int = 1) -> None:
    """
    Apply default settings to all.
    :param num_threads: Number of intra-op threads for torch. The networks are tiny, so a single thread is fastest.
    """
    torch.set_num_threads(num_threads)

    def float_formatter(x: Any) -> str:
        return f"{x: 10.4f}"

    np.set_printoptions(formatter={"float_kind": float_formatter})
    # torch.set_printoptions(profile='full')


def select_activation(name: str | None) -> nn.Module:
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: A object of the activation.
    """
    if name == "none" or name is None:
        return nn.Identity()

    elif name.lower() == "softplus":
        return nn.Softplus()

    elif name.lower() == "elu":
        return nn.ELU()

    elif name.lower() == "prelu":
        return nn.PReLU()

    elif name.lower() == "relu":
        return nn.ReLU()

    elif name.lower() == "lrelu":
        return nn.LeakyReLU()

    elif name.lower() == "tanh":
        return nn.Tanh()

    elif name.lower() == "sigmoid":
        return nn.Sigmoid()

    raise NotImplementedError(f"unknown activation {name}")


def torch_compile_enabled(conf: Any) -> bool:
    """Whether this run asked for ``torch.compile`` (missing key: off)."""
    return bool(conf.to_dict().get("torch_compile", False))


def _warn_compile(exc: BaseException) -> None:
    """Warn once per process that compilation failed and eager mode is in use."""
    global _compile_warned
    if _compile_warned:
        return
    _compile_warned = True
    detail = str(exc).splitlines()[0][:240]
    warnings.warn(
        f"torch.compile failed ({type(exc).__name__}: {detail}); falling back to eager",
        RuntimeWarning,
        stacklevel=3,
    )


def maybe_compile[F: Callable[..., Any]](fn: F, *, dynamic: bool) -> F:
    """Compile ``fn``. On failure, warn once and keep calling the eager function.

    Pass ``dynamic=False`` when every input shape is fixed, so the compiler specializes
    instead of retracing. A varying batch length needs ``dynamic=True``.
    """
    try:
        compiled = torch.compile(fn, dynamic=dynamic, fullgraph=True)
    except Exception as exc:
        _warn_compile(exc)
        return fn

    failed = False

    def run(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if failed:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            failed = True
            _warn_compile(exc)
            return fn(*args, **kwargs)

    return cast(F, run)


def compile_attr(obj: Any, name: str, *, dynamic: bool) -> None:
    """Replace an object's method with a compiled version of itself."""
    setattr(obj, name, maybe_compile(getattr(obj, name), dynamic=dynamic))
