"""The chunked compiled temperature dual (CPU) and the Metal kernels (GPU) against the step-by-step version."""

import math

import mlx.core as mx
import numpy as np
import pytest

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import (
    find_optimal_temperature,
    temperature_dual,
)
from abstract_rl.misc.mixed import set_device


def _step(q, eps, lr, log_eta, m, v, step):
    obj, g = mx.value_and_grad(lambda le: temperature_dual(q, mx.exp(le), eps).sum())(log_eta)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    denom = mx.sqrt(v) / mx.sqrt(1 - 0.999**step) + 1e-8
    return log_eta - (lr / (1 - 0.9**step)) * m / denom, m, v, obj


_compiled_step = mx.compile(_step)


def _step_by_step(q: mx.array, eps: float, eta0: float, lr: float, max_steps: int, dtype: mx.Dtype) -> float:
    """One compiled call per Adam step, as before the chunked version."""
    q = q.astype(dtype)
    log_eta = mx.array([math.log(eta0)], dtype=dtype)
    m, v = mx.zeros_like(log_eta), mx.zeros_like(log_eta)
    eps_a, lr_a = mx.array(eps, dtype=dtype), mx.array(lr, dtype=dtype)
    for step in range(1, max_steps + 1):
        log_eta, m, v, _ = _compiled_step(q, eps_a, lr_a, log_eta, m, v, mx.array(step, dtype=dtype))
    return math.exp(log_eta.item())


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("max_steps", [200, 45])
def test_dual_matches_step_by_step(device: str, max_steps: int):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("needs MLX with Metal (Apple silicon)")
    rng = np.random.default_rng(3)
    set_device(device)
    try:
        dtype = mx.float32 if device == "gpu" else mx.float64
        for n, eta0 in [(300, 1.0), (4000, 3.7)]:
            q = mx.array((rng.standard_normal((n, 20)) * 10 - 30).astype(np.float32))
            expected = np.float32(_step_by_step(q, 0.1, eta0, 0.05, max_steps, dtype))
            eta = find_optimal_temperature(q, 0.1, eta0, 0.05, max_steps)
            assert eta.dtype == mx.float32 and eta.shape == (1,)
            if device == "cpu":
                assert np.array(eta)[0] == expected
            else:
                # the kernels sum in another order than the compiled graph
                np.testing.assert_allclose(np.array(eta)[0], expected, rtol=1e-5)
    finally:
        set_device("cpu")
