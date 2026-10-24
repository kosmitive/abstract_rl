"""Checks of the Metal (GPU) backend against the MLX CPU backend, skipped without Metal (e.g. on Linux)."""

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.mixed import set_device
from test_torch_parity import build_mpo, build_trpo, trpo_batch

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs MLX with Metal (Apple silicon)")


def _on(device: str, fn: Any) -> Any:
    set_device(device)
    try:
        out = fn()
        mx.eval(out)
        return out
    finally:
        set_device("cpu")


def test_scatter_add_gradient_with_duplicate_indices():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 8)).astype(np.float32)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    # duplicates within a row and scatter rows with a batch index > 0
    rows = np.array([1, 1, 3, 5, 5, 5])
    cols = np.array([0, 0, 2, 7, 7, 1])

    def run():
        def f(w):
            out = mx.zeros((6, 8)).at[mx.array(rows), mx.array(cols)].add((mx.array(x) * w)[:, 0])
            return (out**2).sum() + (out[mx.array(rows)] * w[:, :1]).sum()

        return mx.value_and_grad(f)(mx.array(w))

    cpu, gpu = _on("cpu", run), _on("gpu", run)
    np.testing.assert_allclose(np.array(gpu[0]), np.array(cpu[0]), rtol=1e-5)
    np.testing.assert_allclose(np.array(gpu[1]), np.array(cpu[1]), rtol=1e-5, atol=1e-6)


def test_compiled_trpo_step_matches_the_cpu(tmp_path):
    # the step on the inputs of the torch fixture, compiled (including the conjugate gradient and the line search).
    # Elementwise a few weights close to zero differ by more than 1e-4 relative (by < 1e-5 absolute), so the
    # parameters and the step are compared in norm.
    results = {}
    for device in ("cpu", "gpu"):

        def step(device: str = device) -> tuple[mx.array, mx.array]:
            mc = build_trpo(tmp_path / device)
            x0 = get_flat_params_from(mc.get("policy", False))
            with mc["conf"].ns("trpo"):
                alg = TRPOAlgorithm(mc)
            batch = trpo_batch()
            alg.perform_tr_sgd(0, batch, batch)
            return x0, get_flat_params_from(mc.get("policy", False))

        results[device] = [np.array(v) for v in _on(device, step)]
    (x0, x_cpu), (_, x_gpu) = results["cpu"], results["gpu"]
    assert np.linalg.norm(x_gpu - x_cpu) / np.linalg.norm(x_cpu) < 1e-4
    assert np.linalg.norm((x_gpu - x0) - (x_cpu - x0)) / np.linalg.norm(x_cpu - x0) < 1e-3
    np.testing.assert_allclose(x_gpu, x_cpu, rtol=1e-4, atol=1e-5)


def test_q_annotation_of_50k_states_with_20_actions(tmp_path):
    rng = np.random.default_rng(1)
    states = rng.standard_normal((50_000, 3)).astype(np.float32)
    actions = rng.standard_normal((50_000, 20)).astype(np.float32)

    def run(device):
        def q():
            mc = build_mpo(tmp_path / device)
            return mc.get("q_network", True).q_val_multi(mx.array(states), mx.array(actions))

        return np.array(_on(device, q))

    cpu, gpu = run("cpu"), run("gpu")
    assert gpu.shape == (50_000, 20)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


def test_rollout_on_the_cpu_stream_matches_the_gpu(tmp_path):
    # MPO with device=gpu evaluates the policy of the rollout on the CPU stream (env_rollout_on_cpu)
    def rollout(stream: Any, sub: str) -> Any:
        set_device("gpu")
        try:
            mc = build_mpo(tmp_path / sub)
            mx.random.seed(0)
            np.random.seed(0)  # noqa: NPY002
            with mx.stream(stream):
                tc = mc.env.execute_policy(mc.get("policy", target=False), 25, 75)
            mc.env.close()
            return tc.to_batch()
        finally:
            set_device("cpu")

    gpu, cpu = rollout(mx.gpu, "gpu"), rollout(mx.cpu, "cpu")
    assert len(gpu) == len(cpu) == 75
    # the same exploration noise, the trajectories differ by float noise of the policy evaluation only
    for field in ("states", "actions", "log_likelihoods", "rewards", "next_states"):
        assert np.allclose(getattr(gpu, field), getattr(cpu, field), rtol=1e-4, atol=1e-4), field
    assert np.array_equal(gpu.dones, cpu.dones)


def test_default_settings_cap_the_buffer_cache(monkeypatch):
    import abstract_rl.misc.mixed as mixed

    previous, device = mx.set_cache_limit(mixed.CACHE_LIMIT_BYTES), mx.default_device()
    try:
        # default_settings sets the limit
        mx.set_cache_limit(previous)
        mixed.default_settings("gpu")
        assert mx.set_cache_limit(previous) == mixed.CACHE_LIMIT_BYTES

        # a freed buffer larger than the limit leaves the cache with the next allocation
        monkeypatch.setattr(mixed, "CACHE_LIMIT_BYTES", 128 * 1024 * 1024)
        mx.clear_cache()
        big = mx.zeros((1024, 1024, 128), stream=mx.cpu)  # 512 MiB
        mx.eval(big)
        del big
        assert mx.get_cache_memory() >= 512 * 1024 * 1024
        mixed.default_settings("gpu")
        small = mx.ones((16,))
        mx.eval(small)
        assert mx.get_cache_memory() <= 128 * 1024 * 1024
    finally:
        mx.set_cache_limit(previous)
        mx.set_default_device(device)
        mx.clear_cache()


def test_fused_adam_matches_adam():
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from abstract_rl.misc.metal_kernels import FusedAdam
    from abstract_rl.misc.mixed import ELU

    def run(opt_cls: Any) -> list[np.ndarray]:
        mx.random.seed(0)
        # 6 parameters, two dispatches
        model = nn.Sequential(nn.Linear(7, 64), ELU(), nn.Linear(64, 32), ELU(), nn.Linear(32, 1))
        opt = opt_cls(learning_rate=1e-2, bias_correction=True)
        opt.init(model.trainable_parameters())
        x = mx.random.normal([30, 128, 7])
        y = mx.random.normal([30, 128])
        mx.eval(model.parameters(), x, y)
        loss_and_grad = nn.value_and_grad(model, lambda a, b: nn.losses.mse_loss(model(a).reshape(-1), b))
        state = [model.state, opt.state]

        def steps(xs: mx.array, ys: mx.array) -> mx.array:
            for j in range(xs.shape[0]):
                loss, grads = loss_and_grad(xs[j], ys[j])
                opt.update(model, grads)
            return loss

        fn = mx.compile(steps, inputs=state, outputs=state)
        for c in range(0, 30, 10):
            mx.eval(fn(x[c : c + 10], y[c : c + 10]), model.parameters(), opt.state)
        return [np.array(v) for _, v in tree_flatten(model.parameters()) + tree_flatten(opt.state)]

    set_device("gpu")
    try:
        expected, fused = run(optim.Adam), run(FusedAdam)
    finally:
        set_device("cpu")
    assert len(expected) == len(fused) == 20
    for a, b in zip(expected, fused, strict=True):
        np.testing.assert_array_equal(a, b)
