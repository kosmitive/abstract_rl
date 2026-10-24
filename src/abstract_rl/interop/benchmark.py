"""Micro-benchmark of the sharded JAX rollout and the MLX handoff.

Run from the repository root:

    PYTHONPATH=src python -m abstract_rl.interop.benchmark

The process re-execs itself once so ``XLA_FLAGS=--xla_force_host_platform_device_count``
is set before JAX is imported, unless that flag is already set. Numbers printed
here are this machine's. They are not the M3 Max figures.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np


def _reexec_with_devices() -> None:
    """Import JAX only after the host device count is set."""
    if os.environ.get("ABSTRACT_RL_SHARD_READY") == "1":
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        count = os.environ.get("ABSTRACT_RL_BENCH_DEVICES", "8")
        extra = f"--xla_force_host_platform_device_count={count}"
        os.environ["XLA_FLAGS"] = f"{flags} {extra}".strip()
    os.environ["ABSTRACT_RL_SHARD_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _policy_spec() -> Any:
    from abstract_rl.interop.spec import make_gaussian_spec

    return make_gaussian_spec(
        obs_dim=3,
        action_dim=1,
        kind="univariate",
        cov_type="single",
        structure=[64, 64],
        act_fn=["tanh", "tanh", "none"],
        layer_norm=False,
        squash=True,
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
    )


def _time_rollout(
    env: Any, spec: Any, shared: Any, keys: Any, horizon: int, n_devices: int, copy_params: bool
) -> float:
    """Seconds for one rollout. ``copy_params`` is the duplicated-parameter baseline."""
    from abstract_rl.interop.scan import sharded_rollout

    start = time.perf_counter()
    result = sharded_rollout(
        env,
        spec,
        shared,
        np.zeros(0, np.float32),
        keys,
        horizon,
        n_devices,
        deterministic=False,
        normalize=True,
        obs_scale=np.array([1.0, 1.0, 8.0], np.float32),
        reward_scale=1.0,
        reward_offset=0.0,
        n_keep=int(keys.shape[0]),
        copy_params=copy_params,
    )
    # touch the arena so the copy back to MLX is included
    _ = np.array(result.rewards).sum()
    return time.perf_counter() - start


def main() -> None:
    """Print handoff times and Pendulum steps/s at a few batch sizes."""
    _reexec_with_devices()
    import jax
    import mlx.core as mx

    from abstract_rl.interop.envs import make_jax_env
    from abstract_rl.interop.shared import SharedParams, buffer_address

    spec = _policy_spec()
    shared = SharedParams(spec.n_params)
    rng = np.random.default_rng(0)
    shared.write_numpy(rng.normal(size=spec.n_params).astype(np.float32) * 0.05)
    env = make_jax_env("Pendulum-v1")
    n_devices = len(jax.devices("cpu"))
    jax_version = jax.__version__
    mlx_version = mx.__version__  # type: ignore[attr-defined]
    print(f"python {sys.version.split()[0]}  jax {jax_version}  mlx {mlx_version}")
    print(f"cpu devices {n_devices}  XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    placed = shared.as_jax()
    same = buffer_address(placed) == buffer_address(shared.as_mlx())
    print(f"single-device param view shares the MLX pointer: {same}")

    # handoff of the parameter vector, and of one trajectory chunk into the arena
    handoff_n = 1024
    master = jax.random.key(0) if hasattr(jax.random, "key") else jax.random.PRNGKey(0)
    keys = jax.random.split(master, handoff_n)
    # warmup
    _time_rollout(env, spec, shared, keys, 20, 1, False)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        view = shared.as_jax()
        view.block_until_ready()
        times.append(time.perf_counter() - t0)
    print(f"params MLX to JAX, N={handoff_n} theta={spec.n_params}: {np.median(times) * 1e6:.1f} us median (5)")

    from abstract_rl.interop.scan import sharded_rollout

    chunk = sharded_rollout(
        env,
        spec,
        shared,
        np.zeros(0, np.float32),
        keys,
        20,
        min(n_devices, 4) if n_devices >= 4 else 1,
        deterministic=True,
        normalize=True,
        obs_scale=np.array([1.0, 1.0, 8.0], np.float32),
        reward_scale=1.0,
        reward_offset=0.0,
        n_keep=handoff_n,
    )
    t0 = time.perf_counter()
    total = float(np.array(mx.sum(chunk.rewards)))
    arena_s = time.perf_counter() - t0
    print(f"MLX read of the reward arena ({chunk.rewards.size} floats): {arena_s * 1e6:.1f} us  sum={total:.4f}")
    print(f"params were views on that chunk: {chunk.params_are_views}")

    print("env        N     devices  steps/s     ms/chunk  duplicated_steps/s")
    horizon = 50
    for n_envs in (32, 160, 1024):
        usable = [d for d in (1, min(4, n_devices), n_devices) if n_envs % d == 0 and d <= n_devices]
        # unique, keep order
        devices_list = list(dict.fromkeys(usable))
        master = jax.random.key(1) if hasattr(jax.random, "key") else jax.random.PRNGKey(1)
        keys_n = jax.random.split(master, n_envs)
        for devices in devices_list:
            _time_rollout(env, spec, shared, keys_n, horizon, devices, False)
            view_s = _time_rollout(env, spec, shared, keys_n, horizon, devices, False)
            copy_s = _time_rollout(env, spec, shared, keys_n, horizon, devices, True)
            steps = n_envs * horizon
            rate = steps / view_s
            copied = steps / copy_s
            print(f"Pendulum {n_envs:5d}  {devices:7d}  {rate:10.0f}  {view_s * 1e3:8.2f}  {copied:10.0f}")


if __name__ == "__main__":
    main()
