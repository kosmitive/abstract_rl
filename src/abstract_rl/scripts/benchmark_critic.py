"""Micro-benchmark of the MPO critic's Adam scan.

The fit is 1000 Adam steps on minibatches of 256, the Pendulum MPO critic (input 4, hidden
256, 256, ELU). x64 is on, as in training. Each lever is a separate compile. The library
keeps scan unroll 8; the other levers were slower or flat on the machine that produced the
pull request and are here so they can be timed again.

Usage::

    uv run python -m abstract_rl.scripts.benchmark_critic
    uv run python -m abstract_rl.scripts.benchmark_critic --steps 1000 --batch 256

The index stream of the shipped scan is the previous one (one ``split`` and one ``randint``
per step). Levers that draw indices differently say so in the table. A final check feeds one
shared index array to the tree Adam step and to the flat Adam step.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.value_functions.fc_q_networks import _critic_scan, q_forward

jax.config.update("jax_enable_x64", True)


def _wait(tree: object) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _flat_params(tree: object) -> np.ndarray:
    pieces = [np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]
    if not pieces:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(pieces)


def _median_ms(fn: Any, repeats: int) -> tuple[float, Any]:
    """Time ``repeats`` calls after one warmup. Returns the median milliseconds and the last output."""
    out = fn()
    _wait(out)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = fn()
        _wait(out)
        samples.append((time.perf_counter() - started) * 1e3)
    return float(np.median(samples)), out


def _f64_count(fn: Any, args: tuple[Any, ...]) -> tuple[int, int]:
    """Scalar and array ``f64`` mentions in the optimized HLO."""
    text = fn.lower(*args).compile().compiled.as_text()
    scalar = text.count("f64[]")
    array = text.count("f64[") - scalar
    return scalar, array


def _problem(capacity: int) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)
    key, k_net, k_s, k_a, k_t = jax.random.split(key, 5)
    net = MultilayerPerceptron(k_net, [4, 256, 256, 1], ["elu", "elu", "none"], False, "xavier_gaussian")
    diff, static = eqx.partition(net, eqx.is_inexact_array)
    tx = optax.adam(1e-3)
    opt_state = tx.init(diff)
    states = jax.random.normal(k_s, (capacity, 3), dtype=jnp.float32)
    actions = jax.random.normal(k_a, (capacity, 1), dtype=jnp.float32)
    targets = jax.random.normal(k_t, (capacity,), dtype=jnp.float32)
    data = (states, actions, targets, jax.random.PRNGKey(1), jnp.int32(capacity))
    flat, unravel = jax.flatten_util.ravel_pytree(diff)
    return {
        "diff": diff,
        "static": static,
        "opt_state": opt_state,
        "tx": tx,
        "data": data,
        "flat": flat,
        "unravel": unravel,
        "tx_flat": optax.adam(1e-3),
        "opt_flat": optax.adam(1e-3).init(flat),
    }


def _tree_scan(unroll: int, steps: int, batch: int, index_mode: str, pregather: bool) -> Any:
    """Tree-Adam critic. ``index_mode`` is ``inside``, ``upfront`` (same splits) or ``oneshot``."""

    @eqx.filter_jit
    def run(
        data: tuple[Any, ...], diff: Any, opt_state: Any, static: Any, tx: Any
    ) -> tuple[Any, Any, jax.Array, jax.Array]:
        states, actions, targets, key, size = data

        def draw(key_: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
            key_, sample_key = jax.random.split(key_)
            return key_, jax.random.randint(sample_key, (batch,), 0, size)

        if index_mode == "upfront":
            _key, indices = jax.lax.scan(draw, key, None, length=steps)
        elif index_mode == "oneshot":
            indices = jax.random.randint(key, (steps, batch), 0, size)
        else:
            indices = None

        if pregather:
            assert indices is not None
            taken_states = states[indices]
            taken_actions = actions[indices]
            taken_targets = targets[indices]
        else:
            taken_states = taken_actions = taken_targets = None

        def loss_on(weights: Any, step: jax.Array, idx: jax.Array) -> jax.Array:
            network = eqx.combine(weights, static)
            if pregather:
                prediction = q_forward(network, taken_states[step], taken_actions[step]).reshape(-1)
                target = taken_targets[step]
            else:
                prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
                target = targets[idx]
            return jnp.mean(jnp.square(prediction - target))

        def body(
            carry: tuple[Any, Any, jax.Array], step: jax.Array
        ) -> tuple[tuple[Any, Any, jax.Array], tuple[jax.Array, jax.Array]]:
            weights, state, key_ = carry
            if index_mode == "inside":
                key_, sample_key = jax.random.split(key_)
                idx = jax.random.randint(sample_key, (batch,), 0, size)
            else:
                assert indices is not None
                idx = indices[step]
            pre, grads = eqx.filter_value_and_grad(lambda w: loss_on(w, step, idx))(weights)
            updates, state = tx.update(grads, state, weights)
            weights = eqx.apply_updates(weights, updates)
            post = jax.lax.cond(step % 10 == 0, lambda w: loss_on(w, step, idx), lambda _w: pre, weights)
            return (weights, state, key_), (pre, post)

        (diff, opt_state, _), (pre, post) = jax.lax.scan(body, (diff, opt_state, key), jnp.arange(steps), unroll=unroll)
        return diff, opt_state, pre, post

    return run


def _flat_scan(steps: int, batch: int, unroll: int, float32_adam: bool, unravel: Any) -> Any:
    """Flat-parameter Adam. ``float32_adam`` uses an explicit float32 bias correction."""

    @eqx.filter_jit
    def run_optax(
        data: tuple[Any, ...], flat: jax.Array, opt_state: Any, static: Any, tx: Any
    ) -> tuple[jax.Array, jax.Array]:
        states, actions, targets, key, size = data

        def loss_on(vector: jax.Array, idx: jax.Array) -> jax.Array:
            network = eqx.combine(unravel(vector), static)
            prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
            return jnp.mean(jnp.square(prediction - targets[idx]))

        def body(
            carry: tuple[jax.Array, Any, jax.Array], step: jax.Array
        ) -> tuple[tuple[jax.Array, Any, jax.Array], jax.Array]:
            vector, state, key_ = carry
            key_, sample_key = jax.random.split(key_)
            idx = jax.random.randint(sample_key, (batch,), 0, size)
            pre, grads = jax.value_and_grad(lambda v: loss_on(v, idx))(vector)
            updates, state = tx.update(grads, state, vector)
            vector = optax.apply_updates(vector, updates)
            return (vector, state, key_), pre

        (flat, _state, _), pre = jax.lax.scan(body, (flat, opt_state, key), jnp.arange(steps), unroll=unroll)
        return flat, pre

    @eqx.filter_jit
    def run_f32(
        data: tuple[Any, ...], flat: jax.Array, mu: jax.Array, nu: jax.Array, static: Any
    ) -> tuple[jax.Array, jax.Array]:
        states, actions, targets, key, size = data
        b1 = jnp.float32(0.9)
        b2 = jnp.float32(0.999)
        eps = jnp.float32(1e-8)
        lr = jnp.float32(1e-3)

        def loss_on(vector: jax.Array, idx: jax.Array) -> jax.Array:
            network = eqx.combine(unravel(vector), static)
            prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
            return jnp.mean(jnp.square(prediction - targets[idx]))

        def body(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], step: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
            vector, mu_, nu_, count, key_ = carry
            key_, sample_key = jax.random.split(key_)
            idx = jax.random.randint(sample_key, (batch,), 0, size)
            pre, grad = jax.value_and_grad(lambda v: loss_on(v, idx))(vector)
            count = count + jnp.int32(1)
            mu_ = (jnp.float32(1) - b1) * grad + b1 * mu_
            nu_ = (jnp.float32(1) - b2) * (grad * grad) + b2 * nu_
            correction = count.astype(jnp.float32)
            mu_hat = mu_ / (jnp.float32(1) - b1**correction)
            nu_hat = nu_ / (jnp.float32(1) - b2**correction)
            vector = vector - lr * mu_hat / (jnp.sqrt(nu_hat) + eps)
            return (vector, mu_, nu_, count, key_), pre

        (flat, _mu, _nu, _count, _key), pre = jax.lax.scan(
            body,
            (flat, mu, nu, jnp.int32(0), key),
            jnp.arange(steps),
            unroll=unroll,
        )
        return flat, pre

    return run_f32 if float32_adam else run_optax


def _shared_index_check(problem: dict[str, Any], steps: int, batch: int) -> str:
    """Tree Adam and flat Optax Adam on one index array. This is the check for a changed RNG stream."""
    states, actions, targets, key, size = problem["data"]
    indices = jax.random.randint(key, (steps, batch), 0, size)
    static = problem["static"]
    unravel = problem["unravel"]
    tx = problem["tx"]
    tx_flat = problem["tx_flat"]

    @eqx.filter_jit
    def tree_on(diff: Any, opt_state: Any, indices: jax.Array) -> tuple[Any, jax.Array]:
        def loss_on(weights: Any, idx: jax.Array) -> jax.Array:
            network = eqx.combine(weights, static)
            prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
            return jnp.mean(jnp.square(prediction - targets[idx]))

        def body(carry: tuple[Any, Any], idx: jax.Array) -> tuple[tuple[Any, Any], jax.Array]:
            weights, state = carry
            pre, grads = eqx.filter_value_and_grad(lambda w: loss_on(w, idx))(weights)
            updates, state = tx.update(grads, state, weights)
            return (eqx.apply_updates(weights, updates), state), pre

        (diff, _state), pre = jax.lax.scan(body, (diff, opt_state), indices)
        return diff, pre

    @eqx.filter_jit
    def flat_on(flat: jax.Array, opt_state: Any, indices: jax.Array) -> tuple[jax.Array, jax.Array]:
        def loss_on(vector: jax.Array, idx: jax.Array) -> jax.Array:
            network = eqx.combine(unravel(vector), static)
            prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
            return jnp.mean(jnp.square(prediction - targets[idx]))

        def body(carry: tuple[jax.Array, Any], idx: jax.Array) -> tuple[tuple[jax.Array, Any], jax.Array]:
            vector, state = carry
            pre, grads = jax.value_and_grad(lambda v: loss_on(v, idx))(vector)
            updates, state = tx_flat.update(grads, state, vector)
            return (optax.apply_updates(vector, updates), state), pre

        (flat, _state), pre = jax.lax.scan(body, (flat, opt_state), indices)
        return flat, pre

    del size
    copied_diff = jax.tree.map(jnp.array, problem["diff"])
    copied_state = jax.tree.map(jnp.array, problem["opt_state"])
    tree_diff, tree_pre = tree_on(copied_diff, copied_state, indices)
    flat, flat_pre = flat_on(jnp.array(problem["flat"]), jax.tree.map(jnp.array, problem["opt_flat"]), indices)
    params = float(np.max(np.abs(np.asarray(flat) - _flat_params(tree_diff))))
    pre = float(np.max(np.abs(np.asarray(flat_pre) - np.asarray(tree_pre))))
    return f"same indices, tree vs flat optax: params {params:.3e}, per-step loss {pre:.3e}"


def _profile(problem: dict[str, Any], steps: int, batch: int, repeats: int) -> dict[str, float]:
    """Separate the network step from Adam and from index draws.

    The gradient is applied to the weights, so the compiler cannot hoist one matmul out of the
    scan. A scan that returns an unused gradient gets deleted.
    """
    states, actions, targets, key, size = problem["data"]
    static = problem["static"]
    tx = problem["tx"]

    @eqx.filter_jit
    def gemm_sgd(diff: Any) -> Any:
        def loss_on(weights: Any, idx: jax.Array) -> jax.Array:
            network = eqx.combine(weights, static)
            prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
            return jnp.mean(jnp.square(prediction - targets[idx]))

        def body(carry: tuple[Any, jax.Array], step: jax.Array) -> tuple[tuple[Any, jax.Array], jax.Array]:
            weights, key_ = carry
            key_, sample_key = jax.random.split(key_)
            idx = jax.random.randint(sample_key, (batch,), 0, size)
            _pre, grads = eqx.filter_value_and_grad(lambda w: loss_on(w, idx))(weights)
            weights = jax.tree.map(lambda w, g: w - jnp.float32(1e-3) * g, weights, grads)
            return (weights, key_), step

        return jax.lax.scan(body, (diff, key), jnp.arange(steps))

    grad0 = jax.tree.map(lambda x: jnp.zeros_like(x), problem["diff"])

    @eqx.filter_jit
    def adam_only(diff: Any, opt_state: Any, grad: Any) -> Any:
        def body(carry: tuple[Any, Any], _: jax.Array) -> tuple[tuple[Any, Any], None]:
            weights, state = carry
            updates, state = tx.update(grad, state, weights)
            return (eqx.apply_updates(weights, updates), state), None

        return jax.lax.scan(body, (diff, opt_state), jnp.arange(steps))

    @eqx.filter_jit
    def rng_only(key_: jax.Array, size_: jax.Array) -> jax.Array:
        def body(key_i: jax.Array, _: jax.Array) -> tuple[jax.Array, jax.Array]:
            key_i, sample_key = jax.random.split(key_i)
            return key_i, jax.random.randint(sample_key, (batch,), 0, size_)

        _key, idx = jax.lax.scan(body, key_, jnp.arange(steps))
        return idx

    gemm_ms, _ = _median_ms(lambda: gemm_sgd(jax.tree.map(jnp.array, problem["diff"])), repeats)
    adam_ms, _ = _median_ms(
        lambda: adam_only(
            jax.tree.map(jnp.array, problem["diff"]),
            jax.tree.map(jnp.array, problem["opt_state"]),
            grad0,
        ),
        repeats,
    )
    rng_ms, _ = _median_ms(lambda: rng_only(key, size), repeats)
    return {"network step (sgd, not adam)": gemm_ms, "adam update only": adam_ms, "split and randint only": rng_ms}


def _compare(updated: Any, reference: Any, flat: bool) -> str:
    got = np.asarray(updated[0]) if flat else _flat_params(updated[0])
    exp = _flat_params(reference[0])
    params = float(np.max(np.abs(got - exp)))
    pre = float(np.max(np.abs(np.asarray(updated[-1] if flat else updated[2]) - np.asarray(reference[2]))))
    return f"params {params:.3e}, per-step loss {pre:.3e}"


def main() -> None:
    """Print the lever table for a 1000-by-256 critic, or the size given on the command line."""
    parser = argparse.ArgumentParser(description="Time critic Adam scan levers.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    problem = _problem(args.capacity)
    data = problem["data"]
    print(
        f"device {jax.devices()[0]}  x64 {jax.config.jax_enable_x64}  "
        f"steps {args.steps}  batch {args.batch}  capacity {args.capacity}",
        flush=True,
    )

    # The shipped scan is unrolled. This row is the previous unroll of 1.
    baseline_fn = _tree_scan(1, args.steps, args.batch, "inside", False)

    def call_baseline() -> Any:
        return baseline_fn(
            data,
            jax.tree.map(jnp.array, problem["diff"]),
            jax.tree.map(jnp.array, problem["opt_state"]),
            problem["static"],
            problem["tx"],
        )

    base_ms, base_out = _median_ms(call_baseline, args.repeats)
    rows: list[tuple[str, float, str]] = [("unroll 1 (previous scan)", base_ms, "reference")]
    print(f"unroll 1  {base_ms:.1f} ms", flush=True)

    for unroll in (2, 4, 8, 16):
        fn = _tree_scan(unroll, args.steps, args.batch, "inside", False)

        def call(fn: Any = fn) -> Any:
            return fn(
                data,
                jax.tree.map(jnp.array, problem["diff"]),
                jax.tree.map(jnp.array, problem["opt_state"]),
                problem["static"],
                problem["tx"],
            )

        ms, out = _median_ms(call, args.repeats)
        rows.append((f"scan unroll {unroll}", ms, _compare(out, base_out, flat=False)))
        print(f"unroll {unroll}  {ms:.1f} ms  {rows[-1][2]}", flush=True)

    variants: list[tuple[str, Any, bool]] = []
    variants.append(("indices up front, same splits", _tree_scan(1, args.steps, args.batch, "upfront", False), False))
    oneshot = _tree_scan(1, args.steps, args.batch, "oneshot", False)
    variants.append(("indices one randint (different stream)", oneshot, False))
    variants.append(("pregather, same splits", _tree_scan(1, args.steps, args.batch, "upfront", True), False))
    for name, fn, is_flat in variants:
        def call(fn: Any = fn) -> Any:
            return fn(
                data,
                jax.tree.map(jnp.array, problem["diff"]),
                jax.tree.map(jnp.array, problem["opt_state"]),
                problem["static"],
                problem["tx"],
            )

        ms, out = _median_ms(call, args.repeats)
        note = _compare(out, base_out, flat=False) if not is_flat else ""
        if "different stream" in name:
            note = "rng stream differs; " + _shared_index_check(problem, args.steps, args.batch)
        rows.append((name, ms, note))
        print(f"{name}  {ms:.1f} ms  {note}", flush=True)

    flat_optax = _flat_scan(args.steps, args.batch, 1, False, problem["unravel"])

    def call_flat() -> Any:
        return flat_optax(
            data,
            jnp.array(problem["flat"]),
            jax.tree.map(jnp.array, problem["opt_flat"]),
            problem["static"],
            problem["tx_flat"],
        )

    ms, out = _median_ms(call_flat, args.repeats)
    rows.append(("flat optax adam", ms, _compare(out, base_out, flat=True)))
    print(f"flat optax  {ms:.1f} ms  {rows[-1][2]}", flush=True)

    flat_f32 = _flat_scan(args.steps, args.batch, 1, True, problem["unravel"])

    def call_f32() -> Any:
        zeros = jnp.zeros_like(problem["flat"])
        return flat_f32(data, jnp.array(problem["flat"]), zeros, zeros, problem["static"])

    ms, out = _median_ms(call_f32, args.repeats)
    rows.append(("flat adam, float32 bias correction", ms, _compare(out, base_out, flat=True)))
    print(f"flat f32  {ms:.1f} ms  {rows[-1][2]}", flush=True)

    shipped_ms, shipped = _median_ms(
        lambda: _critic_scan(
            data,
            jax.tree.map(jnp.array, problem["diff"]),
            problem["static"],
            jax.tree.map(jnp.array, problem["opt_state"]),
            problem["tx"],
            args.steps,
            args.batch,
        ),
        args.repeats,
    )
    rows.append(("shipped _critic_scan", shipped_ms, _compare(shipped, base_out, flat=False)))
    print(f"shipped  {shipped_ms:.1f} ms  {rows[-1][2]}", flush=True)

    hlo_args = (data, problem["diff"], problem["opt_state"], problem["static"], problem["tx"])
    scalar, array = _f64_count(baseline_fn, hlo_args)
    print(f"HLO unroll 1: f64 scalars {scalar}, f64 arrays {array}", flush=True)
    profile = _profile(problem, args.steps, args.batch, args.repeats)
    print("profile (same step count)", flush=True)
    for name, value in profile.items():
        print(f"  {name}: {value:.1f} ms", flush=True)

    print("\n| lever | critic ms | speedup | equivalence |", flush=True)
    print("|---|---:|---:|---|", flush=True)
    for name, ms, note in rows:
        print(f"| {name} | {ms:.1f} | {base_ms / ms:.2f}x | {note} |", flush=True)


if __name__ == "__main__":
    main()
