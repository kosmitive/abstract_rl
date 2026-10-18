"""Time N epochs of Pendulum-v1 TRPO and MPO on CPU, eager against ``torch_compile``.

A compiled warmup epoch fills the compiler cache. The timed runs are the N epochs after that.
``torch.compile`` does not support the double backward inside TRPO's Fisher-vector product, so that
product and the KL stay eager; the policy forward, the value or Q forward and the TRPO surrogate are compiled.

    uv run python scripts/bench_compile.py --epochs 2
"""

import argparse
import tempfile
import time
from typing import Any

from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts.hydra_conf import make_config


def _run(alg: str, epochs: int, compile: bool, root: str) -> float:
    """Train one Pendulum configuration and return the wall time in seconds."""
    conf: dict[str, Any] = make_config("pendulum", alg)
    conf["num_epochs"] = epochs
    conf["torch_compile"] = compile
    conf["data_root"] = root
    match alg:
        case "trpo":
            runner = run_trpo
        case "mpo":
            runner = run_mpo_non_parametric
        case _:
            raise ValueError(f"unknown algorithm {alg}")
    started = time.perf_counter()
    runner(conf)
    return time.perf_counter() - started


def main() -> None:
    """Print eager and compiled epoch times for Pendulum-v1 TRPO and MPO."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--epochs", type=int, default=2, help="timed epochs per algorithm and mode")
    parser.add_argument("--algs", nargs="+", default=["trpo", "mpo"], choices=["trpo", "mpo"])
    args = parser.parse_args()
    print(f"Pendulum-v1, CPU, {args.epochs} timed epochs (compiled warmup excluded)")
    for alg in args.algs:
        with tempfile.TemporaryDirectory() as root:
            eager = _run(alg, args.epochs, False, f"{root}/eager")
            _run(alg, 1, True, f"{root}/warm")
            compiled = _run(alg, args.epochs, True, f"{root}/compiled")
        ratio = eager / compiled if compiled else float("inf")
        print(f"{alg:5} eager {eager:8.2f}s  compiled {compiled:8.2f}s  speedup {ratio:.2f}x")


if __name__ == "__main__":
    main()
