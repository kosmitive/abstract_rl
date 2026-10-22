"""Throughput of the JAX TRPO training loop: environment steps per second summed over seeds.

``uv run python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6``. The first epoch call
compiles and is reported separately.
"""

import argparse
import json
import tempfile
import time
from typing import Any

from abstract_rl.scripts.hydra_conf import load, run_dict


def bench(env: str, seeds: int, epochs: int) -> dict[str, Any]:
    from abstract_rl.run import run_trpo as module

    conf = run_dict(load([f"env={env}", "alg=trpo"]))
    conf.update({"seeds": list(range(1, seeds + 1)), "data_root": tempfile.mkdtemp()})
    times: list[float] = []
    original = module.SeedRuns.store

    def timed_store(self: Any, name: str, models: Any) -> None:
        times.append(time.perf_counter())
        original(self, name, models)

    module.SeedRuns.store = timed_store  # type: ignore[method-assign]
    try:
        start = time.perf_counter()
        conf["num_epochs"] = epochs
        returns = module.run_trpo(conf)
    finally:
        module.SeedRuns.store = original  # type: ignore[method-assign]
    steps_per_epoch = conf["policy_batch_size"] * seeds
    first = times[0] - start
    steady = (times[-1] - times[0]) / max(len(times) - 1, 1)
    return {"env": env, "seeds": seeds, "first_epoch_s": first, "epoch_s": steady,
            "env_steps_per_s": steps_per_epoch / steady, "final_return_mean": float(returns[:, -1, 0].mean())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="pendulum")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    results = [bench(args.env, s, args.epochs) for s in args.seeds]
    for r in results:
        print(f"{r['env']:12s} seeds {r['seeds']:3d} | first call {r['first_epoch_s']:7.2f}s "
              f"| epoch {r['epoch_s']:7.3f}s | {r['env_steps_per_s']:10.0f} env-steps/s")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
