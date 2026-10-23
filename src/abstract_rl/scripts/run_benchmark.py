"""Run the README benchmark (environment x algorithm cells, several seeds each) and aggregate it.

Usage::

    # train, then evaluate every run greedily (20 episodes, eval.json); one cell after the other
    uv run python -m abstract_rl.scripts.run_benchmark --envs pendulum halfcheetah --algs trpo mpo \\
        --seeds 1 2 3 4 5 --root data/bench --threads 1
    # aggregate all finished cells: Markdown table and learning curves (compare.py), merged timing.json
    uv run python -m abstract_rl.scripts.run_benchmark --report --root data/bench --out docs/

Every cell trains its seeds as one vmapped program (``scripts.run.train`` with ``seeds``) on the CPU, writing to
``<root>/runs/<env>_<alg>/<env id>/<alg>/<timestamp>/``. With ``--mpo-processes`` every MPO seed is trained in its own
process instead. Cells are independent, so several invocations with disjoint ``--envs``/``--algs`` can run in parallel
on the same ``--root``. Invocations of one cell with disjoint ``--seeds`` (e.g. one seed at a time when the vmapped
seeds do not fit into memory) should run one after the other: this invocation's runs are the new run directories of
the cell with one of its seeds (read from ``run.conf``) that were created after it started, so disjoint seeds do not
clobber each other, but the same seed run twice in parallel would. Each invocation writes its wall time and
env-steps/s per cell to ``<root>/timing/<env>_<alg>_seeds<seeds>.json`` and ``--report`` merges these into
``<root>/timing.json``.
``--threads 1`` disables XLA's multi-threaded CPU kernels (set before JAX is imported). Hydra overrides after
``--set`` apply to every cell, e.g. ``--set max_env_steps=8000`` or ``--set alg.num_epochs=2``. A cell which already
holds runs of a requested seed is refused unless ``--fresh`` deletes these runs.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from glob import glob
from typing import Any

ENVS = ["pendulum", "mountaincar", "inverted_pendulum", "inverted_double_pendulum", "halfcheetah"]
ALGS = ["trpo", "mpo"]


def set_threads(threads: int | None) -> None:
    """Restrict the CPU threads of XLA (and of BLAS/OpenMP libraries); only effective before JAX is imported."""
    if threads is None:
        return
    if "jax" in sys.modules:
        raise RuntimeError("--threads must be applied before JAX is imported")
    # XLA has no flag for the size of its CPU thread pool (an ``intra_op_parallelism_threads=N`` entry is not a flag:
    # XLA aborts on it unless a ``--`` flag comes first, and ignores it otherwise); a single thread disables the
    # multi-threaded Eigen kernels (matmuls, convolutions), the rest of a compiled program runs on one thread anyway
    flags = [f for f in os.environ.get("XLA_FLAGS", "").split() if "multi_thread_eigen" not in f]
    if threads == 1:
        flags.append("--xla_cpu_multi_thread_eigen=false")
    os.environ["XLA_FLAGS"] = " ".join(flags)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)


def cell_root(root: str, env: str, alg: str) -> str:
    return os.path.join(root, "runs", f"{env}_{alg}")


def run_dirs(path: str) -> set[str]:
    """Run directories (``run.conf`` present) below ``path``."""
    return {os.path.dirname(p) for p in glob(os.path.join(path, "*", "*", "*", "run.conf"))}


def seed_of(run_dir: str) -> int:
    with open(os.path.join(run_dir, "run.conf")) as f:
        return int(json.load(f)["seed"])


def env_steps(run_dir: str) -> int:
    import numpy as np

    path = os.path.join(run_dir, "data", "steps.txt")
    return int(np.loadtxt(path).sum()) if os.path.exists(path) else 0


def train_cell(env: str, alg: str, seeds: list[int], data_root: str, overrides: list[str]) -> None:
    """Train all ``seeds`` of one cell in this process (one vmapped program)."""
    from abstract_rl.scripts.hydra_conf import load, run_dict
    from abstract_rl.scripts.run import train

    cfg = load([f"env={env}", f"alg={alg}", "jax_platform=cpu", *overrides])
    conf = run_dict(cfg)
    conf.update({"seed": seeds[0], "seeds": seeds, "data_root": data_root})
    train(alg, conf)


def train_processes(env: str, alg: str, seeds: list[int], args: argparse.Namespace, data_root: str) -> None:
    """Train every seed of one cell in its own process, all at once."""
    procs = []
    for seed in seeds:
        cmd = [
            sys.executable,
            "-m",
            "abstract_rl.scripts.run_benchmark",
            "--train-only",
            "--envs",
            env,
            "--algs",
            alg,
            "--seeds",
            str(seed),
            "--data-root",
            data_root,
        ]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        if args.set:
            cmd += ["--set", *args.set]
        log = open(os.path.join(data_root, f"seed{seed}.log"), "w")  # noqa: SIM115
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for (proc, log), seed in zip(procs, seeds, strict=True):
        if proc.wait() != 0:
            failed.append(seed)
        log.close()
    if failed:
        raise RuntimeError(f"{env} {alg}: training failed for seeds {failed}, see {data_root}/seed*.log")


def run_cell(env: str, alg: str, args: argparse.Namespace) -> dict[str, Any]:
    """Train, evaluate and time one cell. Returns its timing record."""
    from abstract_rl.scripts import eval_returns

    root = cell_root(args.root, env, alg)
    existing = sorted(d for d in run_dirs(root) if seed_of(d) in args.seeds)
    if existing:
        if not args.fresh:
            raise SystemExit(f"{root} already holds runs of seeds {args.seeds}, pass --fresh to delete them")
        for d in existing:
            shutil.rmtree(d)
    os.makedirs(root, exist_ok=True)
    before = run_dirs(root)

    start_wall = time.time()
    start = time.perf_counter()
    if alg == "mpo" and args.mpo_processes and len(args.seeds) > 1:
        train_processes(env, alg, args.seeds, args, root)
    else:
        train_cell(env, alg, args.seeds, root, args.set or [])
    train_time = time.perf_counter() - start
    # new run directories of this invocation's seeds, whose run.conf was written after it started (clock slack 2 s)
    dirs = sorted(
        d
        for d in run_dirs(root) - before
        if seed_of(d) in args.seeds and os.path.getmtime(os.path.join(d, "run.conf")) >= start_wall - 2
    )

    start = time.perf_counter()
    eval_returns.main([*dirs, "--episodes", str(args.episodes), "--save"])
    eval_time = time.perf_counter() - start

    steps = [env_steps(d) for d in dirs]
    record = {
        "env": env,
        "alg": alg,
        "seeds": args.seeds,
        "mpo_processes": bool(alg == "mpo" and args.mpo_processes),
        "threads": args.threads,
        "overrides": args.set or [],
        "run_dirs": dirs,
        "env_steps": steps,
        "train_seconds": round(train_time, 1),
        "eval_seconds": round(eval_time, 1),
        # summed over seeds, the throughput of the whole cell
        "env_steps_per_second": round(sum(steps) / train_time, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.join(args.root, "timing"), exist_ok=True)
    # one file per cell and seed set: parallel invocations on the same root never write the same file
    path = os.path.join(args.root, "timing", f"{env}_{alg}_seeds{'-'.join(map(str, args.seeds))}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(path + ".tmp", path)
    print(
        f"{env} {alg}: {sum(steps):,} env steps in {train_time:.0f} s ({record['env_steps_per_second']:.0f}/s), "
        f"eval {eval_time:.0f} s"
    )
    return record


def report(args: argparse.Namespace) -> None:
    """Merge the per-cell timing files into ``timing.json`` and run compare.py on all runs below the root."""
    from abstract_rl.scripts import compare

    records = []
    for path in sorted(glob(os.path.join(args.root, "timing", "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    with open(os.path.join(args.root, "timing.json"), "w") as f:
        json.dump(records, f, indent=1)
    for r in records:
        print(
            f"{r['env']:26s} {r['alg']:5s} train {r['train_seconds']:9.0f} s  eval {r['eval_seconds']:6.0f} s  "
            f"{r['env_steps_per_second']:9.0f} env steps/s"
        )
    compare.main([args.root, "--out", args.out])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=ENVS, choices=ENVS)
    parser.add_argument("--algs", nargs="+", default=ALGS, choices=ALGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--root", default="data/bench")
    parser.add_argument("--episodes", type=int, default=20, help="greedy evaluation episodes per run")
    parser.add_argument("--threads", type=int, default=None, help="1: single-threaded XLA CPU kernels")
    parser.add_argument("--mpo-processes", action="store_true", help="train every MPO seed in its own process")
    parser.add_argument("--fresh", action="store_true", help="delete existing runs of the requested seeds first")
    parser.add_argument("--set", nargs="+", default=None, metavar="OVERRIDE", help="Hydra overrides for every cell")
    parser.add_argument("--report", action="store_true", help="only aggregate the finished cells below --root")
    parser.add_argument("--out", default="docs", help="--report: directory of benchmark.png and benchmark.md")
    # internal: one seed process of --mpo-processes
    parser.add_argument("--train-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    set_threads(args.threads)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.report:
        report(args)
    elif args.train_only:
        train_cell(args.envs[0], args.algs[0], args.seeds, args.data_root, args.set or [])
    else:
        for env in args.envs:
            for alg in args.algs:
                run_cell(env, alg, args)


if __name__ == "__main__":
    main()
