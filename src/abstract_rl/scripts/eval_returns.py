"""Evaluate trained policies with the policy mean on the raw environment return.

Usage::

    uv run python -m abstract_rl.scripts.eval_returns data/HalfCheetah-v5/trpo/<run> [more runs] --episodes 20

Per run the mean return over seeded start states is printed together with the success threshold of the environment
(``success`` of ``config/env/<env>.yaml``). The exit code is non-zero if a run stays below the threshold.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from abstract_rl.scripts.eval_balance import load_policy
from abstract_rl.scripts.hydra_conf import success


def evaluate(run_dir: str, episodes: int = 20, seed: int = 10_000, model: str = "last") -> tuple[str, np.ndarray]:
    """Returns of ``episodes`` greedy (policy mean) episodes of a stored run on the unmodified environment."""
    env_wrapper, policy = load_policy(run_dir, model)
    env = env_wrapper.env
    max_steps = env_wrapper.conf.get_root("max_steps")
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, steps, done = 0.0, 0, False
        while not done and (max_steps is None or steps < max_steps):
            with torch.no_grad():
                s = torch.as_tensor(env_wrapper._preprocess(obs), dtype=torch.float32).view(1, -1)
                a = policy.mode(states=s).reshape(-1).numpy().astype(np.float64)
            obs, r, terminated, truncated, _ = env.step(a)
            total += float(r)
            steps += 1
            done = terminated or truncated
        returns.append(total)
    name = env_wrapper.env_name
    env_wrapper.close()
    return name, np.asarray(returns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000, help="start states are seeded with seed + episode")
    parser.add_argument("--json", default=None, help="write the results as JSON to this file")
    parser.add_argument("--save", action="store_true", help="store the returns as eval.json in each run directory")
    parser.add_argument("--model", choices=["last", "best"], default="last", help="stored policy to evaluate")
    args = parser.parse_args(argv)

    results, ok = {}, True
    for run_dir in args.run_dirs:
        env_id, returns = evaluate(run_dir, args.episodes, args.seed, args.model)
        threshold = success().get(env_id)
        solved = bool(threshold is not None and returns.mean() >= threshold)
        ok &= solved
        results[run_dir] = {"env": env_id, "mean": float(returns.mean()), "std": float(returns.std()), "solved": solved}
        if args.save:
            with open(os.path.join(run_dir, "eval.json"), "w") as f:
                json.dump({"env": env_id, "returns": returns.tolist()}, f)
        print(f"{run_dir}: return {returns.mean():.1f} ± {returns.std():.1f} (threshold {threshold}) solved={solved}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
