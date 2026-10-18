"""Summarise and plot the benchmark runs of abstract_rl (TRPO, MPO) and the Stable-Baselines3 baselines.

Usage::

    uv run python -m abstract_rl.scripts.compare <results dir> --out docs/

The results directory is searched recursively for abstract_rl runs (``run.conf`` plus ``data/*.txt``, the greedy
evaluation ``eval.json`` written by ``eval_returns --json`` next to ``run.conf``) and SB3 runs (``run.json`` plus
``monitor.csv``). For every environment a learning curve plot (training return over environment steps, mean ± std over
seeds) and a Markdown table with the greedy evaluation are written. Learning curves are cut at the environment-step
budget (``steps`` in ``config/env``; the episode crossing it is kept), so runs that stop after a fixed number of
epochs and overshoot the budget are compared on equal steps; the greedy return is the one of the final policy.
"""

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np
from numpy import ndarray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abstract_rl.scripts.hydra_conf import environments, success
from abstract_rl.scripts.hydra_conf import steps as steps_budget

Runs = dict[tuple[str, str], dict[int, tuple[ndarray, ndarray, ndarray | None]]]

LABELS = {"trpo": "TRPO (abstract_rl)", "mpo": "MPO (abstract_rl)", "sb3_trpo": "TRPO (SB3)", "sb3_ppo": "PPO (SB3)"}
COLORS = {"trpo": "C0", "mpo": "C1", "sb3_trpo": "C2", "sb3_ppo": "C3"}


def load_runs(root: str) -> Runs:
    """Find all runs below ``root``. Returns ``{(env, alg): {seed: (steps, returns, greedy_returns or None)}}``."""
    runs: Runs = defaultdict(dict)
    for folder, _, files in os.walk(root, followlinks=True):
        if "run.conf" in files and os.path.isdir(os.path.join(folder, "data")):
            with open(os.path.join(folder, "run.conf")) as f:
                conf = json.load(f)
            try:
                returns = np.loadtxt(os.path.join(folder, "data", "sreward.txt")).reshape(-1, 2)[:, 0]
                steps = np.cumsum(np.loadtxt(os.path.join(folder, "data", "steps.txt")).reshape(-1))
            except (OSError, ValueError):
                continue
            n = min(len(returns), len(steps))
            greedy = None
            if "eval.json" in files:
                with open(os.path.join(folder, "eval.json")) as f:
                    greedy = json.load(f)["returns"]
            runs[(conf["env_name"], conf["alg"])][conf["seed"]] = (steps[:n], returns[:n], greedy)
        elif "run.json" in files and "monitor.csv" in files:
            with open(os.path.join(folder, "run.json")) as f:
                meta = json.load(f)
            data = np.genfromtxt(os.path.join(folder, "monitor.csv"), delimiter=",", skip_header=2)
            data = np.atleast_2d(data)
            # vectorised envs step in lockstep and only the first one is monitored: scale to the total step count
            n_envs = meta.get("hyper", {}).get("n_envs", 1)
            steps = n_envs * np.cumsum(data[:, 1])
            runs[(meta["env"], meta["alg"])][meta["seed"]] = (steps, data[:, 0], meta["returns"])
    return {key: {seed: truncate(key[0], *run) for seed, run in seeds.items()} for key, seeds in runs.items()}


def truncate(
    env: str, steps: ndarray, returns: ndarray, greedy: ndarray | None
) -> tuple[ndarray, ndarray, ndarray | None]:
    """Cut a learning curve after the episode (or epoch) that crosses the environment-step budget of ``env``."""
    budget = steps_budget()
    if env not in budget:
        return steps, returns, greedy
    n = int(np.searchsorted(steps, budget[env], side="left")) + 1
    return steps[:n], returns[:n], greedy


def curve(steps: ndarray, returns: ndarray, grid: Sequence[float] | ndarray, window: float) -> ndarray:
    """Mean return of the episodes (or epochs) ending within ``window`` steps before each grid point."""
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        mask = (steps <= g) & (steps > g - window)
        if mask.any():
            out[i] = returns[mask].mean()
        elif (steps <= g).any():
            out[i] = returns[steps <= g][-1]
    return out


def steps_to(steps: ndarray, returns: ndarray, threshold: float, window: float) -> float:
    """First step at which the smoothed training return reaches the threshold, nan if never."""
    for g in np.arange(window, steps[-1] + window, window / 2):
        c = curve(steps, returns, [g], window)[0]
        if c >= threshold:
            return g
    return np.nan


def summarise(runs: Runs, env: str) -> list[dict[str, Any]]:
    """Rows of the table of an environment."""
    rows = []
    for alg in LABELS:
        seeds = runs.get((env, alg))
        if not seeds:
            continue
        threshold = success()[env]
        greedy = np.array([np.mean(g) for _, _, g in seeds.values() if g is not None])
        window = max(2000, int(max(s[-1] for s, _, _ in seeds.values()) / 100))
        reach = np.array([steps_to(s, r, threshold, window) for s, r, _ in seeds.values()])
        final = np.array([r[-max(3, len(r) // 20) :].mean() for _, r, _ in seeds.values()])
        rows.append(
            {
                "alg": alg,
                "seeds": len(seeds),
                "steps": int(np.median([s[-1] for s, _, _ in seeds.values()])),
                "greedy_mean": float(greedy.mean()) if len(greedy) else float("nan"),
                "greedy_std": float(greedy.std()) if len(greedy) else float("nan"),
                "solved": int((greedy >= threshold).sum()) if len(greedy) else 0,
                "train_final": float(final.mean()),
                "steps_to": float(np.nanmedian(reach)) if np.isfinite(reach).any() else float("nan"),
                "reached": int(np.isfinite(reach).sum()),
            }
        )
    return rows


def table(runs: Runs, envs: Sequence[str]) -> str:
    lines = [
        "| Environment | Algorithm | Seeds | Steps | Greedy return (mean ± std) | Solved | Steps to threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for env in envs:
        for row in summarise(runs, env):
            steps_to_thr = f"{row['steps_to']:,.0f} ({row['reached']}/{row['seeds']})" if row["reached"] else "never"
            lines.append(
                f"| {env} (≥ {success()[env]:g}) | {LABELS[row['alg']]} | {row['seeds']} | {row['steps']:,} | "
                f"{row['greedy_mean']:.1f} ± {row['greedy_std']:.1f} | {row['solved']}/{row['seeds']} | "
                f"{steps_to_thr} |"
            )
    return "\n".join(lines) + "\n"


def plot(runs: Runs, envs: Sequence[str], output: str) -> None:
    envs = [e for e in envs if any(k[0] == e for k in runs)]
    cols = min(len(envs), 3)
    rows = int(np.ceil(len(envs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)
    for ax, env in zip(axes.flat, envs, strict=False):
        for alg in LABELS:
            seeds = runs.get((env, alg))
            if not seeds:
                continue
            horizon = max(s[-1] for s, _, _ in seeds.values())
            grid = np.linspace(horizon / 100, horizon, 100)
            window = horizon / 50
            curves = np.stack([curve(s, r, grid, window) for s, r, _ in seeds.values()])
            mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0)
            ax.plot(grid, mean, color=COLORS[alg], label=f"{LABELS[alg]} ({len(seeds)})")
            ax.fill_between(grid, mean - std, mean + std, color=COLORS[alg], alpha=0.15)
        ax.axhline(success()[env], color="gray", ls=":", lw=1)
        ax.set_title(env)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("training return")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    for ax in list(axes.flat)[len(envs) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    runs = load_runs(args.results)
    os.makedirs(args.out, exist_ok=True)
    text = table(runs, environments())
    with open(os.path.join(args.out, "benchmark.md"), "w") as f:
        f.write(text)
    plot(runs, environments(), os.path.join(args.out, "benchmark.png"))
    print(text)


if __name__ == "__main__":
    main()
