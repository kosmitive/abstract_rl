"""Plot the learning curve (mean episodic return per epoch) of one or more runs.

Usage::

    uv run python -m abstract_rl.scripts.plot_learning_curve data/Pendulum-v1/trpo/<run> -o docs/trpo_pendulum.png
"""

import argparse
import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np


def load_returns(run_dir: str, field: str = "sreward") -> np.ndarray:
    """Load the logged [mean, std] return of each epoch of a run as an array of shape [epochs, 2]."""
    return np.loadtxt(join(run_dir, "data", f"{field}.txt")).reshape(-1, 2)


def steps_per_epoch(run_dir: str) -> int:
    """Number of environment steps collected per epoch, read from the stored run configuration."""
    with open(join(run_dir, "run.conf")) as f:
        conf = json.load(f)
    return conf["env_sample"]


def plot(run_dirs: list[str], labels: list[str], output: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir, label in zip(run_dirs, labels):
        returns = load_returns(run_dir)
        steps = (np.arange(len(returns)) + 1) * steps_per_epoch(run_dir)
        mean, std = returns[:, 0], returns[:, 1]
        ax.plot(steps, mean, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (stochastic policy)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("-o", "--output", default="learning_curve.png")
    parser.add_argument("--title", default="Pendulum-v1")
    args = parser.parse_args()
    plot(args.run_dirs, args.labels or args.run_dirs, args.output, args.title)
