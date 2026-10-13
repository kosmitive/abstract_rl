"""Verify that trained Pendulum-v1 policies swing up and balance the pendulum.

Every run is evaluated with the policy mean on seeded start states. For each episode the pole angle
theta = atan2(sin, cos) is logged. An episode counts as balanced if |theta| stays below ``--tol`` (default 0.1 rad)
for the last ``--window`` steps (default 100). Per run a plot of theta over time (all episodes) and optionally a GIF
of the hardest episode are written. The exit code is non-zero if any run balances fewer than ``--min-balanced`` of its
episodes, so the script can be used as an automated check.

Usage::

    uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run> --out eval/ --gif
"""

import argparse
import os
import sys
from os.path import basename, join, normpath
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from PIL import Image

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.select_policy import select_policy


def load_policy(run_dir: str) -> tuple[MCMCEnvWrapper, ContinuousPolicy]:
    """Rebuild environment wrapper and policy of a stored run and load the last policy parameters."""
    mc = ModelConfiguration({"reload": True, "reload_dir": run_dir})
    conf = mc["conf"]
    conf["env_render"] = False
    with conf.ns("env"):
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)
    with conf.ns("policy"):
        policy = select_policy(mc)
        policy.load_state_dict(torch.load(join(run_dir, "model/last", "policy.data"), weights_only=True))
    policy.eval()
    return env, policy


def run_episode(
    env_wrapper: MCMCEnvWrapper, policy: ContinuousPolicy, env: gym.Env, seed: int, frames: list[Any] | None = None
) -> tuple[float, ndarray]:
    """Run one episode with the policy mean. Returns the episodic return and the angle at each step."""
    obs, _ = env.reset(seed=seed)
    total, thetas, done = 0.0, [], False
    while not done:
        with torch.no_grad():
            s = torch.as_tensor(env_wrapper._preprocess(obs), dtype=torch.float32).view(1, -1)
            a = policy.mode(states=s).numpy()[0].astype(np.float64)
        obs, r, terminated, truncated, _ = env.step(a)
        total += float(r)
        thetas.append(float(np.arctan2(obs[1], obs[0])))
        if frames is not None:
            frames.append(env.render())
        done = terminated or truncated
    return total, np.asarray(thetas)


def evaluate(run_dir: str, episodes: int = 20, seed: int = 10_000) -> tuple[ndarray, ndarray]:
    """Evaluate a run. Returns (returns [E], thetas [E, T])."""
    env_wrapper, policy = load_policy(run_dir)
    returns, thetas = [], []
    for i in range(episodes):
        ret, th = run_episode(env_wrapper, policy, env_wrapper.env, seed + i)
        returns.append(ret)
        thetas.append(th)
    env_wrapper.close()
    return np.asarray(returns), np.stack(thetas)


def save_gif(run_dir: str, path: str, seed: int = 10_000, fps: int = 30) -> float:
    """Render one episode (rgb_array, no display needed) into a GIF."""
    env_wrapper, policy = load_policy(run_dir)
    env = gym.make(env_wrapper.env_name, render_mode="rgb_array")
    frames: list[Any] = []
    ret, _ = run_episode(env_wrapper, policy, env, seed, frames)
    env.close()
    env_wrapper.close()
    images = [Image.fromarray(f).resize((250, 250)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return ret


def plot_thetas(thetas: ndarray, returns: ndarray, tol: float, window: int, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(1, thetas.shape[1] + 1)
    for th in thetas:
        ax.plot(steps, th, color="tab:blue", alpha=0.35, lw=1)
    ax.axhspan(-tol, tol, color="tab:green", alpha=0.2, label=f"|theta| < {tol} rad")
    ax.axvline(thetas.shape[1] - window, color="gray", ls="--", lw=1, label=f"last {window} steps")
    ax.set_xlabel("step")
    ax.set_ylabel("theta [rad] (0 = upright)")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(f"{title}\nreturn {returns.mean():.1f} ± {returns.std():.1f} over {len(returns)} episodes")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000, help="start states are seeded with seed + episode")
    parser.add_argument("--tol", type=float, default=0.1, help="upright tolerance in rad")
    parser.add_argument("--window", type=int, default=100, help="number of final steps that must be upright")
    parser.add_argument("--min-balanced", type=float, default=1.0, help="required fraction of balanced episodes")
    parser.add_argument("--out", default="eval", help="output directory for plots and GIFs")
    parser.add_argument("--gif", action="store_true", help="also render the episode with the lowest return as GIF")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    ok, means = True, []
    for run_dir in args.run_dirs:
        returns, thetas = evaluate(run_dir, args.episodes, args.seed)
        upright = np.abs(thetas[:, -args.window :]) < args.tol
        balanced = upright.all(axis=1)
        means.append(returns.mean())

        parts = normpath(run_dir).split(os.sep)
        name = "_".join(parts[-3:]) if len(parts) >= 3 else basename(run_dir)
        plot_thetas(thetas, returns, args.tol, args.window, name, join(args.out, f"{name}_theta.png"))
        gif_msg = ""
        if args.gif:
            # render the hardest episode (lowest return, usually a swing-up from the bottom)
            gif_seed = args.seed + int(np.argmin(returns))
            gif_ret = save_gif(run_dir, join(args.out, f"{name}.gif"), gif_seed)
            gif_msg = f", gif (seed {gif_seed}) return {gif_ret:.1f}"

        print(
            f"{run_dir}: return {returns.mean():.1f} ± {returns.std():.1f} (min {returns.min():.1f}) | "
            f"upright {100 * upright.mean():.1f}% of the last {args.window} steps | "
            f"balanced {balanced.sum()}/{len(balanced)} episodes | "
            f"max |theta| in last {args.window} steps {np.abs(thetas[:, -args.window :]).max():.3f} rad{gif_msg}"
        )
        ok &= balanced.mean() >= args.min_balanced

    if len(means) > 1:
        print(f"mean return across runs: {np.mean(means):.1f} ± {np.std(means):.1f}")
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
