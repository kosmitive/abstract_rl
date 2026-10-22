"""Replay a stored policy: ``uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render``."""

import argparse
from os.path import join
from typing import Any

import numpy as np
from PIL import Image

from abstract_rl.scripts.eval_balance import load_policy, run_episode


def run_policy(dir: str, episodes: int = 5, render: bool = False, stochastic: bool = False) -> list[float]:
    """
    Loads the last stored policy of a run and executes it.
    :param dir: The run directory, containing run.conf and model/last/policy.eqx.
    :param episodes: The number of episodes to execute.
    :param render: Render every episode offscreen into ``replay_<episode>.gif`` in the run directory.
    :param stochastic: Sample actions instead of using the mode of the policy.
    :return: The list of episodic returns.
    """

    # the model configuration loads run.conf from the run directory, the frozen statistics come with the policy
    env, policy, state = load_policy(dir)
    seed = env.conf.get_root("seed")

    returns = []
    for i in range(episodes):
        frames: list[Any] | None = [] if render else None
        ret, _ = run_episode(env, policy, state, seed + i, frames, stochastic=stochastic)
        returns.append(ret)
        print(f"return: {ret:.1f}")
        if frames:
            images = [Image.fromarray(f) for f in frames]
            images[0].save(join(dir, f"replay_{i}.gif"), save_all=True, append_images=images[1:], duration=33, loop=0)
    env.close()
    print(f"mean return over {episodes} episodes: {np.mean(returns):.1f} +- {np.std(returns):.1f}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    run_policy(args.run_dir, args.episodes, args.render, args.stochastic)
