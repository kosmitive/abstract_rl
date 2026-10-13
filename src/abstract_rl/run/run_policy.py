"""Replay a stored policy: ``uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render``."""

import argparse
from os.path import join

import numpy as np
import torch

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.policy.continuous.select_policy import select_policy


def run_policy(run_dir: str, episodes: int = 5, render: bool = False, stochastic: bool = False) -> list[float]:
    """
    Loads the last stored policy of a run and executes it.
    :param run_dir: The run directory, containing run.conf and model/last/policy.data.
    :param episodes: The number of episodes to execute.
    :param render: Render the environment.
    :param stochastic: Sample actions instead of using the mode of the policy.
    :return: The list of episodic returns.
    """

    # the model configuration loads run.conf from the run directory
    mc = ModelConfiguration({"reload": True, "reload_dir": run_dir})
    conf = mc["conf"]
    conf["env_render"] = render

    # create an environment wrapper
    with conf.ns("env"):
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)

    # policy neural_modules
    with conf.ns("policy"):
        policy = select_policy(mc)
        policy.load_state_dict(torch.load(join(run_dir, "model/last", "policy.data"), weights_only=True))

    returns = []
    with conf.ns("policy"):
        for _ in range(episodes):
            traj = env.execute_policy_once(conf.get_root("max_steps"), policy, render, opt=not stochastic)
            returns.append(traj.total_reward)
            print(f"return: {traj.total_reward:.1f}")
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
