"""Train a Stable-Baselines3 baseline (PPO or sb3-contrib TRPO) on a benchmark environment.

Needs the optional extra ``baselines``. The evaluation protocol equals the one of the abstract_rl runs: the training
episode returns are recorded per environment step (Monitor), and after training the greedy policy is evaluated on 20
episodes with the start states seeded 10000 + episode, reporting the raw environment return.

Usage::

    uv run --extra baselines python -m abstract_rl.scripts.run_baselines Pendulum-v1 ppo --seed 1 --steps 800000
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np

from abstract_rl.scripts.hydra_conf import steps as steps_budget

ALGORITHMS_SB3 = ["ppo", "trpo"]

# rl-baselines3-zoo hyperparameters (hyperparams/{ppo,trpo}.yml), used with ``--zoo``; the zoo budget is replaced
# by the equal env-step budget (``steps`` in ``config/env``). ``n_envs`` and ``normalize`` are handled by ``run``.
ZOO: dict[str, dict[str, dict[str, Any]]] = {
    "Pendulum-v1": {
        "ppo": {
            "n_envs": 4,
            "n_steps": 1024,
            "gae_lambda": 0.95,
            "gamma": 0.9,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 1e-3,
            "clip_range": 0.2,
            "use_sde": True,
            "sde_sample_freq": 4,
        },
        "trpo": {
            "n_envs": 2,
            "n_steps": 1024,
            "gamma": 0.9,
            "n_critic_updates": 15,
            "use_sde": True,
            "sde_sample_freq": 4,
        },
    },
    "MountainCarContinuous-v0": {
        "ppo": {
            "normalize": True,
            "n_envs": 1,
            "batch_size": 256,
            "n_steps": 8,
            "gamma": 0.9999,
            "learning_rate": 7.77e-05,
            "ent_coef": 0.00429,
            "clip_range": 0.1,
            "n_epochs": 10,
            "gae_lambda": 0.9,
            "max_grad_norm": 5,
            "vf_coef": 0.19,
            "use_sde": True,
            "policy_kwargs": {"log_std_init": -3.29, "ortho_init": False},
        },
        "trpo": {"normalize": True, "n_envs": 2, "use_sde": True, "sde_sample_freq": 4},
    },
}


def evaluate(model: Any, env_id: str, episodes: int = 20, seed: int = 10_000, norm: Any = None) -> np.ndarray:
    """Returns of ``episodes`` greedy episodes on the unmodified environment."""

    env = gym.make(env_id)
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            o = norm.normalize_obs(obs) if norm is not None else obs
            action, _ = model.predict(o, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return np.asarray(returns)


def run(
    env_id: str,
    alg: str,
    seed: int = 1,
    steps: int | None = None,
    data_root: str = "data",
    hyper: dict[str, Any] | None = None,
) -> str:
    """Train and evaluate one run, returns the run directory."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    steps = steps or steps_budget()[env_id]
    run_dir = os.path.join(data_root, env_id, f"sb3_{alg}", datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    os.makedirs(run_dir)
    torch.set_num_threads(1)
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    kw = dict(hyper or {})
    n_envs, normalize = kw.pop("n_envs", 1), kw.pop("normalize", False)
    monitors = [os.path.join(run_dir, "monitor.csv" if i == 0 else f"env{i}.monitor.csv") for i in range(n_envs)]

    def make(f: str) -> Any:
        return lambda: Monitor(gym.make(env_id), f)

    env: Any = DummyVecEnv([make(f) for f in monitors])
    if normalize:
        env = VecNormalize(env)
    if alg == "ppo":
        model: Any = PPO("MlpPolicy", env, seed=seed, device="cpu", **kw)
    elif alg == "trpo":
        from sb3_contrib import TRPO

        model = TRPO("MlpPolicy", env, seed=seed, device="cpu", **kw)
    else:
        raise KeyError(f"unknown baseline {alg}, available: {', '.join(ALGORITHMS_SB3)}")
    model.learn(total_timesteps=steps)
    norm = env if normalize else None
    if norm is not None:
        norm.training = False
    env.close()
    returns = evaluate(model, env_id, norm=norm)
    result = {
        "env": env_id,
        "alg": f"sb3_{alg}",
        "seed": seed,
        "steps": steps,
        "hyper": hyper or {},
        "returns": returns.tolist(),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(result, f)
    print(f"{run_dir}: greedy return {returns.mean():.1f} ± {returns.std():.1f}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env", choices=list(steps_budget()))
    parser.add_argument("alg", choices=ALGORITHMS_SB3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="environment steps (default: the budget of the env)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--zoo", action="store_true", help="use the rl-baselines3-zoo hyperparameters (ZOO)")
    parser.add_argument("--hyper", default="{}", help="JSON dict of hyperparameters overriding the zoo ones")
    args = parser.parse_args(argv)
    hyper = dict(ZOO.get(args.env, {}).get(args.alg, {})) if args.zoo else {}
    hyper.update(json.loads(args.hyper))
    run(args.env, args.alg, args.seed, args.steps, args.data_root, hyper)


if __name__ == "__main__":
    main()
