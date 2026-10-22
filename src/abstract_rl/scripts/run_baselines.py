"""Train a Stable-Baselines3 baseline (TD3 or sb3-contrib TRPO) on a benchmark environment.

Needs the optional extra ``baselines``; MuJoCo ids also need ``mujoco``. The evaluation protocol equals the one of
the abstract_rl runs: training episode returns are recorded per environment step (Monitor), and after training the
greedy policy is evaluated on 20 episodes with the start states seeded 10000 + episode, reporting the raw
environment return.
The step budget is ``steps`` in ``conf/env`` unless ``--steps`` is set.

TD3 starts from the rl-baselines3-zoo entry in ``ZOO``. TRPO uses library defaults unless ``--zoo``.
Repeatable ``--hp key=value`` overrides (Python literals) are applied on top of that entry.

Usage::

    uv run --extra baselines python -m abstract_rl.scripts.run_baselines Pendulum-v1 td3 --seed 1 --data-root results
    uv run --extra baselines python -m abstract_rl.scripts.run_baselines Pendulum-v1 td3 --hp learning_rate=1e-4
"""

import argparse
import ast
import copy
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from abstract_rl.scripts.hydra_conf import steps as steps_budget

ALGORITHMS_SB3 = ["td3", "trpo"]


def _td3_mujoco(**overrides: Any) -> dict[str, Any]:
    """Shared MuJoCo TD3 hyperparameters from rl-baselines3-zoo ``hyperparams/td3.yml``.

    Args:
        overrides: Keys replacing the shared entry (Swimmer sets ``gamma``).

    Returns:
        A fresh hyperparameter dict. ``noise_type`` and ``noise_std`` are turned into an action noise later.
    """
    hyper: dict[str, Any] = {
        "learning_starts": 10_000,
        "noise_type": "normal",
        "noise_std": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "policy_kwargs": {"net_arch": [400, 300]},
    }
    hyper.update(overrides)
    return hyper


# rl-baselines3-zoo hyperparameters (hyperparams/{td3,trpo}.yml). The zoo ``n_timesteps`` is not stored: the run
# uses ``steps`` from ``conf/env`` (or ``--steps``). ``n_envs`` and ``normalize`` are handled by ``run``.
# TD3 noise is stored as ``noise_type`` / ``noise_std`` and mapped onto an SB3 action noise sized to the action dim.
ZOO: dict[str, dict[str, dict[str, Any]]] = {
    "Pendulum-v1": {
        "td3": {
            "gamma": 0.98,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "noise_type": "normal",
            "noise_std": 0.1,
            "learning_rate": 1e-3,
            "policy_kwargs": {"net_arch": [400, 300]},
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
        "td3": {
            "noise_type": "ornstein-uhlenbeck",
            "noise_std": 0.5,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [400, 300]},
        },
        "trpo": {"normalize": True, "n_envs": 2, "use_sde": True, "sde_sample_freq": 4},
    },
    "HalfCheetah-v5": {"td3": _td3_mujoco()},
    "Hopper-v5": {"td3": _td3_mujoco()},
    "Walker2d-v5": {"td3": _td3_mujoco()},
    "InvertedPendulum-v5": {"td3": _td3_mujoco()},
    "InvertedDoublePendulum-v5": {"td3": _td3_mujoco()},
    "Reacher-v5": {"td3": _td3_mujoco()},
    "Pusher-v5": {"td3": _td3_mujoco()},
    "Swimmer-v5": {"td3": _td3_mujoco(gamma=0.9999)},
}


def environment_ids() -> list[str]:
    """Gymnasium ids accepted on the CLI: budgets from ``conf/env``, then zoo-only ids."""
    ids = list(steps_budget())
    ids.extend(env_id for env_id in ZOO if env_id not in ids)
    return ids


def _literal(raw: str) -> Any:
    """Parse one ``--hp`` value as a Python literal, otherwise keep the raw string.

    Args:
        raw: Text after ``=``.

    Returns:
        The literal (``10``, ``1e-3``, ``[400, 300]``, ``{'net_arch': [64, 64]}``) or the original string
        (``normal``).
    """
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def parse_hp(items: list[str]) -> dict[str, Any]:
    """Parse repeated ``key=value`` overrides.

    Args:
        items: ``--hp`` arguments. Later keys replace earlier ones.

    Returns:
        Hyperparameter overrides.
    """
    hyper: dict[str, Any] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        if separator != "=" or key == "":
            raise argparse.ArgumentTypeError(f"--hp expects key=value, got {item!r}")
        hyper[key] = _literal(raw)
    return hyper


def resolve_hyper(env_id: str, alg: str, zoo: bool, hp: list[str], hyper_json: str) -> dict[str, Any]:
    """Zoo hyperparameters with ``--hyper`` and ``--hp`` laid on top.

    TD3 always starts from its ``ZOO`` entry. TRPO starts from it only when ``zoo`` is set, otherwise the
    Stable-Baselines3 defaults are kept. ``--hp`` wins over the zoo entry and over ``--hyper``.

    Args:
        env_id: Gymnasium id.
        alg: Baseline name, ``td3`` or ``trpo``.
        zoo: Whether ``--zoo`` was passed.
        hp: Repeated ``key=value`` overrides.
        hyper_json: JSON object of overrides applied before ``--hp``.

    Returns:
        Hyperparameters still in zoo form (``noise_type`` is not yet an action noise object).
    """
    use_zoo = zoo or alg == "td3"
    source = ZOO.get(env_id, {}).get(alg, {}) if use_zoo else {}
    hyper: dict[str, Any] = copy.deepcopy(source)
    loaded = json.loads(hyper_json)
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError("--hyper must be a JSON object")
    hyper.update(loaded)
    hyper.update(parse_hp(hp))
    return hyper


def _noise_sigma(noise_std: Any, action_dim: int) -> np.ndarray:
    """Broadcast ``noise_std`` to one standard deviation per action dimension.

    Args:
        noise_std: Scalar or a vector of length ``action_dim``.
        action_dim: Continuous action dimension.

    Returns:
        Standard deviation vector.
    """
    values = np.asarray(noise_std, dtype=np.float64)
    match values.shape:
        case ():
            return np.full(action_dim, float(values))
        case _ if values.size == action_dim:
            return values.reshape(action_dim)
        case _:
            raise ValueError(f"noise_std length {values.size} does not match action dim {action_dim}")


def make_action_noise(hyper: dict[str, Any], action_dim: int) -> Any:
    """Map zoo ``noise_type`` and ``noise_std`` onto an SB3 action noise.

    The two keys are removed from ``hyper`` so they are not forwarded to the algorithm. ``noise_type`` is
    ``normal`` or ``ornstein-uhlenbeck``.

    Args:
        hyper: Algorithm kwargs, edited in place.
        action_dim: Continuous action dimension.

    Returns:
        ``NormalActionNoise`` or ``OrnsteinUhlenbeckActionNoise``, or ``None`` when no noise is configured.
    """
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    noise_type = hyper.pop("noise_type", None)
    noise_std = hyper.pop("noise_std", None)
    match (noise_type, noise_std is None):
        case (None, True):
            return None
        case (None, False):
            raise ValueError("noise_std requires noise_type")
        case (_, True):
            raise ValueError("noise_type requires noise_std")
        case ("normal", False):
            return NormalActionNoise(mean=np.zeros(action_dim), sigma=_noise_sigma(noise_std, action_dim))
        case ("ornstein-uhlenbeck", False):
            return OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=_noise_sigma(noise_std, action_dim))
        case _:
            raise ValueError(f"unknown noise_type {noise_type!r}, expected 'normal' or 'ornstein-uhlenbeck'")


def _budget(env_id: str, steps: int | None) -> int:
    """Step count from the argument, or from ``steps`` in ``conf/env``.

    Args:
        env_id: Gymnasium id.
        steps: Explicit budget, or ``None`` to read the config.

    Returns:
        Environment steps to train.
    """
    if steps is not None:
        return steps
    budgets = steps_budget()
    if env_id not in budgets:
        raise KeyError(f"{env_id} has no step budget in conf/env; pass steps")
    return budgets[env_id]


def evaluate(model: Any, env_id: str, episodes: int = 20, seed: int = 10_000, norm: Any = None) -> np.ndarray:
    """Returns of ``episodes`` greedy episodes on the unmodified environment."""
    import gymnasium as gym  # optional (`baselines` extra): the library itself does not depend on Gymnasium

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
    import gymnasium as gym
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    steps = _budget(env_id, steps)
    run_dir = os.path.join(data_root, env_id, f"sb3_{alg}", datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    os.makedirs(run_dir)
    torch.set_num_threads(1)

    kw = copy.deepcopy(hyper or {})
    n_envs, normalize = kw.pop("n_envs", 1), kw.pop("normalize", False)
    monitors = [os.path.join(run_dir, "monitor.csv" if i == 0 else f"env{i}.monitor.csv") for i in range(n_envs)]

    def make(f: str) -> Any:
        return lambda: Monitor(gym.make(env_id), f)

    env: Any = DummyVecEnv([make(f) for f in monitors])
    if normalize:
        env = VecNormalize(env)
    match alg:
        case "td3":
            from stable_baselines3 import TD3

            noise = make_action_noise(kw, int(env.action_space.shape[0]))
            model: Any = TD3("MlpPolicy", env, seed=seed, device="cpu", action_noise=noise, **kw)
        case "trpo":
            from sb3_contrib import TRPO

            model = TRPO("MlpPolicy", env, seed=seed, device="cpu", **kw)
        case _:
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
    parser.add_argument("env", choices=environment_ids())
    parser.add_argument("alg", choices=ALGORITHMS_SB3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="environment steps (default: the budget of the env)")
    parser.add_argument("--data-root", default="data", help="output directory (default: data)")
    parser.add_argument(
        "--zoo",
        action="store_true",
        help="use the rl-baselines3-zoo hyperparameters (always on for TD3)",
    )
    parser.add_argument("--hyper", default="{}", help="JSON object of hyperparameters overriding the zoo entry")
    parser.add_argument(
        "--hp",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="hyperparameter override applied on top of the zoo entry, as a Python literal; repeatable",
    )
    args = parser.parse_args(argv)
    if args.steps is None and args.env not in steps_budget():
        parser.error(f"{args.env} has no step budget in conf/env; pass --steps")
    try:
        hyper = resolve_hyper(args.env, args.alg, args.zoo, list(args.hp or []), args.hyper)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    run(args.env, args.alg, args.seed, args.steps, args.data_root, hyper)


if __name__ == "__main__":
    main()
