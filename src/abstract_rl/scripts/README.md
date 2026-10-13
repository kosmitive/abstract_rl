# Scripts

Benchmark runs with Hydra (`config/` at the repository root holds the tuned TRPO / MPO configuration of every
environment, `hydra_conf.py` composes it):

```
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=2 alg.num_epochs=100 alg.policy_sigma=0.5
uv run python -m abstract_rl.scripts.trpo_pendulum    # thin wrappers for Pendulum-v1
uv run python -m abstract_rl.scripts.mpo_pendulum
```

MuJoCo environments need the `mujoco` extra (`uv sync --extra mujoco`).
Every run writes to `data/<env>/<alg>/<timestamp>/` (configuration, logged returns, model checkpoints, a
performance plot and per-epoch CLI logs).

Evaluation:

```
uv run python -m abstract_rl.scripts.eval_returns data/<env>/<alg>/<run> --save   # greedy return vs. success threshold
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/*/<run> --out eval --gif   # Pendulum balance check
uv run python -m abstract_rl.scripts.plot_learning_curve data/Pendulum-v1/trpo/<run> -o trpo.png
```

Baselines (`uv sync --extra baselines`, Stable-Baselines3 PPO and sb3-contrib TRPO) and the comparison of everything:

```
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 ppo --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/
```
