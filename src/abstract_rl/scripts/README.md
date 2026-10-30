# Scripts

Benchmark runs with Hydra (`src/abstract_rl/conf/` selects the algorithm and environment groups, and `hydra_conf.py` turns the composed algorithm node into the training dictionary):

```
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=2 alg.num_epochs=100 alg.policy_sigma=0.5
uv run python -m abstract_rl.scripts.trpo_pendulum    # thin wrappers for Pendulum-v1
uv run python -m abstract_rl.scripts.mpo_pendulum
```

MuJoCo environments need the `mujoco` extra (`uv sync --extra mujoco`).
Every run writes to `data/<env>/<alg>/<timestamp>/` (configuration, logged returns, model checkpoints, a
performance plot and per-epoch CLI logs). `model/last` holds the final policy, `model/best` the policy of the epoch with
the highest mean training return (epoch and return in `model/best.json`); pass `--model best` to the evaluation scripts
to use it.

MPO runs can be resumed. `alg.checkpoint_every=50` writes the full training state (networks, optimizers, replay memory,
random states) to `<run>/checkpoint` every 50 epochs, replacing the previous one. Starting the same command again with
`alg.resume=true` continues the newest run below `data_root` that has a checkpoint, bit identically to a run that was
never stopped. A run directory can also be given, `alg.resume=data/<env>/mpo/<run>`.

Evaluation:

```
uv run python -m abstract_rl.scripts.eval_returns data/<env>/<alg>/<run> --save   # greedy return vs. success threshold
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/*/<run> --out eval --gif   # Pendulum balance check
uv run python -m abstract_rl.scripts.plot_learning_curve data/Pendulum-v1/trpo/<run> -o trpo.png
```

Baselines (`uv sync --extra baselines`, Stable-Baselines3 TD3 and sb3-contrib TRPO) and the comparison of everything:

```
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 td3 --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/
```
