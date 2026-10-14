# Scripts

Python 3.13. From a checkout, `uv sync` installs the library (`uv sync --extra mujoco` for the MuJoCo environments, `uv sync --extra baselines` for Stable-Baselines3). After installation the same entry points are console scripts: `abstract-rl`, `abstract-rl-eval`, `abstract-rl-baselines`, `abstract-rl-compare`.

Benchmark runs use Hydra. `config/` at the repository root holds the tuned TRPO / MPO configuration of every environment, and `hydra_conf.py` composes it (the wheel ships a copy of that directory):

```
abstract-rl env=pendulum alg=trpo seed=1
abstract-rl env=halfcheetah alg=mpo seed=2 alg.num_epochs=100 alg.policy_sigma=0.5
abstract-rl env=pendulum alg=trpo --cfg job
abstract-rl -m env=mountaincar alg=mpo data_root=results seed=1,2,3,4,5
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
```

`alg.<key>=value` overrides the environment table. HalfCheetah, Humanoid and the other MuJoCo ids need the `mujoco` extra. Humanoid-v5 (`env=humanoid`) is a 3,000,000-step budget with observation 348 and action 17. `trpo_pendulum` and `mpo_pendulum` are thin wrappers for the full Pendulum-v1 budgets and take no Hydra overrides:

```
uv run python -m abstract_rl.scripts.trpo_pendulum
uv run python -m abstract_rl.scripts.mpo_pendulum
```

Every run writes to `data/<env>/<alg>/<timestamp>/` (configuration, logged returns, model checkpoints, a performance plot and per-epoch CLI logs).

Evaluation:

```
abstract-rl-eval data/<env>/<alg>/<run> --save
uv run python -m abstract_rl.scripts.eval_returns data/<env>/<alg>/<run> --save
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run> --out eval --gif
uv run python -m abstract_rl.scripts.plot_learning_curve data/Pendulum-v1/trpo/<run> -o trpo.png
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run>
```

`abstract-rl-eval` exits with status 1 when a run's greedy return is below the environment threshold. `eval_balance --gif` writes an rgb-array GIF. `run_policy --render` requests a display window.

Baselines (the `baselines` extra: Stable-Baselines3 PPO and sb3-contrib TRPO) and the comparison of a results tree:

```
abstract-rl-baselines HalfCheetah-v5 ppo --seed 1 --data-root results
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 ppo --seed 1 --data-root results
abstract-rl-compare results --out docs/
```

`abstract-rl-compare` writes `benchmark.md` and `benchmark.png` into `--out`. Passing `--out docs/` replaces the learning-curve figure stored in this repository.

The critic micro-benchmark times the Adam scan without training:

```
uv run python -m abstract_rl.scripts.benchmark_critic --steps 1000 --batch 256
```
