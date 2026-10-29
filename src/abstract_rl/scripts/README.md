# Scripts

Benchmark runs with Hydra (`src/abstract_rl/conf/` selects the algorithm and environment groups, and `hydra_conf.py` turns the composed algorithm node into the training dictionary):

```
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seeds=[1,2,3,4,5]   # seeds vmapped in one program
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=2 alg.num_epochs=100 alg.policy_sigma=0.5
uv run --extra gym python -m abstract_rl.scripts.run env=humanoid alg=mpo seed=1 \
    alg.mpo_outer_lr=0.0001 alg.rnk_eps=0.05 'alg.mpo_lm_bound=[-18.0,18.0]'
# same run on an NVIDIA GPU (unset JAX_PLATFORMS if it is already cpu; that variable wins over jax_platform)
uv sync --extra cuda --extra gym
uv run python -m abstract_rl.scripts.run env=humanoid alg=mpo jax_platform=cuda seed=1 \
    alg.mpo_outer_lr=0.0001 alg.rnk_eps=0.05 'alg.mpo_lm_bound=[-18.0,18.0]'
uv run python -m abstract_rl.scripts.trpo_pendulum    # thin wrappers for Pendulum-v1
uv run python -m abstract_rl.scripts.mpo_pendulum
```

All environments are JAX ports (gymnax, MJX, Brax) and ship with the core dependencies.
Every run writes to `data/<env>/<alg>/<timestamp>/` (configuration, logged returns, model checkpoints, a
performance plot and per-epoch CLI logs). `model/last` holds the final policy, `model/best` the policy of the epoch with
the highest mean training return (epoch and return in `model/best.json`); pass `--model best` to the evaluation scripts
to use it.

Evaluation (prefix with `JAX_PLATFORMS=cpu` when the `mps` extra is installed; the MJX environments need the CPU):

```
uv run python -m abstract_rl.scripts.eval_returns data/<env>/<alg>/<run> --save   # greedy return vs. success threshold
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/*/<run> --out eval --gif   # Pendulum balance check
uv run python -m abstract_rl.scripts.plot_learning_curve data/Pendulum-v1/trpo/<run> -o trpo.png
```

`eval_balance --gif` renders the hardest of the evaluation episodes. `env/render.py` renders any recorded episode
offscreen: Pendulum and MountainCar with PIL, the MuJoCo environments with `mujoco.Renderer` (on a headless Linux
machine set `MUJOCO_GL=egl` or `osmesa`). `run.run_policy <run> --render` writes one GIF per replayed episode.

The README benchmark (every environment x algorithm cell with seeds 1-5, greedy evaluation of every run, wall time per
cell in `<root>/timing/`):

```
uv run python -m abstract_rl.scripts.run_benchmark --envs pendulum inverted_pendulum --algs trpo mpo \
    --seeds 1 2 3 4 5 --root data/bench --threads 1
uv run python -m abstract_rl.scripts.run_benchmark --envs mountaincar --algs trpo --seeds 1 2 3 4 5 \
    --root data/bench --threads 1 --set max_env_steps=400000      # cap at the budget as main did
uv run python -m abstract_rl.scripts.run_benchmark --envs inverted_pendulum --algs mpo --seeds 1 \
    --root data/bench --threads 1                                  # one seed at a time: 5 vmapped seeds need ~45 GB
uv run python -m abstract_rl.scripts.run_benchmark --report --root data/bench --out docs/   # table, plot, timing.json
```

Different cells can run in parallel on one `--root`; invocations of the same cell must run one after the other.

Baselines (`uv sync --extra baselines`, Stable-Baselines3 TD3 and sb3-contrib TRPO on Gymnasium) and the comparison of
everything. `compare` searches its directory recursively (symbolic links are followed), so a directory with links to
the JAX benchmark runs and the SB3 runs gives the README table and `docs/benchmark.png`:

```
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 td3 --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/
```
