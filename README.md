# Abstract RL

[![CI](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Modular [JAX](https://docs.jax.dev) ([Equinox](https://docs.kidger.site/equinox/), [Optax](https://optax.readthedocs.io))
implementations of **TRPO** and **MPO** for continuous control, built from small composable pieces (environment
wrapper, trajectory operators, policies, value functions). Python 3.12+, [uv](https://docs.astral.sh/uv/), ruff, mypy.
The environments are jittable JAX ports ([gymnax](https://github.com/RobertTLange/gymnax) for the classic-control
tasks, [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) on Gymnasium's own XML for the MuJoCo pendulums and
[Brax](https://github.com/google/brax) for HalfCheetah), so
a whole epoch, rollouts included, is one jitted program and many seeds train at once under `vmap`. The library depends
on neither PyTorch nor Gymnasium; both are only used by the optional Stable-Baselines3 baselines.

| MPO on Pendulum-v1 (swing-up from hanging down) | MPO on MountainCarContinuous-v0 | MPO on InvertedPendulum-v5 |
|---|---|---|
| ![MPO Pendulum](docs/mpo_pendulum_best.gif) | ![MPO MountainCar](docs/mpo_mountaincar_best.gif) | ![MPO InvertedPendulum](docs/mpo_inverted_pendulum_best.gif) |
| **MPO on InvertedDoublePendulum-v5** | **MPO on HalfCheetah-v5** | |
| ![MPO InvertedDoublePendulum](docs/mpo_inverted_double_pendulum_best.gif) | ![MPO HalfCheetah](docs/mpo_halfcheetah_best.gif) | |

The GIFs show one greedy episode of the best JAX MPO run per environment (highest greedy return of the 5 benchmark
seeds), rendered offscreen (`env/render.py`); Pendulum shows the hardest of the 20 balance-check start states.

- **TRPO**: Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) with
  [GAE](https://arxiv.org/abs/1506.02438), conjugate gradient natural gradient step and a KL line search.
- **MPO**: Maximum A Posteriori Policy Optimization ([Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920)) with
  a non-parametric E-step, a KL constrained M-step and a [Retrace](https://arxiv.org/abs/1606.02647) critic.
- Univariate and diagonal (multi-dimensional) Gaussian policies, optionally tanh-squashed into the action bounds.

## Install

```bash
git clone https://github.com/kosmitive/abstract_rl.git && cd abstract_rl
uv sync                     # CPU
uv sync --extra mps         # plus the Metal backend for JAX on Apple silicon
uv sync --extra baselines   # plus Stable-Baselines3, SB3-Contrib and Gymnasium for the baseline runner
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
```

## JAX: many seeds at once

TRPO and MPO train all seeds in one jitted, vmapped program:

```bash
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seeds=[1,2,3,4]
uv run python -m abstract_rl.scripts.run env=pendulum alg=mpo seeds=[1,2,3,4]
uv run python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6   # throughput
uv run python -m abstract_rl.scripts.bench_jax --alg mpo --env pendulum --seeds 1 4 --epochs 4
```

The `mps` extra (`uv sync --extra mps`) installs the Metal backend for JAX, selected with `jax_platform=mps`. Metal
runs the TRPO networks on the gymnax environments; the MJX environments and MPO (whose seed-batched replay-memory
update uses a scatter Metal does not support) need `jax_platform=cpu` (the default).

Measured on an Apple M3 Max (16 cores), JAX on CPU, 6 epochs (3 for HalfCheetah), TRPO with `policy_batch_size=4000`.
JAX env-steps/s is summed over seeds and excludes the compiling first call.

| env | seeds | JAX env-steps/s (cpu) | torch env-steps/s (1 seed, steady state) | speedup | first call (compile) |
|---|---|---|---|---|---|
| Pendulum | 1 | 55156 | 5333 | 10x | 4.0 s |
| Pendulum | 4 | 65374 | 5333 | 12x | 3.0 s |
| Pendulum | 16 | 71182 | 5333 | 13x | 4.1 s |
| HalfCheetah | 1 | 2043 | n/a | n/a | 27.7 s |
| HalfCheetah | 4 | 2757 | n/a | n/a | 18.2 s |

The torch figure is the last torch commit (ab552fb, measured as 4f7828e before the rebase onto main) running Pendulum
TRPO (default config, one seed). Its steady state is `6 * 4000 / (T12 - T6)`, where T6 = 6.93 s and T12 = 11.43 s are
the wall times of a 6 and a 12 epoch run, so interpreter startup and imports cancel out (a single 6 epoch run, startup
included, gives only 1879 env-steps/s).
Torch HalfCheetah is not measured: the torch MuJoCo env is Gymnasium v5, with different dynamics and rewards. The
speedup is against the single-seed torch figure.

MPO on Pendulum (JAX on CPU, `env_sample=1000` steps per epoch, 4 epochs, same machine; the line is steps summed over seeds):

| seeds | JAX env-steps/s (cpu) | epoch |
|---|---|---|
| 1 | 806 | 1.24 s |
| 4 | 1042 | 3.84 s |

MPO is compute-bound (the E-step annotates the whole replay memory every epoch), so four vmapped seeds give only
about 1.3x the throughput of one; the benefit of the batched run is one process and one compile for all seeds. The full benchmark under [Results](#results) compares the learning with
the torch version (Pendulum MPO reaches -150 after 44k steps, median over 5 seeds; torch 36k).

### Deviations from the torch implementation

The full list with reasons is in the [design spec](docs/superpowers/specs/2026-10-22-jax-port-design.md).

- **D1** Random streams differ from torch and numpy; every draw uses an explicit key.
- **D2** An epoch overshoots `batch_size` by at most the episode still running in each of the E lanes (E = 1 is
  identical to torch).
- **D3** `run.conf` stores the real seed instead of `np.random.randint(1e6)`.
- **D4** The temperature dual runs in float32 instead of float64.
- **D5** Environment dynamics come from JAX ports instead of Gymnasium v5: gymnax (Pendulum, MountainCar), MJX on
  Gymnasium's XML (the pendulums, matching Gymnasium v5 up to float32 rounding) and Brax (HalfCheetah, small residuals).
- **D6** Observations are float32 from the environment on.
- **D7** When capping drops part of one epoch's episodes, ties between lanes finishing at the same step are broken by
  lane order.
- **D8** With `max_env_steps`, the seeds of one run stop together when the seed with the most steps reaches the cap,
  so the other seeds get up to about 5 % fewer steps than a torch run of that seed.
- **D9** The network-initialisation keys coincide with two of epoch 0's per-step keys (rollout and minibatch/resample
  key). Every draw is still seeded and independent across seeds; the effect is a correlation within one epoch.

## Results

The abstract_rl rows are the **JAX implementation** on the JAX environment ports (gymnax Pendulum-v1 and
MountainCarContinuous-v0, MJX InvertedPendulum-v5 and InvertedDoublePendulum-v5 built from Gymnasium's XML, Brax
HalfCheetah-v5; see the environment notes below). The SB3 rows are unchanged from the torch benchmark on `main` and
were measured on **Gymnasium v5** (labelled "SB3, Gymnasium").

Configurations are Hydra YAML files in [`src/abstract_rl/conf/`](src/abstract_rl/conf)
(`run env=<env> alg=<alg> seed=N`), **5 seeds (1-5) per cell**, and every algorithm gets the same environment-step
budget per environment (`steps` in `conf/env/<env>.yaml`).
*Greedy return* is the mean of 20 seeded evaluation episodes with the policy mean (final policy), averaged over seeds
± std over seeds. *Solved* counts seeds whose greedy return reaches the threshold (Gymnasium reward threshold;
Pendulum-v1 has none, -150 corresponds to swing-up and balance). *Steps to threshold* is the first step at which the
smoothed training return reaches it (median over the seeds that do). The thresholds are Gymnasium's; none was
re-derived for the JAX environments, which reproduce Gymnasium's dynamics and rewards (notes below).

| Environment | Algorithm | Seeds | Steps | Greedy return (mean ± std) | Solved | Steps to threshold |
|---|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO (abstract_rl) | 5 | 800,000 | -111.3 ± 1.3 | 5/5 | 172,000 (5/5) |
| Pendulum-v1 (≥ -150) | MPO (abstract_rl) | 5 | 800,000 | -106.2 ± 1.4 | 5/5 | 44,000 (5/5) |
| Pendulum-v1 (≥ -150) | TRPO (SB3, Gymnasium) | 5 | 800,000 | -124.0 ± 1.6 | 5/5 | 204,000 (5/5) |
| Pendulum-v1 (≥ -150) | TD3 (SB3, Gymnasium) | 5 | 800,000 | -120.9 ± 1.0 | 5/5 | 36,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (abstract_rl) | 5 | 390,911 | 94.9 ± 0.5 | 5/5 | 128,032 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | MPO (abstract_rl) | 5 | 389,000 | 93.7 ± 0.1 | 5/5 | 70,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (SB3, Gymnasium) | 5 | 400,072 | 74.7 ± 37.5 | 4/5 | 52,013 (4/5) |
| MountainCarContinuous-v0 (≥ 90) | TD3 (SB3, Gymnasium) | 5 | 399,966 | 54.9 ± 77.4 | 4/5 | 28,993 (4/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (abstract_rl) | 5 | 150,520 | 1000.0 ± 0.0 | 5/5 | 27,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | MPO (abstract_rl) | 5 | 150,417 | 1000.0 ± 0.0 | 5/5 | 19,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (SB3, Gymnasium) | 5 | 150,523 | 1000.0 ± 0.0 | 5/5 | 29,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TD3 (SB3, Gymnasium) | 5 | 149,310 | 1000.0 ± 0.0 | 5/5 | 25,000 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (abstract_rl) | 5 | 604,394 | 7574.9 ± 3303.2 | 3/5 | 457,454 (3/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | MPO (abstract_rl) | 5 | 600,288 | 8771.4 ± 731.9 | 2/5 | 60,060 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (SB3, Gymnasium) | 5 | 599,517 | 8270.3 ± 602.2 | 0/5 | 107,946 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TD3 (SB3, Gymnasium) | 5 | 599,243 | 9347.9 ± 10.3 | 5/5 | 26,996 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (abstract_rl) | 5 | 1,200,000 | 4433.2 ± 485.5 | 1/5 | 1,080,000 (1/5) |
| HalfCheetah-v5 (≥ 4800) | MPO (abstract_rl) | 5 | 1,200,000 | 8761.8 ± 1083.8 | 5/5 | 192,000 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (SB3, Gymnasium) | 5 | 1,200,000 | 2547.9 ± 1314.1 | 0/5 | never |
| HalfCheetah-v5 (≥ 4800) | TD3 (SB3, Gymnasium) | 5 | 1,200,000 | 11009.6 ± 522.5 | 5/5 | 96,000 (5/5) |

Learning curves on environment steps (training return, mean ± std over the 5 seeds; the runs of the table):

![benchmark](docs/benchmark.png)

Compared with the torch implementation on `main` (greedy return, solved seeds):

| Environment | TRPO torch | TRPO JAX | MPO torch | MPO JAX |
|---|---|---|---|---|
| Pendulum-v1 | -127.7 ± 2.7, 5/5 | -111.3 ± 1.3, 5/5 | -123.3 ± 0.4, 5/5 | -106.2 ± 1.4, 5/5 |
| MountainCarContinuous-v0 | 94.1 ± 0.7, 5/5 | 94.9 ± 0.5, 5/5 | 93.6 ± 0.0, 5/5 | 93.7 ± 0.1, 5/5 |
| InvertedPendulum-v5 | 1000.0 ± 0.0, 5/5 | 1000.0 ± 0.0, 5/5 | 1000.0 ± 0.0, 5/5 | 1000.0 ± 0.0, 5/5 |
| InvertedDoublePendulum-v5 | 9332.8 ± 3.8, 5/5 | 7574.9 ± 3303.2, **3/5** | 9332.9 ± 7.4, 5/5 | 8771.4 ± 731.9, **2/5** |
| HalfCheetah-v5 | 4471.7 ± 1645.7, 3/5 | 4433.2 ± 485.5, **1/5** | 7940.5 ± 1398.2, 5/5 | 8761.8 ± 1083.8, 5/5 |

- **Where JAX matches or beats torch.** Pendulum, MountainCar, InvertedPendulum and HalfCheetah MPO solve 5/5 seeds as
  on `main`, with equal or higher greedy returns. The Pendulum returns are about 16-17 points higher than torch's.
  gymnax's Pendulum matches Gymnasium step by step (observations within 2e-6, rewards within 1e-5 from the same
  state), but the 20 evaluation start states are drawn from JAX keys instead of Gymnasium seeds (D1). With a spread of
  about 85 over start states, the mean of 20 episodes moves by tens of points with the start states alone.
- **Where JAX is below torch.** These cells are reported as measured; no threshold was changed.
  - *HalfCheetah TRPO: 1/5 instead of 3/5, with the same mean (4433 vs 4472).* The JAX seeds are less spread
    (greedy 4074, 4242, 4286, 4169 and 5394, against torch's mix of solved and failed seeds). The four unsolved
    seeds end 510-730 below the threshold, and three of them were still rising over the last 10 % of training.
    Two single-setting trials did not raise the solved count: KL trust region 0.03 (mean 4525, 1/5) and 0.05 (mean
    4252, 1/5). The tuned setting of `main` (0.02) is kept.
  - *InvertedDoublePendulum TRPO: 3/5 instead of 5/5.* Seed 1 (greedy 8894) fails because one of its 20 episodes
    falls. Seed 5 (977) has not learned to balance within the budget; its training return was still rising
    (mean 725 over epochs 62-68, 1036 over epochs 69-75). A trial with a KL trust region of 0.01 instead of 0.005
    also gave 3/5 (mean 9157, every failure one falling episode out of 20), so `main`'s 0.005 is kept.
  - *InvertedDoublePendulum MPO: 2/5 instead of 5/5* (greedy 9334, 9357, 8897, 8907 and 7362). Every seed reaches
    the threshold in training (median 60k steps, `main` 66k). The final policies of seeds 3 and 4 drop one of their
    20 greedy episodes. Seed 5 drops 6 of 20, and its training return falls at the end (mean 5386 over epochs
    481-540, 4308 over epochs 541-600): the kind of late collapse that `main` removed in torch by halving the KL
    trust region to 0.005, which these runs already use. No further tuning was run.
  - *The InvertedDoublePendulum falls come from the policies, not from the JAX environment.* Run in Gymnasium's
    float64 MuJoCo from the same start states, the policies fall on the same episodes at the same steps, and a
    float64 JAX env does not remove the falls. 8 of the 10 runs fall on about 1-2 % of start states, mostly shortly
    after the reset. With 20 evaluation episodes, a single fall is enough to drop a seed below 9100. MPO seed 5 is
    a marginal balancer (about half of its episodes fall), and TRPO seed 5 never learned to balance within the
    budget.
  - *Steps to threshold* is slower than torch for MountainCar TRPO (128k vs 44k; all seeds still solve), MountainCar
    MPO (70k vs 56k), Pendulum MPO (44k vs 36k), InvertedDoublePendulum TRPO (457k vs 431k, over 3 instead of 5
    seeds) and HalfCheetah TRPO (1.08M for 1 seed vs 732k for 2). It is equal or faster for the other cells.
- **Environments (spec D5).** gymnax provides Pendulum-v1 and MountainCarContinuous-v0. InvertedPendulum-v5 and
  InvertedDoublePendulum-v5 run Gymnasium's own XML on MJX with RK4, Gymnasium's frame skip (2 and 5), reward,
  termination, observations (4 and 9 dims), reset noise and `ctrl = action`. Their trajectories match Gymnasium v5 to
  float32 precision (`tests/test_mujoco_fidelity.py`). HalfCheetah-v5 is Brax's model on MJX. It has the same dt,
  gear, reward weights, observation and reset noise. The solver runs 4 instead of 100 iterations, only the head,
  thighs and feet collide with the floor, and the forward-velocity reward lags one 0.01 s substep. The lag moves an
  episode's return by well under 0.1 % of the threshold, and open-loop returns match Gymnasium, so the threshold of
  4800 is kept.
- **Tuned settings (unchanged from `main`, see `conf/tune/`):**
  - Pendulum: TRPO discount 0.95; MPO E-step epsilon `rnk_eps` 0.05.
  - MountainCar: MPO uses a fixed (not learned) exploration std of 2.0. A learned std shrinks to ≈ 0 before the
    sparse goal is found.
  - MPO, one setting each: a 300k replay memory (`tc_size`) on InvertedPendulum, and a KL trust region of 0.005 on
    InvertedDoublePendulum.
  - TRPO: InvertedDoublePendulum uses 75 × 8000 steps and a KL trust region of 0.005. HalfCheetah uses 300 × 4000
    steps, a KL trust region of 0.02 and `obs_norm`. Both drop the value L2 penalty.
- **Budgets.** TRPO epochs are `policy_batch_size` steps, and the episodes still running in the rollout lanes are
  finished, so InvertedPendulum and InvertedDoublePendulum end slightly past the budget (D2). Both MountainCar cells
  run with `max_env_steps` 400000; TRPO gets it via `--set max_env_steps=400000`, because finishing the lanes would
  otherwise give about 500k steps. All seeds of a run stop together once the seed with the most steps reaches the cap (D8),
  so the other seeds end between 380k and 400k steps. MPO stops after a fixed number of epochs and finishes the running episode, so on
  InvertedDoublePendulum it runs past the budget: up to 22 % for the seed whose episodes fall mid-length. The learning
  curves are cut at the budget; the greedy return is the one of the final policy.
- **Pendulum balance check** (`scripts/eval_balance.py`, |θ| < 0.1 rad in the last 100 of 200 steps, 20 start
  states): TRPO and MPO balance in 5/5 seeds and 20/20 episodes each (max |θ| 0.001-0.034 rad TRPO, 0.021-0.089 rad
  MPO).
- **SB3 baselines (Gymnasium v5, from `main`):** TRPO uses the library defaults, except on MountainCarContinuous-v0,
  which uses the [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyper-parameters (`--zoo`, gSDE
  exploration) at the same 400k budget; with the defaults it scores 0 there. On Pendulum-v1 the zoo settings were
  worse than the defaults (-136), so the defaults are shown. TD3 always uses the zoo hyper-parameters (`ZOO` in
  `scripts/run_baselines.py`); on MountainCarContinuous-v0 one of five seeds (seed 3) never finds the goal (-99.9).
- **Runtime.** Runs were made on an Apple M3 Max (16 cores, 48 GB) with JAX 0.11 on the CPU, using
  `run_benchmark --threads 1`. This turns XLA's multi-threaded kernels off, but a process still uses 1-2 cores while
  compiling. The 5 seeds of a cell train vmapped in one process. The exception is InvertedPendulum MPO: with its 300k
  replay memory, 5 vmapped seeds need about 45 GB, so it ran one seed per process, one after the other. Up to five
  cells ran at the same time, so the wall times below include that contention. Seeds are fixed, but results are not
  bit-reproducible across machines or XLA versions.

  | Training wall time per cell (5 seeds) | TRPO | MPO |
  |---|---|---|
  | Pendulum-v1 | 1.8 min | 3.4 h |
  | MountainCarContinuous-v0 | 1.2 min | 1.6 h |
  | InvertedPendulum-v5 | 2.5 min | 1.5 h (5 × 17-20 min, one seed at a time) |
  | InvertedDoublePendulum-v5 | 18 min | 2.4 h |
  | HalfCheetah-v5 | 51 min | 5.6 h |

  Times are from `data/bench/timing.json` and exclude the greedy evaluation (2 s to 3.5 min per cell). Reproduce
  with `scripts/run_benchmark.py` (see [`src/abstract_rl/scripts/README.md`](src/abstract_rl/scripts/README.md)).

## Usage

```bash
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=1 alg.num_epochs=100  # Hydra overrides
uv run python -m abstract_rl.scripts.run env=mountaincar alg=mpo data_root=results -m seed=1,2,3,4,5  # sweep
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo --cfg job              # print the config
JAX_PLATFORMS=cpu uv run python -m abstract_rl.scripts.eval_returns data/HalfCheetah-v5/mpo/<run> --save  # greedy eval
JAX_PLATFORMS=cpu uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run>  # balance check
uv run python -m abstract_rl.scripts.run_benchmark --envs pendulum --algs trpo mpo --seeds 1 2 3 4 5 \
    --root data/bench --threads 1                                                       # benchmark cells
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 td3 --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/  # table and plots of all runs below results/
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render         # replay, GIF per episode
# MuJoCo GIF rendering (eval_balance --gif, --render) on headless Linux needs MUJOCO_GL=egl in the environment
```

The standalone evaluation scripts run on the default JAX backend; with the `mps` extra installed, prefix them with
`JAX_PLATFORMS=cpu` (the MJX environments do not run on Metal).

Configuration ([Hydra](https://hydra.cc)) ships inside the package in `src/abstract_rl/conf/`: `config.yaml` (`seed`,
`data_root`, `max_env_steps`), `alg/{trpo,mpo}.yaml` (Pendulum-v1 defaults), `env/<env>.yaml` (Gymnasium `id`,
`success` threshold, `steps` budget), `tune_all/<env>.yaml` (overrides for both algorithms) and
`tune/<env>_<alg>.yaml` (per-algorithm overrides). Environments: `pendulum`,
`mountaincar`, `inverted_pendulum`, `inverted_double_pendulum`, `halfcheetah`. `alg.<key>=...` overrides beat the
tune files. The training loops get a flat dictionary with prefixes (`env_*`, `policy_*`, `val_*`, `trpo_*`,
`mpo_*`). Runs are written to `data/<env>/<alg>/<timestamp>/` (configuration, returns, models, plots).
Environments with another action dimension work out of the box: `select_policy` chooses the univariate or the
diagonal Gaussian policy from the action space.

## Architecture

```
src/abstract_rl/
├── algorithms/
│   ├── continuous/policy_gradient/   TRPO and MPO (policy update rules)
│   └── mixed/                        conjugate gradient, trust region gradient (natural gradient + line search),
│                                     vanilla gradient (Adam), non-parametric temperature dual
├── data_structures/
│   ├── abstract_conf/                SharedConf (namespaced config), ModelConfiguration (registry, IO),
│   │                                 ModelState (the traced pytree of models, optimiser states and buffers)
│   └── temporal_difference_data/     Trajectory, TrajectoryCollection (replay buffer), Batch (equinox pytrees)
├── env/                              JAX env ports (jax_envs.py: gymnax, MJX, Brax), RolloutEnvWrapper (jitted
│                                     rollouts in parallel lanes), offscreen rendering (render.py)
├── operator/                         trajectory operators: TD, GAE, Retrace, action resampling, MPO reweighting, filters
├── policy/                           univariate and diagonal Gaussian policy (+ select_policy), non-parametric policy
├── value_functions/                  fully connected V and Q networks
├── neural_modules/                   MLP building block
├── run/                              training loops (run_trpo, run_mpo_non_parametric), vmapped seeds (run_seeds),
│                                     policy replay
├── scripts/                          Hydra entry point (run.py, hydra_conf.py), evaluate / benchmark / baselines /
│                                     compare
└── misc/ and plots/                  CLI printer, data logger, running filters, plots
```

A training loop is a sequence of operator applications on trajectory collections:

- **TRPO** per epoch: collect `policy_batch_size` steps → `TD` operator (returns, bootstrapped at time limits) →
  `GAE` operator → natural gradient step with KL trust region → V function fit (Adam).
- **MPO** per epoch: collect `env_sample` steps into the replay collection → `Resample` operator (actions from the
  target policy, evaluated by the target Q) → temperature dual + `VariationalRanking` operator (non-parametric
  E-step) → KL constrained weighted maximum likelihood (M-step) → `Retrace` operator → Q function fit (Adam).

An epoch, rollouts included, is one jitted function of a `ModelState` pytree (networks, optimiser states, replay
buffer, observation statistics), and `run_seeds` vmaps it over the seeds. Target networks are registered as
`Syncable`s in the `ModelConfiguration`; inside the jitted epoch they are synced with `state.sync()`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy src                                        # types
uv run pytest                                          # unit and short training smoke tests
```

## Changes compared to the 2019 version

The original code (2019, Python 3.6, PyTorch 1.0, gym 0.12) targeted robot control experiments. The port keeps the
module layout and operator design. In short:

- uv + `pyproject.toml` (Python 3.12-3.13, src layout, `abstract_rl.*` imports), ruff, mypy, pytest; `gym` →
  `gymnasium` (`terminated`/`truncated`); modern PyTorch APIs (until the JAX port).
- Removed (still in the history of `main`): the experimental SAC, the dueling Q network, the circular tail
  memory, the robot specific scripts, unused helpers.
- Added: `DiagonalGaussianPolicy` for multi-dimensional actions, tanh action squashing, Hydra run configs,
  return based evaluation, SB3 baselines.
- 2026 port to JAX (Equinox, Optax): every epoch is jitted and the seeds of a run are vmapped; the environments
  are jittable JAX ports instead of Gymnasium, and PyTorch and Gymnasium are only needed by the SB3 baselines. The
  differences to the torch version are listed above (D1-D9) and the benchmark was re-run with JAX (see Results).
- Bug fixes needed for both algorithms to learn (see the commit history):
  TRPO surrogate broadcast (`[D,1]*[D]`), target-policy sync without sigma, time-limit episodes treated as terminal,
  replay index sampling, Q-function target broadcasting and state/action pairing, Retrace off-by-one and aliasing,
  MPO per-state weight normalisation and Adam temperature dual, unbounded
  Gaussian actions (now tanh-squashed with the Jacobian in the log-likelihood), Retrace policy mismatch, KL check
  in the TRPO line search, a larger ELU critic for MPO, and a state-dependent sigma that stays positive without
  being passed through softplus twice.

## Acknowledgements

- The running state filter (`misc/running_centered_filter.py`) is adapted from John Schulman's
  [modular_rl](https://github.com/joschu/modular_rl) and Ilya Kostrikov's
  [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo).
- The baselines use [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
  [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) and hyper-parameters from
  [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo); the environments come from
  [Gymnasium](https://gymnasium.farama.org) and [MuJoCo](https://mujoco.org).

## Citing

If you use this code, please cite the repository (see [`CITATION.cff`](CITATION.cff)) and the papers of the methods:

```bibtex
@software{semmler_abstract_rl,
  author = {Semmler, Markus},
  title  = {Abstract RL: modular JAX TRPO and MPO},
  url    = {https://github.com/kosmitive/abstract_rl},
  year   = {2026}
}

@inproceedings{peters2010reps,
  author    = {Peters, Jan and M{\"u}lling, Katharina and Alt{\"u}n, Yasemin},
  title     = {Relative Entropy Policy Search},
  booktitle = {Proceedings of the 24th AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2010},
  pages     = {1607--1612},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/7727}
}

@inproceedings{schulman2015trpo,
  author    = {Schulman, John and Levine, Sergey and Moritz, Philipp and Jordan, Michael I. and Abbeel, Pieter},
  title     = {Trust Region Policy Optimization},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning (ICML)},
  year      = {2015},
  url       = {https://arxiv.org/abs/1502.05477}
}

@inproceedings{schulman2016gae,
  author    = {Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael I. and Abbeel, Pieter},
  title     = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2016},
  url       = {https://arxiv.org/abs/1506.02438}
}

@inproceedings{abdolmaleki2018mpo,
  author    = {Abdolmaleki, Abbas and Springenberg, Jost Tobias and Tassa, Yuval and Munos, Remi and Heess, Nicolas
               and Riedmiller, Martin},
  title     = {Maximum a Posteriori Policy Optimisation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1806.06920}
}

@article{abdolmaleki2018rerpi,
  author    = {Abdolmaleki, Abbas and Springenberg, Jost Tobias and Degrave, Jonas and Bohez, Steven and Tassa, Yuval
               and Belov, Dan and Heess, Nicolas and Riedmiller, Martin},
  title     = {Relative Entropy Regularized Policy Iteration},
  journal   = {arXiv preprint arXiv:1812.02256},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.02256}
}

@inproceedings{munos2016retrace,
  author    = {Munos, R{\'e}mi and Stepleton, Tom and Harutyunyan, Anna and Bellemare, Marc G.},
  title     = {Safe and Efficient Off-Policy Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2016},
  url       = {https://arxiv.org/abs/1606.02647}
}
```

## License

MIT, see [LICENSE](LICENSE).
