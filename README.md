# Abstract RL

[![CI](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Modular [MLX](https://github.com/ml-explore/mlx) implementations of **TRPO** and **MPO** for continuous control, built from small composable pieces
(environment wrapper, trajectory operators, policies, value functions). Python 3.13, [uv](https://docs.astral.sh/uv/),
ruff, mypy; runs on Gymnasium environments with one- and multi-dimensional actions. This branch is an experimental
MLX port of `main`, see [MLX port (experiment)](#mlx-port-experiment).

- **TRPO**: Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) with
  [GAE](https://arxiv.org/abs/1506.02438), conjugate gradient natural gradient step and a KL line search.
- **MPO**: Maximum A Posteriori Policy Optimization ([Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920)) with
  a non-parametric E-step, a KL constrained M-step and a [Retrace](https://arxiv.org/abs/1606.02647) critic.
- Univariate and diagonal (multi-dimensional) Gaussian policies, optionally tanh-squashed into the action bounds.

## Install

```bash
git clone https://github.com/kosmitive/abstract_rl.git && cd abstract_rl
uv sync                                   # add --extra mujoco for the MuJoCo environments (HalfCheetah-v5, ...)
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
```

MLX uses Metal on Apple silicon and its CPU backend on Linux (`mlx[cpu]`).

## Results

Results of this branch on a MacBook M3 Max. Configurations are Hydra YAML files in
[`src/abstract_rl/conf/`](src/abstract_rl/conf) (`run env=<env> alg=<alg> seed=N`), **5 seeds (1-5) per cell**, with
the environment-step budget of the environment (`steps` in `conf/env/<env>.yaml`). *Greedy return* is the mean of 20
seeded evaluation episodes with the policy mean (final policy), averaged over seeds ± std over seeds. *Solved* counts
seeds whose greedy return reaches the threshold (Gymnasium reward threshold; Pendulum-v1 has none, -150 corresponds to
swing-up and balance).

| Environment | Algorithm | MLX device | Seeds | Steps | Greedy return (mean ± std) | Solved |
|---|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO | CPU | 5 | 800,000 | -124.5 ± 1.2 | 5/5 |
| Pendulum-v1 (≥ -150) | MPO | GPU | 5 | 800,000 | -122.9 ± 2.8 | 5/5 |

MountainCarContinuous-v0, InvertedPendulum-v5, InvertedDoublePendulum-v5 and HalfCheetah-v5 have not been trained
with MLX yet. The per-seed returns and the throughput are in [MLX port (experiment)](#mlx-port-experiment).

Tuned settings (`conf/tune/`):

- Pendulum-v1: TRPO discount 0.95, MPO E-step epsilon `rnk_eps` 0.05.
- MountainCarContinuous-v0: MPO uses a fixed (not learned) exploration std of 2.0. A learned std shrinks to ≈ 0
  before the sparse goal is found.
- InvertedPendulum-v5: MPO uses a 300k instead of 50k replay memory (`tc_size`).
- InvertedDoublePendulum-v5: MPO uses a KL trust region (`mpo_trust_regions`) of 0.005 instead of 0.01; TRPO
  75 × 8000 steps with a KL trust region of 0.005, without the value L2 penalty.
- HalfCheetah-v5: MPO 1,200 epochs with `policy_sigma` 0.5 and a KL trust region of 0.02; TRPO 300 × 4000 steps with
  a KL trust region of 0.02 and `obs_norm`, without the value L2 penalty.
- The Pendulum and MountainCar runs stop at the exact step budget (`max_env_steps`). Elsewhere MPO stops after a
  fixed number of epochs and finishes the running episode.
- Seeds are fixed, but the results are not bit-reproducible across machines or MLX devices.

## Usage

```bash
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=1 alg.num_epochs=100  # Hydra overrides
uv run python -m abstract_rl.scripts.run env=mountaincar alg=mpo data_root=results -m seed=1,2,3,4,5  # sweep
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo --cfg job              # print the config
uv run python -m abstract_rl.scripts.eval_returns data/HalfCheetah-v5/mpo/<run> --save       # greedy evaluation
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run>              # Pendulum balance check
uv run python -m abstract_rl.scripts.compare results --out docs/                         # table and plots
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render         # replay a policy
```

Configuration ([Hydra](https://hydra.cc)) ships inside the package in `src/abstract_rl/conf/`: `config.yaml` (`seed`,
`data_root`, `max_env_steps`, `device`), `alg/{trpo,mpo}.yaml` (Pendulum-v1 defaults), `env/<env>.yaml` (Gymnasium `id`,
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
│   ├── abstract_conf/                SharedConf (namespaced config), ModelConfiguration (registry, target sync, IO)
│   └── temporal_difference_data/     Trajectory, TrajectoryBuilder, TrajectoryCollection (replay), Batch
├── env/                              RolloutEnvWrapper: runs whole episodes of a policy in a Gymnasium env
├── operator/                         trajectory operators: TD, GAE, Retrace, action resampling, MPO reweighting, filters
├── policy/                           univariate and diagonal Gaussian policy (+ select_policy), non-parametric policy
├── value_functions/                  fully connected V and Q networks
├── neural_modules/                   MLP building block
├── run/                              training loops (run_trpo, run_mpo_non_parametric), policy replay
├── scripts/                          Hydra entry point (run.py, hydra_conf.py), evaluate / baselines / compare
└── misc/ and plots/                  CLI printer, data logger, running filters, plots
```

A training loop is a sequence of operator applications on trajectory collections:

- **TRPO** per epoch: collect `policy_batch_size` steps → `TD` operator (returns, bootstrapped at time limits) →
  `GAE` operator → natural gradient step with KL trust region → V function fit (Adam).
- **MPO** per epoch: collect `env_sample` steps into the replay collection → `Resample` operator (actions from the
  target policy, evaluated by the target Q) → temperature dual + `VariationalRanking` operator (non-parametric
  E-step) → KL constrained weighted maximum likelihood (M-step) → `Retrace` operator → Q function fit (Adam).

Target networks are registered as `Syncable`s in the `ModelConfiguration` and synced with `mc.sync()`.

## MLX port (experiment)

This branch implements the existing modules with [MLX](https://github.com/ml-explore/mlx) (`mlx>=0.32.4`). There is no backend abstraction: the layout, the operators and the training loops are the ones of `main`
(eb41282). The Gymnasium environments run on the host. The networks and the optimization run on the MLX device set by
`device` in `conf/config.yaml`:

- `cpu` or `gpu` (Metal) select the device directly.
- `auto` is the default. It selects the GPU for MPO, if Metal is available, and the CPU for TRPO (see the
  throughput below).
- With MPO on the GPU, the rollout still evaluates the policy on the CPU stream (`env_rollout_on_cpu: true` in
  `conf/alg/mpo.yaml`). One small policy evaluation per environment step is faster on the CPU, and the parameters
  are shared through the unified memory, so nothing is copied.

### Deviations from `main`

- **Compiled steps.** The TRPO step (surrogate, Fisher-vector products via `jvp` of `grad`, conjugate gradient, line
  search) is one `mx.compile`d function. The conjugate gradient runs a fixed number of iterations and freezes the
  iterate once it has converged, instead of exiting early. The line search evaluates all step sizes and selects the
  first accepted one with `where`. The Adam steps of the V and Q functions, the MPO M-step and the temperature dual
  are compiled as well, with the parameters and the Adam moments as implicit state.
- **Precision.** Everything is float32, except the MPO temperature dual. The dual runs in float64 on the CPU, as
  before, and in float32 on the GPU, because Metal has no float64.
- **Random numbers.** MLX has its own generator, so a seed draws other actions than on `main`. Runs are seeded,
  but they are not identical to the ones of `main`.
- **Diagonal covariance only.** `cov_type: full` raises an error, because `mx.linalg.cholesky` has no gradient
  (vjp).
- **Minibatches.** The M-step and the V and Q fits gather their minibatches on the device from one flat float32
  copy of the replay memory per fit (`FlatReplay`). The indices are drawn with the same numpy calls as in
  `TrajectoryCollection.sample`, so the batches are identical. The V and Q fits and the M-step run ten Adam steps
  per compiled call; the log line of every tenth step is printed after the fit.
- **Batched annotation.** The resampling, the Q annotation and Retrace run over all trajectories at once, in calls of
  2,048 states (compiled) on the GPU and 512 states (not compiled, which is faster on the MLX CPU) on the CPU. The
  noise is still drawn per trajectory, and the Retrace bootstrap stops at trajectory seams. The annotations, the
  temperature dual and the ranking stay on the device. This changes the results by float rounding only.
- **Buffer cache.** `default_settings` caps the MLX buffer cache at 2 GiB (`misc/mixed.py`). Without a cap it grows
  to the memory limit.
- **ELU.** `mlx.nn.elu` has a NaN gradient for pre-activations above about 88. The port uses its own ELU
  (`misc/mixed.py`). Without it, the MPO critic diverges after about 25 epochs on Pendulum.
- **Storage.** Models are stored as safetensors.
- **Not used:** `vmap` over seeds and custom kernels. Seeds run as separate processes.

### Tests and Metal checks

- The parity tests compare the port with a fixture generated from `main` (eb41282, `tests/golden/`) on the same
  weights and batches. It covers the policy log-likelihood, mode and KL, the conjugate gradient, the TRPO step, the V and Q
  Adam steps, TD, GAE, Retrace, the temperature dual and the M-steps.
- `tests/test_mlx_capabilities.py` checks the MLX features the port relies on.
- `tests/test_metal.py` compares Metal with the CPU and is skipped without Metal:

| Metal vs CPU (M3 Max) | max abs diff | other |
|---|---|---|
| scatter-add gradient (duplicate indices) | 2.4e-7 | max \|g\| 3.8 |
| compiled TRPO step, parameters | 8.4e-6 | norm rel 5.5e-6, step norm rel 2.7e-4 |
| Q annotation 50k × 20 | 3.6e-7 | CPU 1.33 s, GPU 0.033 s |

### Throughput (MacBook M3 Max, Pendulum-v1, environment steps per second)

Notes on the setup:

- All runs use nice 10 and are run one configuration at a time. Other jobs were running on the machine, so the
  numbers vary by about 10–30% between repetitions.
- TRPO is measured on epochs 6–11 (4,000 steps per epoch). MPO uses 1,000 steps per epoch. Its cost grows with the
  replay memory, so MPO is reported at three memory sizes:
  - epochs 1–4: up to 5k transitions
  - epochs 15–24: about 20k transitions
  - epochs 50–59: the full 50k memory, which is how most of a run looks
- "5 seeds" is 5 processes, summed.
- The MLX numbers for MPO epochs 50–59 include the speedups described below the tables (re-timed on
  2026-10-24). The other MPO columns were measured before them.

| MLX device | TRPO, 1 seed | TRPO, 5 seeds | MPO epochs 1–4, 1 seed | MPO epochs 1–4, 5 seeds | MPO epochs 15–24 | MPO epochs 50–59 |
|---|---|---|---|---|---|---|
| CPU | **6,357** | **26,054** | 1,068 | 3,730 | 444 | 385 |
| GPU | 2,478 | 10,952 | **1,086** | **3,940** | **523** | **2,135** |

Seconds per epoch by phase, MPO at epochs 50–59 (full memory):

| MLX | rollout | resample + annotation | E-step | M-step | Retrace | Q fit | epoch |
|---|---|---|---|---|---|---|---|
| CPU, before the speedups | 0.09 | 0.83 | 0.75 | 0.24 | 0.55 | 1.97 | 4.45 |
| GPU, before the speedups | 0.35 | 0.14 | 0.10 | 0.24 | 0.13 | 2.00 | 3.00 |
| **CPU** | 0.08 | 0.79 | 0.62 | 0.05 | 0.46 | 0.57 | 2.60 |
| **GPU** | 0.08 | 0.04 | 0.07 | 0.04 | 0.03 | 0.19 | 0.47 |

- **TRPO.** The rollout evaluates the policy once per environment step. On Metal every such call costs about
  0.4 ms of dispatch and synchronization, which makes the rollout 3× slower than on the CPU (1.5 s vs 0.48 s per
  epoch). The compiled TRPO step itself is faster on the GPU (0.012 s vs 0.034 s), but it is a small part of an
  epoch.
- **MPO.** With the full memory, resampling, annotation, E-step and Retrace are 4–8× faster on the GPU than on the
  MLX CPU. Before the speedups below, the Q fit (1,000 small Adam steps per epoch) cost about the same on both
  devices, and the rollout was slower on the GPU for the same reason as in TRPO. The first GPU epoch includes
  compilation and the Metal warm-up.
- Peak memory, with the 2 GiB cache cap: the MLX CPU runs reach about 2.7 GB RSS and MLX on the GPU about 0.7 GB.
  The batched annotation peaks at 0.5 GB of live MLX buffers on the CPU and 1.6 GB on the
  GPU.
- This is why `auto` uses the GPU for MPO and the CPU for TRPO.

**MLX speedups for MPO.** The per-step costs of the M-step and the Q fit were the host work around each small
step: walking the trajectories for a minibatch, the copy from numpy and one dispatch per Adam step. The rollout on
the GPU paid a Metal round trip per environment step. Four changes address this:

1. The minibatches are gathered on the device from one flat copy of the replay memory, and the value fits run ten
   Adam steps per compiled call. The results are bit-identical: the rewards of all 60 epochs are the same as before
   on both devices.
2. With the networks on the GPU, the rollout evaluates the policy on the CPU stream. This changes the results by
   float noise only (about 5e-7 on a 75-step rollout, `tests/test_metal.py`), which compounds over training.
3. The resampling, the Q annotation and Retrace run over all trajectories in a few large calls instead of one call
   per trajectory. The annotations, the temperature dual and the ranking stay on the device (float noise; the ranking
   differs from the numpy one by at most 4.8e-7).
4. The temperature dual and the M-step run 20 and 10 Adam steps per compiled call, and the rollout reads the log
   likelihoods once per episode. These are bit-identical.

Re-timed on 2026-10-24: MPO, seed 1, 60 epochs, the median epoch of epochs 50–59 (full memory), 3 repeats each, one
run at a time with nice 10. The load average was 5–14, from desktop apps. The repeats differ by less than 4%.

| MLX device | before (ebd5a3b) | 1. flat replay, ten Adam steps per call | 1. + 2. rollout on the CPU stream |
|---|---|---|---|
| GPU | 2.89 s, 346 steps/s | 0.92 s, 1,084 steps/s (3.1×) | 0.69 s, 1,442 steps/s (4.2×) |
| CPU | 4.35 s, 230 steps/s | 2.90 s, 345 steps/s (1.5×) | – (the rollout is on the CPU already) |

Steps 3 and 4 were timed the same way, at a load average of 4–6, against 84b558e (steps 1 and 2) as the baseline.
The repeats differ by less than 1.5%. The baseline took 0.64 s on the GPU in this session (0.69 s above, at a
higher load).

| MLX device | 1. + 2. (84b558e) | 1.–4. | gain |
|---|---|---|---|
| GPU | 0.638 s, 1,568 steps/s | **0.468 s, 2,135 steps/s** | 1.36× |
| CPU | 2.785 s, 359 steps/s | **2.595 s, 385 steps/s** | 1.07× |

On the GPU, the Q fit (1,000 small Adam steps, 39% of the epoch) is now the largest phase, ahead of the rollout (17%)
and the temperature dual (14%). On the CPU, the annotation and Retrace (48%) dominate. Batching them is not faster
there, because the MLX CPU spends that time in the network arithmetic, not in dispatch.

### HalfCheetah-v5 (MPO)

Timed with the configuration of the repository (1,000 steps per epoch, 62 epochs), the median epoch of epochs 50–59
(full 50k memory), one run at a time with nice 10, at a load average of 4–6. The GPU number with the speedups is the
median of 3 repeats (0.470–0.490 s); the other cells are single runs.

| MLX device | before the speedups (ebd5a3b) | with the speedups | gain |
|---|---|---|---|
| GPU | 1.419 s, 705 steps/s | **0.477 s, 2,095 steps/s** | 2.97× |
| CPU | 3.092 s, 323 steps/s | **2.898 s, 345 steps/s** | 1.07× |

The gain on the GPU is smaller than on Pendulum because HalfCheetah episodes have 1,000 steps. The memory holds 50
trajectories instead of 250, so the per-trajectory host work that the speedups remove was smaller to begin with. With
the speedups, the GPU epoch splits into the Q fit (0.17 s), the rollout (0.10 s), the temperature dual (0.07 s), the
annotation and the M-step (0.04 s each) and Retrace (0.03 s).

### Learning (Pendulum-v1, 800k steps, 5 seeds, greedy return of the final policy over 20 episodes)

| Algorithm | MLX device | Greedy return (mean ± std over seeds) | Solved |
|---|---|---|---|
| TRPO | CPU | -124.5 ± 1.2 | 5/5 |
| MPO | GPU (`auto`), with speedups 1–4 | -122.9 ± 2.8 | 5/5 |
| MPO | GPU, with speedups 1–2 (84b558e) | -122.7 ± 1.5 | 5/5 |
| MPO | GPU, before the speedups (ebd5a3b) | -122.3 ± 1.6 | 5/5 |

Per seed (seeds 1–5): TRPO -123.4, -124.2, -126.5, -124.4, -124.1. MPO -120.4, -120.6, -125.8, -125.9, -121.7, with
speedups 1–2 -120.6, -122.4, -124.4, -122.4, -123.8, and before the speedups -120.9, -120.6, -122.3, -124.5, -123.1.
The wall time for 800k steps is about 2 min for TRPO (CPU) and about 8 min for MPO (GPU, two seeds at a time; 11 min
with speedups 1–2 and 44 min before the speedups).

### Verdict

- **The MLX port is a drop-in for the implementation on `main`.**
  - It keeps the architecture.
  - It matches `main` numerically on the same weights and batches (parity tests).
  - It learns: Pendulum 5/5 for both algorithms.
  - MPO with the full memory runs at 2,135 steps/s on the GPU (Pendulum-v1, 2,095 on HalfCheetah-v5), and TRPO at
    6,357 steps/s on the CPU.
- **The GPU only helps MPO.** That is where large batched operations over the replay memory dominate. TRPO stays
  bound by the per-step policy evaluation of the Gymnasium rollout and is best on the CPU.
- **Remaining limits:** no full covariance (Cholesky without a gradient), one extra workaround (ELU), and
  Apple-only GPU support.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy src                                        # types
uv run pytest                                          # unit and short training smoke tests
```

## Changes compared to the 2019 version

The original code (2019, Python 3.6, gym 0.12) targeted robot control experiments. The port keeps the
module layout and operator design. In short:

- uv + `pyproject.toml` (Python ≥ 3.13, src layout, `abstract_rl.*` imports), ruff, mypy, pytest; `gym` → `gymnasium`
  (`terminated`/`truncated`). On this branch the networks and the optimization use MLX, see
  [MLX port (experiment)](#mlx-port-experiment).
- Removed (still in the history of `main`): the experimental SAC, the dueling Q network, the circular tail
  memory, the robot specific scripts, unused helpers.
- Added: `DiagonalGaussianPolicy` for multi-dimensional actions, tanh action squashing, Hydra run configs,
  return based evaluation, SB3 baselines.
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
  title  = {Abstract RL: modular PyTorch TRPO and MPO},
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
