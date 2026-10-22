# Abstract RL

[![CI](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kosmitive/abstract_rl/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Modular PyTorch implementations of **TRPO** and **MPO** for continuous control, built from small composable pieces
(environment wrapper, trajectory operators, policies, value functions). Python 3.13, [uv](https://docs.astral.sh/uv/),
ruff, mypy; runs on Gymnasium environments with one- and multi-dimensional actions and is benchmarked against
Stable-Baselines3.

| MPO on Pendulum-v1 (swing-up from hanging down) | MPO on MountainCarContinuous-v0 | MPO on InvertedPendulum-v5 |
|---|---|---|
| ![MPO Pendulum](docs/mpo_pendulum_best.gif) | ![MPO MountainCar](docs/mpo_mountaincar_best.gif) | ![MPO InvertedPendulum](docs/mpo_inverted_pendulum_best.gif) |
| **MPO on InvertedDoublePendulum-v5** | **MPO on HalfCheetah-v5** | |
| ![MPO InvertedDoublePendulum](docs/mpo_inverted_double_pendulum_best.gif) | ![MPO HalfCheetah](docs/mpo_halfcheetah_best.gif) | |

The GIFs show one greedy episode of the best MPO policy per environment (highest greedy return over all runs).

- **TRPO**: Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) with
  [GAE](https://arxiv.org/abs/1506.02438), conjugate gradient natural gradient step and a KL line search.
- **MPO**: Maximum A Posteriori Policy Optimization ([Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920)) with
  a non-parametric E-step, a KL constrained M-step and a [Retrace](https://arxiv.org/abs/1606.02647) critic.
- Univariate and diagonal (multi-dimensional) Gaussian policies, optionally tanh-squashed into the action bounds.

## Install

```bash
git clone https://github.com/kosmitive/abstract_rl.git && cd abstract_rl
uv sync                                   # add --extra mujoco and/or baselines for the benchmark extras
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
```

PyTorch comes from the CPU wheel index (enough for these small networks); remove the `[tool.uv.sources]` entry in
`pyproject.toml` for the default CUDA build.

## JAX: many seeds at once

TRPO trains all seeds in one jitted, vmapped program:

```bash
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seeds=[1,2,3,4]
uv run python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6   # throughput
```

The `mps` extra (`uv sync --extra mps`) installs the Metal backend for JAX, selected with `jax_platform=mps`. Metal
runs the networks; the MJX environments need `jax_platform=cpu` (the default).

Measured on an Apple M3 Max (16 cores), JAX on CPU, 6 epochs (3 for HalfCheetah), TRPO with `policy_batch_size=4000`.
JAX env-steps/s is summed over seeds and excludes the compiling first call.

| env | seeds | JAX env-steps/s (cpu) | torch env-steps/s (1 seed, steady state) | speedup | first call (compile) |
|---|---|---|---|---|---|
| Pendulum | 1 | 55156 | 5333 | 10x | 4.0 s |
| Pendulum | 4 | 65374 | 5333 | 12x | 3.0 s |
| Pendulum | 16 | 71182 | 5333 | 13x | 4.1 s |
| HalfCheetah | 1 | 2043 | n/a | n/a | 27.7 s |
| HalfCheetah | 4 | 2757 | n/a | n/a | 18.2 s |

The torch figure is the last torch commit (4f7828e) running Pendulum TRPO (default config, one seed). Its steady state
is `6 * 4000 / (T12 - T6)`, where T6 = 6.93 s and T12 = 11.43 s are the wall times of a 6 and a 12 epoch run, so
interpreter startup and imports cancel out (a single 6 epoch run, startup included, gives only 1879 env-steps/s).
Torch HalfCheetah is not measured: the torch MuJoCo env is Gymnasium v5, with different dynamics and rewards. The
speedup is against the single-seed torch figure. MPO is still the torch implementation; a later plan ports it.

## Results

Configurations are Hydra YAML files in [`src/abstract_rl/conf/`](src/abstract_rl/conf) (`run env=<env> alg=<alg> seed=N`),
**5 seeds (1-5) per cell**, and every algorithm gets the same environment-step budget per environment (`steps` in `conf/env/<env>.yaml`).
*Greedy return* is the mean of 20 seeded evaluation episodes with the policy mean (final policy), averaged over seeds
± std over seeds. *Solved* counts seeds whose greedy return reaches the threshold (Gymnasium reward threshold;
Pendulum-v1 has none, -150 corresponds to swing-up and balance). *Steps to threshold* is the first step at which the
smoothed training return reaches it (median over the seeds that do).

| Environment | Algorithm | Seeds | Steps | Greedy return (mean ± std) | Solved | Steps to threshold |
|---|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO (abstract_rl) | 5 | 800,000 | -127.7 ± 2.7 | 5/5 | 192,000 (5/5) |
| Pendulum-v1 (≥ -150) | MPO (abstract_rl) | 5 | 800,000 | -123.3 ± 0.4 | 5/5 | 36,000 (5/5) |
| Pendulum-v1 (≥ -150) | TRPO (SB3) | 5 | 800,000 | -124.0 ± 1.6 | 5/5 | 204,000 (5/5) |
| Pendulum-v1 (≥ -150) | TD3 (SB3) | 5 | 800,000 | -120.9 ± 1.0 | 5/5 | 36,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (abstract_rl) | 5 | 401,254 | 94.1 ± 0.7 | 5/5 | 44,286 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | MPO (abstract_rl) | 5 | 400,038 | 93.6 ± 0.0 | 5/5 | 56,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (SB3) | 5 | 400,072 | 74.7 ± 37.5 | 4/5 | 52,013 (4/5) |
| MountainCarContinuous-v0 (≥ 90) | TD3 (SB3) | 5 | 399,966 | 54.9 ± 77.4 | 4/5 | 28,993 (4/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (abstract_rl) | 5 | 151,370 | 1000.0 ± 0.0 | 5/5 | 28,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | MPO (abstract_rl) | 5 | 150,357 | 1000.0 ± 0.0 | 5/5 | 20,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (SB3) | 5 | 150,523 | 1000.0 ± 0.0 | 5/5 | 29,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TD3 (SB3) | 5 | 149,310 | 1000.0 ± 0.0 | 5/5 | 25,000 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (abstract_rl) | 5 | 605,543 | 9332.8 ± 3.8 | 5/5 | 431,112 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | MPO (abstract_rl) | 5 | 600,620 | 9332.9 ± 7.4 | 5/5 | 66,110 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (SB3) | 5 | 599,517 | 8270.3 ± 602.2 | 0/5 | 107,946 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TD3 (SB3) | 5 | 599,243 | 9347.9 ± 10.3 | 5/5 | 26,996 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (abstract_rl) | 5 | 1,200,000 | 4471.7 ± 1645.7 | 3/5 | 732,000 (2/5) |
| HalfCheetah-v5 (≥ 4800) | MPO (abstract_rl) | 5 | 1,200,000 | 7940.5 ± 1398.2 | 5/5 | 210,000 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (SB3) | 5 | 1,200,000 | 2547.9 ± 1314.1 | 0/5 | never |
| HalfCheetah-v5 (≥ 4800) | TD3 (SB3) | 5 | 1,200,000 | 11009.6 ± 522.5 | 5/5 | 96,000 (5/5) |

Learning curves of the four algorithms on environment steps (training return, mean ± std over the 5 seeds; the
runs of the table):

![benchmark](docs/benchmark.png)

- **Tuned for Pendulum-v1 and MountainCarContinuous-v0** (one setting each): TRPO Pendulum discount 0.95, MPO
  Pendulum E-step epsilon `rnk_eps` 0.05, MPO MountainCar fixed (not learned) exploration std 2.0. A learned std
  shrinks to ≈ 0 before the sparse goal is found, so MountainCar uses a fixed standard deviation. These runs stop at the exact
  step budget (`max_env_steps`).
- **Pendulum balance check** (`scripts/eval_balance.py`, |θ| < 0.1 rad in the last 100 of 200 steps, 20 start
  states): TRPO and MPO balance in 5/5 seeds and 20/20 episodes each (max |θ| 0.012-0.057 rad TRPO, 0.032-0.098 rad MPO).
- **SB3 baselines**: TRPO uses the library defaults, except on MountainCarContinuous-v0, which uses the
  [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyper-parameters (`--zoo`, gSDE exploration) at
  the same 400k budget; with the defaults it scores 0 there. On Pendulum-v1 the zoo settings were worse than the
  defaults (-136), so the defaults are shown. TD3 always uses the zoo hyper-parameters (`ZOO` in
  `scripts/run_baselines.py`); on MountainCarContinuous-v0 one of five seeds (seed 3) never finds the goal (-99.9).
- **MPO tuned for InvertedPendulum and InvertedDoublePendulum** (one setting each, see `conf/tune/`): a 300k
  instead of 50k replay memory (`tc_size`) on InvertedPendulum (4/5 → 5/5 seeds) and a KL trust region
  (`mpo_trust_regions`) of 0.005 instead of 0.01 on InvertedDoublePendulum (3/5 → 5/5); both remove the late-training
  collapse of single seeds. HalfCheetah MPO is not re-tuned. MPO stops after a fixed number of epochs and finishes the
  running episode, so on the two pendulums it ran up to 5% (InvertedPendulum) and 7% (InvertedDoublePendulum) past the
  budget; the learning curves are cut at the budget, the greedy return is the one of the final policy.
- **TRPO tuned for InvertedDoublePendulum and HalfCheetah** (see `conf/tune/`): InvertedDoublePendulum 75 × 8000
  steps with a KL trust region of 0.005, HalfCheetah 300 × 4000 steps with a KL trust region of 0.02 and `obs_norm`,
  both without the value L2 penalty (InvertedDoublePendulum 2/5 → 5/5, HalfCheetah 0/5 → 3/5 seeds).
- Runs were made on Linux and a MacBook (Python 3.13) with one CPU thread; seeds are fixed but
  results are not bit-reproducible across machines.

## Usage

```bash
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=1 alg.num_epochs=100  # Hydra overrides
uv run python -m abstract_rl.scripts.run env=mountaincar alg=mpo data_root=results -m seed=1,2,3,4,5  # sweep
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo --cfg job              # print the config
uv run python -m abstract_rl.scripts.eval_returns data/HalfCheetah-v5/mpo/<run> --save       # greedy evaluation
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run>              # Pendulum balance check
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 td3 --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/                         # table and plots
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render         # replay a policy
```

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

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy src                                        # types
uv run pytest                                          # unit and short training smoke tests
```

## Changes compared to the 2019 version

The original code (2019, Python 3.6, PyTorch 1.0, gym 0.12) targeted robot control experiments. The port keeps the
module layout and operator design. In short:

- uv + `pyproject.toml` (Python ≥ 3.13, src layout, `abstract_rl.*` imports), ruff, mypy, pytest; `gym` → `gymnasium`
  (`terminated`/`truncated`); modern PyTorch APIs.
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
