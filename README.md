# Abstract RL

Modular JAX implementations of **TRPO** and **MPO** for continuous control, built from small composable pieces
(environment wrapper, trajectory operators, policies, value functions). Networks are Equinox modules and the
optimisers are Optax. Python 3.13, [uv](https://docs.astral.sh/uv/), ruff, mypy; runs on Gymnasium environments
with one- and multi-dimensional actions and is benchmarked against Stable-Baselines3.

| MPO on Pendulum-v1 (swing-up from hanging down) | MPO on MountainCarContinuous-v0 |
|---|---|
| ![MPO Pendulum](docs/mpo_pendulum_best.gif) | ![MPO MountainCar](docs/mpo_mountaincar_best.gif) |

- **TRPO**: Trust Region Policy Optimization ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) with
  [GAE](https://arxiv.org/abs/1506.02438), conjugate gradient natural gradient step and a KL line search.
- **MPO**: Maximum A Posteriori Policy Optimization ([Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920),
  and [1812.02256](https://arxiv.org/abs/1812.02256)) with a non-parametric E-step, a decoupled mean/covariance KL
  in the M-step (ε_μ = 0.1, ε_Σ = 1e-4), a state-dependent Cholesky Gaussian, an Adam dual for the temperature,
  and a [Retrace](https://arxiv.org/abs/1606.02647) critic. TRPO keeps a state-independent std
  (`policy_cov_type: single`).
- Gaussian policies, optionally tanh-squashed into the action bounds. `policy_cov_type` is `full` (state-dependent
  Cholesky, the MPO default), `single` (one learned std), `fixed` (a constant std) or `state` (a state-dependent
  diagonal). One-dimensional actions use the univariate policy unless the covariance is `full`.

## Install

Python 3.13 (`requires-python` is `>=3.13,<3.14`). The package installs from git. Hydra configs are inside the wheel, so training does not need a checkout.

```bash
uv add git+https://github.com/kosmitive/abstract_rl
uv add "abstract-rl[mujoco] @ git+https://github.com/kosmitive/abstract_rl"
uv add "abstract-rl[baselines] @ git+https://github.com/kosmitive/abstract_rl"
uv add "abstract-rl[dev] @ git+https://github.com/kosmitive/abstract_rl"

pip install git+https://github.com/kosmitive/abstract_rl
pip install "abstract-rl[mujoco] @ git+https://github.com/kosmitive/abstract_rl"
pip install "abstract-rl[baselines] @ git+https://github.com/kosmitive/abstract_rl"
pip install "abstract-rl[dev] @ git+https://github.com/kosmitive/abstract_rl"
```

From a checkout, `uv sync` installs the same tree. `--extra mujoco` and `--extra baselines` add those extras. `--group dev` (the same packages as the `dev` extra) adds pytest, ruff and mypy.

```bash
abstract-rl env=pendulum alg=trpo seed=1
abstract-rl env=pendulum alg=trpo --cfg job          # print the composed Hydra config
python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1   # same entry point
```

```python
from abstract_rl import make_config, run_trpo

conf = make_config(
    "pendulum",
    "trpo",
    seed=1,
    overrides=["env.steps=400", "alg.env_sample=400", "data_root=data"],
)
returns = run_trpo(conf)  # shape (epochs, 2): mean and std of the raw episodic return
```

The JAX dependency is the CPU extra `jax[cpu]` (a fresh install reports a CPU device). `jax-metal` is not a dependency, and the package does not install a CUDA build of JAX. The `baselines` extra installs Stable-Baselines3, which depends on PyTorch. The core package does not. On Linux that torch wheel comes from PyPI and pulls the CUDA libraries; `tool.uv.sources` names a CPU index, and the lockfile currently records the PyPI wheel. `mujoco` adds the Gymnasium MuJoCo environments.

## Results

JAX greedy evaluation at the default environment-step budgets, 5 seeds per cell. A per-epoch timing comparison with the previous implementation is in [Epoch time](#epoch-time).

Configurations are Hydra YAML files in [`config/`](config) (`abstract-rl env=<env> alg=<alg> seed=N`).
*Greedy return* for abstract_rl is the mean over the five seeds. Each seed is the mean of 20 evaluation episodes (start seeds 10000 + episode, policy mean, raw Gymnasium return) of the final policy. *Solved* counts seeds that reach the threshold (the Gymnasium reward threshold; Pendulum-v1 has none, so -150 is swing-up and balance). abstract_rl stops at `steps` in `config/env/<env>.yaml`. The Stable-Baselines3 rows are the earlier comparison and still report mean ± std, the step count that comparison recorded, and *steps to threshold* (the first step at which the smoothed training return reaches the threshold, median over the seeds that do). Steps to threshold was not recorded for these JAX runs.

| Environment | Algorithm | Seeds | Steps | Greedy return | Solved | Steps to threshold |
|---|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO (abstract_rl) | 5 | 800,000 | -128.7 | 5/5 | — |
| Pendulum-v1 (≥ -150) | MPO (abstract_rl) | 5 | 800,000 | -121.9 | 5/5 | — |
| Pendulum-v1 (≥ -150) | TRPO (SB3) | 5 | 800,000 | -124.0 ± 1.6 | 5/5 | 204,000 (5/5) |
| Pendulum-v1 (≥ -150) | PPO (SB3) | 5 | 800,000 | -156.1 ± 38.5 | 3/5 | 536,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (abstract_rl) | 5 | 400,000 | 94.7 | 5/5 | — |
| MountainCarContinuous-v0 (≥ 90) | MPO (abstract_rl) | 5 | 400,000 | 93.6 | 5/5 | — |
| MountainCarContinuous-v0 (≥ 90) | TRPO (SB3) | 5 | 400,072 | 74.7 ± 37.5 | 4/5 | 52,013 (4/5) |
| MountainCarContinuous-v0 (≥ 90) | PPO (SB3) | 5 | 399,927 | 54.8 ± 76.7 | 4/5 | 5,998 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (abstract_rl) | 5 | 150,000 | 910.3 | 4/5 | — |
| InvertedPendulum-v5 (≥ 950) | MPO (abstract_rl) | 5 | 150,000 | 1000 | 5/5 | — |
| InvertedPendulum-v5 (≥ 950) | TRPO (SB3) | 5 | 150,523 | 1000.0 ± 0.0 | 5/5 | 29,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | PPO (SB3) | 5 | 150,220 | 1000.0 ± 0.0 | 5/5 | 28,000 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (abstract_rl) | 5 | 600,000 | 8579.6 | 3/5 | — |
| InvertedDoublePendulum-v5 (≥ 9100) | MPO (abstract_rl) | 5 | 600,000 | 9345.6 | 5/5 | — |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (SB3) | 5 | 599,517 | 8270.3 ± 602.2 | 0/5 | 107,946 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | PPO (SB3) | 5 | 599,587 | 9175.8 ± 225.2 | 3/5 | 197,967 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (abstract_rl) | 5 | 1,200,000 | 2336.3 | 0/5 | — |
| HalfCheetah-v5 (≥ 4800) | MPO (abstract_rl) | 5 | 1,200,000 | 6889.3 | 4/5 | — |
| HalfCheetah-v5 (≥ 4800) | TRPO (SB3) | 5 | 1,200,000 | 2547.9 ± 1314.1 | 0/5 | never |
| HalfCheetah-v5 (≥ 4800) | PPO (SB3) | 5 | 1,200,000 | 1620.3 ± 59.9 | 0/5 | never |

A seed counts as solved on its own greedy return, so the five-seed mean can sit below the threshold while most seeds solve (InvertedPendulum TRPO 910.3 with 4/5, InvertedDoublePendulum TRPO 8579.6 with 3/5).

![benchmark](docs/benchmark.png)

`docs/benchmark.png` is the learning-curve figure from the earlier PyTorch abstract_rl runs plotted against these same SB3 baselines. The greedy numbers above are the JAX runs at the default budgets.

- **Tuned for Pendulum-v1 and MountainCarContinuous-v0** (one setting each): TRPO Pendulum discount 0.95, MPO
  Pendulum E-step epsilon `rnk_eps` 0.05. Every environment sets `policy_init_std: 0.5` (a placeholder).
  MountainCar is the only environment with `policy_min_std` (0.3, annealed over the first 160000 steps). A learned
  std that shrinks to about 0 before the sparse goal is found made 2/5 seeds fail before. MountainCar MPO keeps a
  fixed covariance (`policy_cov_type: fixed`), which is the univariate policy because the action is one dimensional.
  That std is `policy_init_std` 0.5. The training budget is each environment's `steps` field.
- **Pendulum balance check** (`abstract_rl.scripts.eval_balance`, |θ| < 0.1 rad in the last 100 of 200 steps). The command is in [Usage](#usage).
- **SB3 baselines**: library defaults, except MountainCarContinuous-v0, which uses the
  [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyper-parameters (`--zoo`, gSDE exploration) at
  the same 400k budget; with defaults both SB3 algorithms score 0 there. On Pendulum-v1 the zoo settings were worse
  than the defaults (TRPO -136, PPO -226), so the defaults are shown.
- **InvertedPendulum, InvertedDoublePendulum, HalfCheetah** use the overrides in `config/env` and are not separately re-tuned. Solved counts for the JAX runs are in the table.
- **Humanoid-v5** is in the environment list and has no 5-seed result yet. The budget is 3,000,000 steps, which is shorter than the 5–10 million steps usually used for this task. Gymnasium sets no reward threshold. `healthy_reward` is 5 per step, so 5000 is only standing for the full 1000-step episode. The solved line is 8000, which requires forward reward and so a real walk. The observation is 348 and the action is 17 torques in [-0.4, 0.4]. A fall (torso height outside [1, 2]) terminates the episode. TRPO uses a [256, 256] policy and value net and `env_sample` 15,000 (200 epochs). MPO uses a [256, 256] policy, the existing [256, 256] critic, `env_sample` 1000 (3000 epochs) and initial std `policy_init_std` 0.5. The M-step constraints are the decoupled bounds 0.1 and 1e-4. `abstract-rl env=humanoid alg=trpo seed=1` needs the `mujoco` extra. On this 4-CPU machine, after the compile epoch, TRPO takes 6.66 s/epoch (rollout about 3100 env steps/s), so 200 epochs is about 22 minutes. MPO takes 3.29 s/epoch while the replay still holds a few thousand rows (rollout about 3000 env steps/s, the critic is most of the epoch). At that rate 3000 epochs is about 2.7 hours; a full replay adds the resample and Retrace work on top.

## Usage

```bash
abstract-rl env=halfcheetah alg=mpo seed=1 env.steps=100000             # shorter budget; needs the mujoco extra
abstract-rl -m env=mountaincar alg=mpo data_root=results seed=1,2,3,4,5  # Hydra sweep; -m has to come first
abstract-rl env=pendulum alg=trpo --cfg job                              # print the config
abstract-rl-eval data/HalfCheetah-v5/mpo/<run> --save                    # greedy evaluation
python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run>   # Pendulum balance check
abstract-rl-baselines HalfCheetah-v5 ppo --seed 1 --data-root results    # baselines extra
abstract-rl-compare results --out docs/                                  # writes benchmark.md and benchmark.png
python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run>         # replay; add --render on a display
```

```python
from abstract_rl import make_config, run_mpo_non_parametric, run_trpo

# The YAML defaults are the benchmark budgets. These overrides are a one-epoch smoke run.
run_trpo(make_config("pendulum", "trpo", seed=1, overrides=["env.steps=400", "alg.env_sample=400"]))
run_mpo_non_parametric(make_config("mountaincar", "mpo", seed=1, overrides=["env.steps=200", "alg.env_sample=200"]))
```

Configuration ([Hydra](https://hydra.cc)): `config/config.yaml` (`seed`, `data_root`),
`config/alg/{trpo,mpo}.yaml` (Pendulum-v1 defaults) and `config/env/<env>.yaml` (Gymnasium `id`, `success`
threshold, `steps` budget, `overrides` of the algorithm defaults for `all`/`trpo`/`mpo`). Environments: `pendulum`,
`mountaincar`, `inverted_pendulum`, `inverted_double_pendulum`, `halfcheetah`, `humanoid`. `alg.<key>=...` overrides beat the
environment table. The training loops get a flat dictionary with prefixes (`env_*`, `policy_*`, `val_*`, `trpo_*`,
`mpo_*`). Runs are written to `data/<env>/<alg>/<timestamp>/` (configuration, returns, models, plots).
`alg=mpo` is the paper M-step: decoupled KL, Cholesky policy, SGD temperature. TRPO keeps a state-independent std
(`policy_cov_type: single`). `select_policy` uses the univariate or the diagonal Gaussian when the covariance is not `full`.
Both algorithms collect `env_sample` transitions per epoch. The epoch count is `ceil(steps / env_sample)`. MPO's
`policy_batch_size` is the M-step minibatch.

## Training environments

| Environment | Training rollout | Evaluation |
|---|---|---|
| Pendulum-v1 | JAX step in `env/jax_classic.py`, `lax.scan` over time and `vmap` over parallel episodes | Gymnasium |
| MountainCarContinuous-v0 | same JAX step | Gymnasium |
| InvertedPendulum-v5 | Gymnasium (MuJoCo) | Gymnasium |
| InvertedDoublePendulum-v5 | Gymnasium (MuJoCo) | Gymnasium |
| HalfCheetah-v5 | Gymnasium (MuJoCo) | Gymnasium |
| Humanoid-v5 | Gymnasium (MuJoCo) | Gymnasium |

Each collection stores exactly the transitions the epoch asked for, including when several environments step together. Both training loops stop once `steps` transitions are stored.

The classic-control steps copy the Gymnasium formulas. gymnax Pendulum-v1 matches those formulas. gymnax MountainCarContinuous-v0 does not: it zeros velocity at the goal when velocity is negative, and it folds the time limit into `done`. This step zeros velocity only on the left wall (`position == -1.2` and velocity negative), and a time limit is not a done. Reset ranges match; the generator is JAX, so a seeded Gymnasium reset is not the same sequence.

Parity is an open-loop rollout from the same physics state with the same actions (`tests/test_jax_classic.py`). Over 200 Pendulum steps, float64 state error is 0 and the reward error is 0; the observation differs by at most 2.4e-7 because Gymnasium casts it to float32. Float32, which is what training runs, drifts by 3.8e-5 in the state and 8.4e-5 in the reward. Over 200 MountainCar steps the float64 state differs by 2.1e-7 because Gymnasium stores that state as float32 after every step, and the reward matches. Float32 state error is 3.6e-7.

MuJoCo stays on Gymnasium. MJX on this 4-CPU machine, same XML, `frame_skip` as a scan of `mjx.step` (there is no `nstep` argument), 30 matched actions, float32 MJX against float64 MuJoCo:

| Environment | Gymnasium steps/s | MJX steps/s | max \|Δqpos\| | max \|Δqvel\| |
|---|---:|---:|---:|---:|
| InvertedPendulum-v5 | 46679 | 2506 | 2.6e-7 | 1.3e-6 |
| InvertedDoublePendulum-v5 | 20001 | 1998 | 8.1e-6 | 9.6e-5 |
| HalfCheetah-v5 | 23455 | 1563 | 4.0e-4 | 2.5e-2 |

MJX is 10–19× slower here, and HalfCheetah velocity has already drifted by 2.5e-2. The Warp backend was not installed. brax and mujoco_playground are different tasks (different models, reset noise, rewards and terminations), so they are not drop-in replacements. The reward, observation and termination of the v5 environments live in the Gymnasium Python around `mj_step`.

## Epoch time

Wall-clock seconds per epoch on one 4-CPU Linux machine, Hydra defaults, `seed=1`. The PyTorch column is the previous implementation pinned to one thread. "JAX loops" is the port after the scan pass and before the classic-control rollout. "JAX now" skips epoch 0 (the compile). The later epochs are flat, so the compile sits in epoch 0. The MPO critic time in this table is the Adam scan before it was unrolled; the shipped scan is below.

| Run | PyTorch | JAX loops | JAX now | Window after compile |
|---|---:|---:|---:|---|
| TRPO Pendulum-v1 | 2.676 | 0.336 | 0.123 | epochs 1–4 |
| MPO Pendulum-v1 | 2.686 | 1.416 | 1.361 | epochs 1–7 (8k of the 50k buffer) |
| TRPO HalfCheetah-v5 | 3.103 | 1.070 | 0.953 | epochs 1–3 |
| MPO HalfCheetah-v5 | 3.270 | 1.475 | 1.489 | epochs 1–5 |

On the same windows, the TRPO Pendulum rollout is 0.041s, the trust-region step 0.013s and the value fit 0.026s. TRPO HalfCheetah rollout stays on Gymnasium at 0.801s, the trust-region step is 0.049s and the value fit 0.059s. Those value-fit seconds are the Optax L-BFGS fit. The fit in this tree is minibatch Adam: 10 passes over the rollout, minibatches of 64, step size 3e-4 (`val_epochs`, `val_batch_size`, `val_lr`). MPO Pendulum rollout is 0.018s. MPO HalfCheetah rollout is 0.084s. The MPO epoch is still the resample, Retrace and critic. The shipped critic scan unrolls 8 Adam steps. A later measurement of that 1000 × 256 fit on a 4-CPU machine is 503 ms, against 690 ms for the same scan unrolled one step at a time.

Phases of the earlier loop pass, in seconds (mean before that pass, then mean after). The epoch table above is the later measurement.

| Phase | TRPO Pendulum | TRPO HalfCheetah | MPO Pendulum | MPO HalfCheetah |
|---|---:|---:|---:|---:|
| act | 0.442 → 0.048 | 1.449 → 0.136 | 0.144 → 0.016 | 0.346 → 0.034 |
| env step | 0.107 → 0.066 | 0.759 → 0.494 | 0.036 → 0.021 | — |
| rollout | 0.664 → 0.193 | 2.504 → 0.809 | 0.220 → 0.064 | 0.451 → 0.083 |
| TRPO step | 0.644 → 0.016 | 1.232 → 0.050 | — | — |
| value fit | 0.247 → 0.081 | 0.362 → 0.151 | — | — |
| resample | — | — | 0.261 → 0.469 | 0.242 → 0.573 |
| temperature | — | — | 0.198 → 0.057 | 0.185 → 0.050 |
| ranking | — | — | 0.074 → <0.005 | 0.082 → <0.005 |
| Retrace | — | — | 0.021 → 0.094 across epochs 1–11 | 0.229 → 0.063 |
| M-step | — | — | 0.124 → 0.036 | 0.172 → 0.065 |
| critic | — | — | 0.699 → 0.686 | 0.605 → 0.594 |
| device sync | — | — | climbed 0.06 → 0.22; now <0.01 | now <0.01 |

Those phase times predate the filled-prefix scans. Resample, the temperature dual, ranking, target statistics and Retrace now run on a power-of-two prefix of the filled rows (`misc/buckets.py`), and on the whole allocation once the next power of two would pass the capacity. On a 50,000-row Pendulum buffer that switch happens after 32,768 stored transitions. The full-buffer resample costs above (about 0.47s on Pendulum, 0.57s on HalfCheetah) are the steady-state figures. Retrace still skips empty episode slots. In that earlier window it grew with the stored Pendulum episodes (0.021s at epoch 1, 0.094s at epoch 11) while the epoch stayed between 1.39s and 1.44s. HalfCheetah stores one episode per epoch, and the first eight share one slot chunk, so Retrace stayed flat in that window.

The 5-seed greedy evaluation after this round is in [Results](#results). On an earlier seed-1 run, before the loop pass, the stochastic Pendulum return moved from -1172 to -178 over 40 TRPO epochs and from -1477 to -150 over 40 MPO epochs.

### Loops

Each training method keeps its name and its maths. The inner iteration is a JAX loop. Timings are from this machine.

| Loop | File | Became | Measured |
|---|---|---|---|
| Retrace backup `k = 1 .. bootstrap` | `operator/retrace_operator.py` `retrace_accumulate` | `lax.scan` | max abs err 6.7e-16. A 32×1000 backup is 0.0001s unrolled and 0.0006s as a scan (0.21×). The epoch win is the fused slot kernel: Pendulum Retrace 0.207s → 0.021s at the same fill |
| Retrace chunks, host pad, Python loop | `operator/retrace_operator.py` `apply` | one `filter_jit`; `lax.cond` skips empty chunks; device scatter-add; donated `q` and `tq` | included in the Retrace timing above |
| Resample chunks and a `.at.set` per chunk | `operator/resample_actions_operator.py` `apply` | `lax.scan` over a fixed chunk count; donated `add_act`, `add_q`, `add_pi` | flat 0.47s Pendulum / 0.57s HalfCheetah. The previous path started near 0.24s and rose with the buffer |
| Ranking on `add_q[:size]` | `operator/np_rew_operator.py` | `filter_jit` on a power-of-two prefix, traced fill | 0.07–0.10s → under 0.005s on the full buffer. A new width compiles once |
| Temperature L-BFGS-B with `float()` each iteration | `algorithms/mixed/non_parametric_trust_region_optimizatioon.py` | one Adam step on the dual inside the M-step scan. An intermediate port measured damped Newton; that solve is not in the training loop | Newton, (20000, 20) float64: 1.031s → 0.052s (19.7×). On a (256, 20) draw that optimum had KL 0.1000 against ε 0.1 |
| TRPO gradient, Fisher, and Python conjugate gradient, with a new jit every epoch | `trust_region_gradient.py`, `conjugate_gradient.py` | one `filter_jit`; `cg_solve` is `lax.while_loop` | Pendulum TRPO step 0.644s → 0.016s (40×); HalfCheetah 1.232s → 0.050s (25×) |
| Line search `while` with `float(tr_obj)` per probe | `trust_region_gradient.py` `line_search` | `_backtrack` `lax.while_loop`; one log copy; one parameter write | 10 probes on a 4000-step batch: 0.0036s → 0.0033s (1.10×) |
| Value fit, SciPy L-BFGS-B, host `float()` each iteration | `value_functions/fc_v_networks.py` `perform_sgd` | minibatch Adam inside one `lax.scan` (`_fit_v`): 10 epochs, batch 64, step size 3e-4 | L-BFGS measurement, same 25 steps on a 64-64 value net: Pendulum-sized batch 0.067s → 0.022s, loss 1.02018 → 1.02031; HalfCheetah-sized 0.138s → 0.048s, loss 0.95017 → 0.94989. In the epoch: 0.081s → 0.026s and 0.151s → 0.059s |
| Critic Adam Python loop | `value_functions/fc_q_networks.py` | `lax.scan` unrolled 8 steps, donated weights and Adam state | loop pass 0.69s; shipped 1000 × 256 scan 503 ms (690 ms unrolled once) |
| M-step Python loop on the default path | `maximum_a_posteriori_policy_optimization_algorithm.py` `_mpo_scan` | `lax.scan`, with donated weights, both Adam states, and the multiplier | Pendulum 0.124s → 0.036s (3.4×); HalfCheetah 0.172s → 0.065s (2.6×) |
| GAE and TD reverse loops | `gae_operator.py`, `td_operator.py` | `lax.scan` | a few hundredths of a second after epoch 0 |
| `policy.act` once per environment step | `continuous_policy.py`, `numpy_act.py` | NumPy matmul; weights cached on `id(params)` | HalfCheetah act 1.449s → 0.136s (11×); Pendulum 0.442s → 0.048s (9×) |
| Gymnasium rollout `while` | `env/mcmc_env_wrapper.py` | JAX `lax.scan` + `vmap` for Pendulum and MountainCarContinuous (`env/jax_classic.py`). MuJoCo stays Python | Pendulum rollout 0.193s → 0.041s. HalfCheetah stays Gymnasium, 0.801s |
| Replay `.at[start:end].set` with Python bounds | `replay_buffer.py` | one host pack, then `_insert_transitions` / `_shift_transitions` (`jnp.where`, traced start and length, donated buffers) | device sync stays under 0.01s, and steady MPO epochs compile 0 times |
| Q evaluation over the K sampled actions | `fc_q_networks.py` `q_values` | one matmul with the action axis folded into the batch | counted inside resample and Retrace |

Host `if` statements remain on concrete values: empty buffers, `n_envs == 1`, and which environment has a JAX step. Traced branches inside the jitted updates are `lax.cond` or `jnp.where`. The value fit is minibatch Adam (`val_lr` 3e-4, `val_epochs` 10, `val_batch_size` 64). An earlier Optax zoom L-BFGS fit was faster than Optax backtracking and than optimistix BFGS/L-BFGS on the value loss, and it matched the SciPy loss. `jax` and `jaxlib` still install SciPy themselves; the library does not import it and does not list it. jaxopt L-BFGS is comparable and is not a dependency. The single-trajectory resample path still chunks in Python. Training uses the device scan. Equinox takes `donate="all-except-first"` rather than `donate_argnums`. MuJoCo acting is still the NumPy policy; Pendulum and MountainCarContinuous sample with `gaussian_math` inside the scan.

## Architecture

```
src/abstract_rl/
├── algorithms/
│   ├── continuous/policy_gradient/   TRPO and MPO (policy update rules)
│   └── mixed/                        conjugate gradient, trust region gradient (natural gradient + line search),
│                                     vanilla gradient (Adam), non-parametric temperature dual
├── data_structures/
│   ├── abstract_conf/                SharedConf (namespaced config), ModelConfiguration (registry, target sync, IO)
│   └── temporal_difference_data/     Trajectory, TrajectoryCollection, on-device ReplayBuffer, Batch
├── env/                              MCMCEnvWrapper; JAX steps for Pendulum and MountainCar, Gymnasium otherwise
├── operator/                         trajectory operators: TD, GAE, Retrace, action resampling, MPO reweighting
├── policy/                           univariate, diagonal and Cholesky Gaussian policies (Equinox), E-step temperature
├── value_functions/                  fully connected V (minibatch Adam) and Q (Adam) networks
├── neural_modules/                   Equinox MLP building block
├── run/                              training loops (run_trpo, run_mpo_non_parametric), policy replay
├── scripts/                          Hydra entry point (run.py, hydra_conf.py), evaluate / baselines / compare
└── misc/ and plots/                  CLI printer, data logger, observation normalisation, plots
```

A training loop is a sequence of operator applications on trajectory collections:

- **TRPO** per epoch: collect `env_sample` steps → `TD` operator (returns, bootstrapped at time limits) →
  `GAE` operator → natural gradient step with KL trust region → V function fit (minibatch Adam, `grad_clip`).
- **MPO** per epoch: collect `env_sample` steps into the trajectory collection → `Resample` operator (actions from the
  target policy, evaluated by the target Q) → temperature dual + `VariationalRanking` operator (non-parametric
  E-step) → KL constrained weighted maximum likelihood (M-step) → `Retrace` operator → Q function fit (Adam).
  The collection keeps an on-device copy, and the default M-step and the critic Adam loop are each one `lax.scan`.
  Each KL multiplier is `softplus(lm) + 1e-8`, clipped to `[mpo_alpha_min, mpo_alpha_max]` after the dual step.
  `grad_clip` (default 40, `0` off) caps the policy, critic and multiplier gradients. The M-step log prints
  `alpha`, the KL and the slack (`eps - KL`) as separate columns. Episode entropy is `-mean log π(a|s)`.

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

- uv + `pyproject.toml` (Python 3.13, src layout, `abstract_rl.*` imports), ruff, mypy, pytest; `gym` → `gymnasium`
  (`terminated`/`truncated`). Networks are Equinox modules, optimisers are Optax, and the core package does not
  depend on PyTorch.
- Removed (still in the history of `main`): the experimental SAC, the dueling Q network, the circular tail
  memory, the robot specific scripts, unused helpers.
- Added: `DiagonalGaussianPolicy` for multi-dimensional actions, tanh action squashing, table driven benchmark
  configs, return based evaluation, SB3 baselines.
- Bug fixes needed for both algorithms to learn (see the commit history):
  TRPO surrogate broadcast (`[D,1]*[D]`), target-policy sync without sigma, time-limit episodes treated as terminal,
  replay index sampling, Q-function target broadcasting and state/action pairing, Retrace off-by-one and aliasing,
  MPO per-state weight normalisation and exact temperature dual, L-BFGS closure without `zero_grad`, unbounded
  Gaussian actions (now tanh-squashed with the Jacobian in the log-likelihood), Retrace policy mismatch, KL check
  in the TRPO line search, a larger ELU critic for MPO, and a state-dependent sigma that stays positive without
  being passed through softplus twice.

## License

MIT, see [LICENSE](LICENSE).
