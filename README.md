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
- **MPO**: Maximum A Posteriori Policy Optimization ([Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920)) with
  a non-parametric E-step, a KL constrained M-step and a [Retrace](https://arxiv.org/abs/1606.02647) critic.
- Univariate and diagonal (multi-dimensional) Gaussian policies, optionally tanh-squashed into the action bounds.

## Install

```bash
git clone https://github.com/kosmitive/abstract_rl.git && cd abstract_rl
uv sync                                   # add --extra mujoco and/or baselines for the benchmark extras
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1
```

JAX is installed from the CPU extra (`jax[cpu]`). A machine with CUDA can still run on the GPU; `jax-metal` is
not used. The optional `baselines` extra installs Stable-Baselines3, and with it a CPU PyTorch wheel. That
dependency stays out of the core package.

## Results

The table below is the previous PyTorch run of this repository (5 seeds, same Hydra configs). The JAX port keeps
those configs and the algorithm. A per-epoch timing comparison of the two implementations is in
[Epoch time](#epoch-time).

Configurations are Hydra YAML files in [`config/`](config) (`run env=<env> alg=<alg> seed=N`),
**5 seeds (1-5) per cell**, and every algorithm gets the same environment-step budget per environment (`steps` in `config/env/<env>.yaml`).
*Greedy return* is the mean of 20 seeded evaluation episodes with the policy mean (final policy), averaged over seeds
± std over seeds. *Solved* counts seeds whose greedy return reaches the threshold (Gymnasium reward threshold;
Pendulum-v1 has none, -150 corresponds to swing-up and balance). *Steps to threshold* is the first step at which the
smoothed training return reaches it (median over the seeds that do).

| Environment | Algorithm | Seeds | Steps | Greedy return (mean ± std) | Solved | Steps to threshold |
|---|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO (abstract_rl) | 5 | 800,000 | -127.7 ± 2.7 | 5/5 | 192,000 (5/5) |
| Pendulum-v1 (≥ -150) | MPO (abstract_rl) | 5 | 800,000 | -123.8 ± 1.0 | 5/5 | 40,000 (5/5) |
| Pendulum-v1 (≥ -150) | TRPO (SB3) | 5 | 800,000 | -124.0 ± 1.6 | 5/5 | 204,000 (5/5) |
| Pendulum-v1 (≥ -150) | PPO (SB3) | 5 | 800,000 | -156.1 ± 38.5 | 3/5 | 536,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (abstract_rl) | 5 | 401,254 | 94.1 ± 0.7 | 5/5 | 44,286 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | MPO (abstract_rl) | 5 | 400,039 | 93.6 ± 0.0 | 5/5 | 52,000 (5/5) |
| MountainCarContinuous-v0 (≥ 90) | TRPO (SB3) | 5 | 400,072 | 74.7 ± 37.5 | 4/5 | 52,013 (4/5) |
| MountainCarContinuous-v0 (≥ 90) | PPO (SB3) | 5 | 399,927 | 54.8 ± 76.7 | 4/5 | 5,998 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (abstract_rl) | 5 | 151,370 | 1000.0 ± 0.0 | 5/5 | 28,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | MPO (abstract_rl) | 5 | 150,663 | 950.2 ± 99.6 | 4/5 | 19,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | TRPO (SB3) | 5 | 150,523 | 1000.0 ± 0.0 | 5/5 | 29,000 (5/5) |
| InvertedPendulum-v5 (≥ 950) | PPO (SB3) | 5 | 150,220 | 1000.0 ± 0.0 | 5/5 | 28,000 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (abstract_rl) | 5 | 602,473 | 9092.1 ± 218.8 | 2/5 | 211,015 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | MPO (abstract_rl) | 5 | 600,178 | 5937.0 ± 4162.1 | 3/5 | 66,066 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | TRPO (SB3) | 5 | 599,517 | 8270.3 ± 602.2 | 0/5 | 107,946 (5/5) |
| InvertedDoublePendulum-v5 (≥ 9100) | PPO (SB3) | 5 | 599,587 | 9175.8 ± 225.2 | 3/5 | 197,967 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (abstract_rl) | 5 | 1,200,000 | 1903.7 ± 862.3 | 0/5 | never |
| HalfCheetah-v5 (≥ 4800) | MPO (abstract_rl) | 5 | 1,200,000 | 8350.6 ± 1043.4 | 5/5 | 198,000 (5/5) |
| HalfCheetah-v5 (≥ 4800) | TRPO (SB3) | 5 | 1,200,000 | 2547.9 ± 1314.1 | 0/5 | never |
| HalfCheetah-v5 (≥ 4800) | PPO (SB3) | 5 | 1,200,000 | 1620.3 ± 59.9 | 0/5 | never |

![benchmark](docs/benchmark.png)

- **Tuned for Pendulum-v1 and MountainCarContinuous-v0** (one setting each): TRPO Pendulum discount 0.95, MPO
  Pendulum E-step epsilon `rnk_eps` 0.05, MPO MountainCar fixed (not learned) exploration std 2.0. A learned std
  shrinks to ≈ 0 before the sparse goal is found, which made 2/5 seeds fail before. These runs stop at the exact
  step budget (`max_env_steps`).
- **Pendulum balance check** (`scripts/eval_balance.py`, |θ| < 0.1 rad in the last 100 of 200 steps, 20 start
  states): TRPO and MPO balance in 5/5 seeds and 20/20 episodes each (max |θ| 0.012-0.057 rad TRPO, 0.032-0.098 rad MPO).
- **SB3 baselines**: library defaults, except MountainCarContinuous-v0, which uses the
  [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyper-parameters (`--zoo`, gSDE exploration) at
  the same 400k budget; with defaults both SB3 algorithms score 0 there. On Pendulum-v1 the zoo settings were worse
  than the defaults (TRPO -136, PPO -226), so the defaults are shown.
- **InvertedPendulum, InvertedDoublePendulum, HalfCheetah** are not re-tuned. MPO stops after a fixed number of epochs
  and finishes the running episode, so on the two pendulums it ran up to 4-6% past the budget; the learning curves are
  cut at the budget, the greedy return is the one of the final policy. MPO is less stable late in training there
  (one InvertedPendulum seed, two InvertedDoublePendulum seeds collapse after reaching the maximum).
- Runs were made on Linux (Python 3.13) and a MacBook (Python 3.14) with one CPU thread; seeds are fixed but results
  are not bit-reproducible across machines.

## Usage

```bash
uv run python -m abstract_rl.scripts.run env=halfcheetah alg=mpo seed=1 alg.num_epochs=100  # Hydra overrides
uv run python -m abstract_rl.scripts.run env=mountaincar alg=mpo data_root=results -m seed=1,2,3,4,5  # sweep
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo --cfg job              # print the config
uv run python -m abstract_rl.scripts.eval_returns data/HalfCheetah-v5/mpo/<run> --save       # greedy evaluation
uv run python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run>              # Pendulum balance check
uv run python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 ppo --seed 1 --data-root results
uv run python -m abstract_rl.scripts.compare results --out docs/                         # table and plots
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render         # replay a policy
```

Configuration ([Hydra](https://hydra.cc)): `config/config.yaml` (`seed`, `data_root`, `max_env_steps`),
`config/alg/{trpo,mpo}.yaml` (Pendulum-v1 defaults) and `config/env/<env>.yaml` (Gymnasium `id`, `success`
threshold, `steps` budget, `overrides` of the algorithm defaults for `all`/`trpo`/`mpo`). Environments: `pendulum`,
`mountaincar`, `inverted_pendulum`, `inverted_double_pendulum`, `halfcheetah`. `alg.<key>=...` overrides beat the
environment table. The training loops get a flat dictionary with prefixes (`env_*`, `policy_*`, `val_*`, `trpo_*`,
`mpo_*`). Runs are written to `data/<env>/<alg>/<timestamp>/` (configuration, returns, models, plots).
Environments with another action dimension work out of the box: `select_policy` chooses the univariate or the
diagonal Gaussian policy from the action space.

## Epoch time

Wall-clock seconds per epoch on one 4-CPU Linux machine, Hydra defaults, `seed=1`. PyTorch is the previous implementation pinned to one thread. JAX "before" is the port before the loop pass, measured after its compile. JAX "after" skips epoch 0 (the compile). Steady epochs then report 0 compiles from `jax_log_compiles`.

| Run | PyTorch | JAX before | JAX after | Window after compile |
|---|---:|---:|---:|---|
| TRPO Pendulum-v1 | 2.676 | 1.615 | 0.336 | epochs 1–4 |
| MPO Pendulum-v1 | 2.686 | 2.02, climbing to 2.98 over 40 epochs | 1.416 | epochs 1–11 (12k of the 50k buffer) |
| TRPO HalfCheetah-v5 | 3.103 | 4.160 | 1.070 | epochs 1–3 |
| MPO HalfCheetah-v5 | 3.270 | 2.03, climbing to 2.66 over 8 epochs | 1.475 | epochs 1–7 |

Phases over those same windows, in seconds. Each cell is the mean before this pass, then the mean after.

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

Resample scans the whole allocation, so its cost is the full-buffer figure from the first steady epoch. Retrace skips empty episode slots. On Pendulum it grows with the stored episodes (0.021s at epoch 1, 0.094s at epoch 11) while the epoch stays between 1.39s and 1.44s. The slope is about 0.007s of Retrace per thousand new steps. HalfCheetah stores one episode per epoch, and the first eight share one slot chunk, so Retrace stayed flat in that window.

On an earlier seed-1 run, before this pass, the stochastic Pendulum return moved from -1172 to -178 over 40 TRPO epochs and from -1477 to -150 over 40 MPO epochs. That learning curve was not repeated after the loop changes. The solve bar in the results table is a greedy evaluation. The 5-seed Pendulum and MountainCar budgets were not re-run.

### Loops

Each training method keeps its name and its maths. The inner iteration is a JAX loop. Timings are from this machine.

| Loop | File | Became | Measured |
|---|---|---|---|
| Retrace backup `k = 1 .. bootstrap` | `operator/retrace_operator.py` `retrace_accumulate` | `lax.scan` | max abs err 6.7e-16. A 32×1000 backup is 0.0001s unrolled and 0.0006s as a scan (0.21×). The epoch win is the fused slot kernel: Pendulum Retrace 0.207s → 0.021s at the same fill |
| Retrace chunks, host pad, Python loop | `operator/retrace_operator.py` `apply` | one `filter_jit`; `lax.cond` skips empty chunks; device scatter-add; donated `q` and `tq` | included in the Retrace timing above |
| Resample chunks and a `.at.set` per chunk | `operator/resample_actions_operator.py` `apply` | `lax.scan` over a fixed chunk count; donated `add_act`, `add_q`, `add_pi` | flat 0.47s Pendulum / 0.57s HalfCheetah. The previous path started near 0.24s and rose with the buffer |
| Ranking on `add_q[:size]` | `operator/np_rew_operator.py` | `filter_jit` on the full buffer, traced `size` | 0.07–0.10s → under 0.005s. Growing `size` no longer retraces |
| Temperature L-BFGS-B with `float()` each iteration | `algorithms/mixed/non_parametric_trust_region_optimizatioon.py` | `lax.while_loop` damped Newton on the same dual; traced `size` | (20000, 20) float64: 1.031s → 0.052s (19.7×). On a (256, 20) draw the optimum has KL 0.1000 against ε 0.1 |
| TRPO gradient, Fisher, and Python conjugate gradient, with a new jit every epoch | `trust_region_gradient.py`, `conjugate_gradient.py` | one `filter_jit`; `cg_solve` is `lax.while_loop` | Pendulum TRPO step 0.644s → 0.016s (40×); HalfCheetah 1.232s → 0.050s (25×) |
| Line search `while` with `float(tr_obj)` per probe | `trust_region_gradient.py` `line_search` | `_backtrack` `lax.while_loop`; one log copy; one parameter write | 10 probes on a 4000-step batch: 0.0036s → 0.0033s (1.10×) |
| Value fit built a new `value_and_grad` every epoch | `value_functions/fc_v_networks.py` | module-level `_v_value_and_grad`. The driver remains scipy L-BFGS-B | Pendulum 0.247s → 0.081s (3.0×); HalfCheetah 0.362s → 0.151s (2.4×) |
| Critic Adam Python loop | `value_functions/fc_q_networks.py` | `lax.scan`, with donated weights and Adam state | Pendulum critic stays 0.69s |
| M-step Python loop on the default path | `maximum_a_posteriori_policy_optimization_algorithm.py` `_mpo_scan` | `lax.scan`, with donated weights, both Adam states, and the multiplier | Pendulum 0.124s → 0.036s (3.4×); HalfCheetah 0.172s → 0.065s (2.6×) |
| GAE and TD reverse loops | `gae_operator.py`, `td_operator.py` | `lax.scan` | a few hundredths of a second after epoch 0 |
| `policy.act` once per environment step | `continuous_policy.py`, `numpy_act.py` | NumPy matmul; weights cached on `id(params)` | HalfCheetah act 1.449s → 0.136s (11×); Pendulum 0.442s → 0.048s (9×) |
| Gymnasium rollout `while` | `env/mcmc_env_wrapper.py` | stays Python, because `env.step` is not a JAX function | HalfCheetah env step 0.759s → 0.494s |
| Replay `.at[start:end].set` with Python bounds | `replay_buffer.py` | one host pack, then `_insert_transitions` / `_shift_transitions` (`jnp.where`, traced start and length, donated buffers) | device sync stays under 0.01s, and steady MPO epochs compile 0 times |
| Q evaluation over the K sampled actions | `fc_q_networks.py` `q_values` | one matmul with the action axis folded into the batch | counted inside resample and Retrace |

Host `if` statements remain on concrete values: empty buffers, filter flags, and `n_envs == 1`. Traced branches inside the jitted updates are `lax.cond` or `jnp.where`. Scipy still reads the value-fit scalar and gradient on the host once per L-BFGS iteration; optax 0.2.8 has no L-BFGS to put inside `lax.scan`. The single-trajectory resample path still chunks in Python. Training uses the device scan. Equinox takes `donate="all-except-first"` rather than `donate_argnums`.

## Architecture

```
src/abstract_rl/
├── algorithms/
│   ├── continuous/policy_gradient/   TRPO and MPO (policy update rules)
│   └── mixed/                        conjugate gradient, trust region gradient (natural gradient + line search),
│                                     vanilla gradient (L-BFGS / Adam), non-parametric temperature dual
├── data_structures/
│   ├── abstract_conf/                SharedConf (namespaced config), ModelConfiguration (registry, target sync, IO)
│   └── temporal_difference_data/     Trajectory, TrajectoryCollection, on-device ReplayBuffer, Batch
├── env/                              MCMCEnvWrapper: whole episodes, optional Gymnasium vector env for TRPO
├── operator/                         trajectory operators: TD, GAE, Retrace, action resampling, MPO reweighting, filters
├── policy/                           univariate and diagonal Gaussian policy (Equinox), non-parametric policy
├── value_functions/                  fully connected V (L-BFGS) and Q (Adam) networks
├── neural_modules/                   Equinox MLP building block
├── run/                              training loops (run_trpo, run_mpo_non_parametric), policy replay
├── scripts/                          Hydra entry point (run.py, hydra_conf.py), evaluate / baselines / compare
└── misc/ and plots/                  CLI printer, data logger, running filters, plots
```

A training loop is a sequence of operator applications on trajectory collections:

- **TRPO** per epoch: collect `policy_batch_size` steps → `TD` operator (returns, bootstrapped at time limits) →
  `GAE` operator → natural gradient step with KL trust region → V function fit (L-BFGS).
- **MPO** per epoch: collect `env_sample` steps into the trajectory collection → `Resample` operator (actions from the
  target policy, evaluated by the target Q) → temperature dual + `VariationalRanking` operator (non-parametric
  E-step) → KL constrained weighted maximum likelihood (M-step) → `Retrace` operator → Q function fit (Adam).
  The collection keeps an on-device copy, and the default M-step and the critic Adam loop are each one `lax.scan`.

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
