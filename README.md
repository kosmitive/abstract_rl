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

The package installs from git. Hydra configs are inside the wheel, so training does not need a checkout.

```bash
uv add git+https://github.com/kosmitive/abstract_rl
# extras: uv add "abstract-rl[mujoco] @ git+https://github.com/kosmitive/abstract_rl"
#         uv add "abstract-rl[baselines] @ git+https://github.com/kosmitive/abstract_rl"
#         uv add "abstract-rl[dev] @ git+https://github.com/kosmitive/abstract_rl"
```

From a checkout, `uv sync` (add `--extra mujoco`, `--extra baselines` or `--group dev`) installs the same tree.
The console script is `abstract-rl`:

```bash
abstract-rl env=pendulum alg=trpo seed=1
uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seed=1   # equivalent, from a checkout
```

```python
from abstract_rl import make_config, run_trpo

conf = make_config(
    "pendulum",
    "trpo",
    seed=1,
    overrides=["alg.num_epochs=1", "alg.policy_batch_size=400", "data_root=data"],
)
returns = run_trpo(conf)  # shape (epochs, 2): mean and std of the raw episodic return
```

JAX is installed from the CPU extra (`jax[cpu]`). A machine with CUDA can still run on the GPU; `jax-metal` is
not used. The optional `baselines` extra installs Stable-Baselines3, and with it a CPU PyTorch wheel. That
dependency stays out of the core package. `mujoco` adds the Gymnasium MuJoCo environments. `dev` adds pytest,
ruff and mypy.

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

The same greedy protocol on the JAX port (20 episodes, start seeds 10000 + episode, policy mean, raw Gymnasium return). Std is over the five seed means.

| Environment | Algorithm | Per-seed greedy return | Mean ± std | Solved | Env steps |
|---|---|---|---|---|---|
| Pendulum-v1 (≥ -150) | TRPO | -124.9, -124.9, -126.9, -132.8, -132.8 | -128.5 ± 4.0 | 5/5 | 800,000 |
| Pendulum-v1 (≥ -150) | MPO | -123.9, -126.5, -123.1, -124.5, -122.3 | -124.1 ± 1.6 | 5/5 | 1,280,000 |
| MountainCarContinuous-v0 (≥ 90) | TRPO | 95.0, 95.1, 94.5, 94.0, 95.5 | 94.8 ± 0.6 | 5/5 | 425k–431k |
| MountainCarContinuous-v0 (≥ 90) | MPO | 93.5, 93.6, 93.5, 93.7, 93.6 | 93.6 ± 0.1 | 5/5 | 400,009–400,059 |

Against the PyTorch table, Pendulum TRPO is 0.8 points lower and Pendulum MPO 0.3 points lower, both 5/5. MountainCar TRPO is 0.7 points higher and MountainCar MPO matches 93.6, both 5/5. Inside a Pendulum seed the 20 starts are bimodal (some near 0, some near -250), so the within-seed std is about 85–93; the mean still clears -150.

MPO on Pendulum has no `max_env_steps`. Each epoch asks for 1000 transitions and the 4-wide JAX round finishes 8 episodes of 200, so 800 epochs stored 1,280,000 steps rather than 800,000. MountainCar TRPO is 100 epochs of `policy_batch_size` 4000 and finishes the last round, which lands between 424,895 and 431,280. MountainCar MPO stops on `max_env_steps` 400,000.

A short Gymnasium run, one seed, not the full budget: InvertedPendulum TRPO 20 epochs (51,169 steps) stochastic return 3.6 → 756, greedy 1000; MPO 30 epochs (31,060 steps) 6.6 → 1000, greedy 1000. InvertedDoublePendulum TRPO 15 epochs (60,408 steps) 34 → 91, greedy 162; MPO 20 epochs (20,217 steps) 74 → 356, greedy 573. HalfCheetah TRPO 8 epochs (88,000 steps) -313 → -232, greedy 337; MPO 10 epochs (10,000 steps) -524 → -164, greedy -114. The return moves up. Only the short InvertedPendulum runs clear their threshold.

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
abstract-rl env=halfcheetah alg=mpo seed=1 alg.num_epochs=100          # Hydra overrides
abstract-rl env=mountaincar alg=mpo data_root=results -m seed=1,2,3,4,5  # sweep
abstract-rl env=pendulum alg=trpo --cfg job                            # print the config
abstract-rl-eval data/HalfCheetah-v5/mpo/<run> --save                  # greedy evaluation
python -m abstract_rl.scripts.eval_balance data/Pendulum-v1/trpo/<run> # Pendulum balance check
python -m abstract_rl.scripts.run_baselines HalfCheetah-v5 ppo --seed 1 --data-root results
abstract-rl-compare results --out docs/                                # table and plots
python -m abstract_rl.run.run_policy data/Pendulum-v1/trpo/<run> --render  # replay a policy
```

```python
from abstract_rl import make_config, run_mpo_non_parametric, run_trpo

# The YAML defaults are the benchmark budgets. These overrides are a one-epoch smoke run.
run_trpo(make_config("pendulum", "trpo", seed=1, overrides=["alg.num_epochs=1", "alg.policy_batch_size=400"]))
run_mpo_non_parametric(
    make_config("mountaincar", "mpo", seed=1, overrides=["alg.num_epochs=1", "alg.env_sample=200"])
)
```

Configuration ([Hydra](https://hydra.cc)): `config/config.yaml` (`seed`, `data_root`, `max_env_steps`),
`config/alg/{trpo,mpo}.yaml` (Pendulum-v1 defaults) and `config/env/<env>.yaml` (Gymnasium `id`, `success`
threshold, `steps` budget, `overrides` of the algorithm defaults for `all`/`trpo`/`mpo`). Environments: `pendulum`,
`mountaincar`, `inverted_pendulum`, `inverted_double_pendulum`, `halfcheetah`. `alg.<key>=...` overrides beat the
environment table. The training loops get a flat dictionary with prefixes (`env_*`, `policy_*`, `val_*`, `trpo_*`,
`mpo_*`). Runs are written to `data/<env>/<alg>/<timestamp>/` (configuration, returns, models, plots).
Environments with another action dimension work out of the box: `select_policy` chooses the univariate or the
diagonal Gaussian policy from the action space.

## Training environments

| Environment | Training rollout | Evaluation |
|---|---|---|
| Pendulum-v1 | JAX step in `env/jax_classic.py`, `lax.scan` over time and `vmap` over parallel episodes | Gymnasium |
| MountainCarContinuous-v0 | same JAX step | Gymnasium |
| InvertedPendulum-v5 | Gymnasium (MuJoCo) | Gymnasium |
| InvertedDoublePendulum-v5 | Gymnasium (MuJoCo) | Gymnasium |
| HalfCheetah-v5 | Gymnasium (MuJoCo) | Gymnasium |

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

Wall-clock seconds per epoch on one 4-CPU Linux machine, Hydra defaults, `seed=1`. PyTorch is the previous implementation pinned to one thread. "JAX loops" is the port after the scan pass and before this round. "JAX now" skips epoch 0 (the compile). The later epochs are flat, so the compile sits in epoch 0.

| Run | PyTorch | JAX loops | JAX now | Window after compile |
|---|---:|---:|---:|---|
| TRPO Pendulum-v1 | 2.676 | 0.336 | 0.123 | epochs 1–4 |
| MPO Pendulum-v1 | 2.686 | 1.416 | 1.361 | epochs 1–7 (8k of the 50k buffer) |
| TRPO HalfCheetah-v5 | 3.103 | 1.070 | 0.953 | epochs 1–3 |
| MPO HalfCheetah-v5 | 3.270 | 1.475 | 1.489 | epochs 1–5 |

On the same windows, the TRPO Pendulum rollout is 0.041s, the trust-region step 0.013s and the value fit 0.026s. TRPO HalfCheetah rollout stays on Gymnasium at 0.801s, the trust-region step is 0.049s and the value fit 0.059s. MPO Pendulum rollout is 0.018s. MPO HalfCheetah rollout is 0.084s. The MPO epoch is still the resample, Retrace and critic.

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

Resample scans the whole allocation, so its cost is the full-buffer figure from the first steady epoch. Retrace skips empty episode slots. On Pendulum it grows with the stored episodes (0.021s at epoch 1, 0.094s at epoch 11) while the epoch stays between 1.39s and 1.44s. The slope is about 0.007s of Retrace per thousand new steps. HalfCheetah stores one episode per epoch, and the first eight share one slot chunk, so Retrace stayed flat in that window.

The 5-seed greedy evaluation after this round is in [Results](#results). On an earlier seed-1 run, before the loop pass, the stochastic Pendulum return moved from -1172 to -178 over 40 TRPO epochs and from -1477 to -150 over 40 MPO epochs.

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
| Value fit, SciPy L-BFGS-B, host `float()` each iteration | `value_functions/fc_v_networks.py` `perform_sgd` | `optax.lbfgs` zoom line search inside one `lax.scan` (`_fit_v`) | Same 25 steps on a 64-64 value net: Pendulum-sized batch 0.067s → 0.022s, loss 1.02018 → 1.02031; HalfCheetah-sized 0.138s → 0.048s, loss 0.95017 → 0.94989. In the epoch: 0.081s → 0.026s and 0.151s → 0.059s |
| Critic Adam Python loop | `value_functions/fc_q_networks.py` | `lax.scan`, with donated weights and Adam state | Pendulum critic stays 0.69s |
| M-step Python loop on the default path | `maximum_a_posteriori_policy_optimization_algorithm.py` `_mpo_scan` | `lax.scan`, with donated weights, both Adam states, and the multiplier | Pendulum 0.124s → 0.036s (3.4×); HalfCheetah 0.172s → 0.065s (2.6×) |
| GAE and TD reverse loops | `gae_operator.py`, `td_operator.py` | `lax.scan` | a few hundredths of a second after epoch 0 |
| `policy.act` once per environment step | `continuous_policy.py`, `numpy_act.py` | NumPy matmul; weights cached on `id(params)` | HalfCheetah act 1.449s → 0.136s (11×); Pendulum 0.442s → 0.048s (9×) |
| Gymnasium rollout `while` | `env/mcmc_env_wrapper.py` | JAX `lax.scan` + `vmap` for Pendulum and MountainCarContinuous (`env/jax_classic.py`). MuJoCo stays Python | Pendulum rollout 0.193s → 0.041s. HalfCheetah stays Gymnasium, 0.801s |
| Replay `.at[start:end].set` with Python bounds | `replay_buffer.py` | one host pack, then `_insert_transitions` / `_shift_transitions` (`jnp.where`, traced start and length, donated buffers) | device sync stays under 0.01s, and steady MPO epochs compile 0 times |
| Q evaluation over the K sampled actions | `fc_q_networks.py` `q_values` | one matmul with the action axis folded into the batch | counted inside resample and Retrace |

Host `if` statements remain on concrete values: empty buffers, filter flags, `n_envs == 1`, and which environment has a JAX step. Traced branches inside the jitted updates are `lax.cond` or `jnp.where`. The value fit no longer calls SciPy. `jax` and `jaxlib` still install SciPy themselves; the library does not import it and does not list it. Optax zoom L-BFGS was faster than Optax backtracking and than optimistix BFGS/L-BFGS on the value loss, and it matches the SciPy loss. jaxopt L-BFGS is comparable and is not a dependency. The single-trajectory resample path still chunks in Python. Training uses the device scan. Equinox takes `donate="all-except-first"` rather than `donate_argnums`. MuJoCo acting is still the NumPy policy; Pendulum and MountainCarContinuous sample with `gaussian_math` inside the scan.

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
├── env/                              MCMCEnvWrapper; JAX steps for Pendulum and MountainCar, Gymnasium otherwise
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
