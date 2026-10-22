# JAX port of abstract_rl: design

Date: 2026-10-22. Branch: `feature/jax-vmap-port`, cut from `origin/feature/tune-on-basic-envs` (8d3dc51).

## 1. Goal

Port the PyTorch TRPO and MPO stack to JAX so that one call trains many seeds at once. The seeds are vmapped, the
environments inside each seed are vmapped, and every update is jitted. The algorithms must behave exactly as before and
the class and module architecture must stay recognisable. Torch, Gymnasium and dm_control leave the training path.

### Success criteria

1. `abstract-rl env=pendulum alg=trpo seeds=[1,2,3,4]` trains four seeds in one jitted, vmapped program and writes one
   run directory per seed in the current layout (`run.conf`, `data/*.txt`, `model/last`, `model/best`, plots).
2. The deterministic pieces match the torch implementation on fixed inputs, checked against golden fixtures recorded
   from the torch code (section 9): TD, GAE, Retrace, the squashed log-probs, the KL divergences, CG, the TRPO step
   with its line search, the temperature dual, the MPO objective and its gradients, and the network forward passes.
3. Wall-clock throughput, in environment steps per second summed over seeds, is reported against the torch baseline on
   the same machine (an M3 Max) for Pendulum TRPO and MPO and HalfCheetah TRPO.
4. Every array argument and return carries a jaxtyping shape annotation. Tests run with `jaxtyped` and beartype
   enabled.
5. Shipped configs keep their meaning. Hyperparameters, defaults and Hydra composition are unchanged, except for the
   new keys listed in section 8.

### Non-goals

- Bit-identical random streams with torch, which is impossible across RNG designs.
- New algorithms, new tuning, or "paper-style" changes to MPO and TRPO.
- Multi-device sharding.

## 1a. Change policy: minimal diff

- **No renames.** Files, classes, methods, attributes, config keys and local variables keep their current names. That
  includes the existing typo `non_parametric_trust_region_optimizatioon.py`.
- **No restructuring unless jittability requires it.** A change is allowed only if jit, vmap or grad cannot work
  without it, in which case it takes the smallest form that does (section 5a, R-items). Anything beyond that is an
  optional improvement and needs explicit approval (section 5a, P-items).
- **Keep the code flow.** A function keeps its signature, adding only explicit inputs and outputs (a PRNG key, the
  state it used to mutate) and jaxtyping annotations. The body is converted line by line where possible: `torch.` to
  `jnp.`, an in-place update to a returned value, a Python loop on traced values to the matching `lax` primitive.
- **Keep comments and docstrings.** Update them only where they describe torch-specific behaviour.

## 2. Decisions taken in brainstorming

| Topic | Decision |
|---|---|
| Base branch | `origin/feature/tune-on-basic-envs`. The old `origin/migration/port-torch-to-jax` is ignored entirely. |
| Framework | **Equinox** (modules), **Optax** (optimisers), **jaxtyping** (shapes). |
| Accelerator | **jax-mps** (Metal through MLX) as the GPU backend, `JAX_PLATFORMS=mps`. CPU stays the reference backend. |
| JAX version | `jax>=0.11,<0.12` everywhere, because jax-mps 0.11 needs jaxlib 0.11.x. jaxlib is never pinned separately. |
| Python | `requires-python = ">=3.12,<3.14"`, `.python-version` = `3.12`. jax-mps needs Python 3.12 or later. |
| Rollouts | **Padded full episodes**. E parallel envs each run one whole episode into `[E, max_steps]` with a validity mask. Rounds repeat until at least `batch_size` valid steps are collected. |
| Environments | Pure JAX only. gymnax covers Pendulum and MountainCarContinuous, Brax with the MJX backend covers InvertedPendulum, InvertedDoublePendulum and HalfCheetah, and MuJoCo Playground covers AcrobotSwingup. Gymnasium and dm_control are dropped. SB3 baselines keep Gymnasium inside the `baselines` extra only. |

## 3. Stack and dependencies

```toml
requires-python = ">=3.12,<3.14"
dependencies = [
    "jax>=0.11,<0.12", "equinox", "optax", "jaxtyping", "beartype",
    "gymnax",                     # Pendulum-v1, MountainCarContinuous-v0
    "brax", "mujoco", "mujoco-mjx", "playground",   # MJX envs (resolved against jax 0.11)
    "hydra-core", "matplotlib", "numpy", "pandas", "pillow",
]
[project.optional-dependencies]
mps = ["jax-mps>=0.11,<0.12"]
baselines = ["stable-baselines3", "sb3-contrib", "gymnasium[classic-control,mujoco]"]
[dependency-groups]
dev = ["pytest", "ruff", "mypy", "pandas-stubs", "torch>=2.5"]   # torch only to record parity fixtures
```

Torch moves to the dev group and is used only by the fixture recorder (section 9). The `dm-control` extra, the
`labmaze` exclusion and the CPU torch index go away. Every brax, mjx and playground version must resolve with jax 0.11.
Spike S1 verifies that before anything else.

## 4. Execution model

```
init:  keys = split(PRNGKey(base_seed_from(seeds)), n_seeds)
       state = vmap(Algorithm.init)(keys)                   # [n_seeds, ...] batched pytree
epoch: state, metrics = jit(vmap(Algorithm.epoch))(state)   # one call per epoch, all seeds
host:  for each seed: logger.log(metrics[s]), plots, store("last"), store_best(...)
```

- **Algorithm state** is one `eqx.Module` per algorithm. `TRPOState` and `MPOState` hold:
  - online and target networks,
  - Optax states,
  - Lagrange multipliers, `eta` and `v`,
  - env state and the padded episode buffer or replay buffer,
  - obs-norm statistics,
  - counters (`overall_step`, `total_steps`, `created_trajectories`),
  - the PRNG key.
- **The epoch loop stays in Python** in `run_trpo.py` and `run_mpo_non_parametric.py`, with the same order of phases
  and `cli.header` blocks.
  - Exactly one jitted call runs per epoch, followed by one device-to-host copy of the metrics.
  - The `max_env_steps` budget check stays on the host. A shortened last epoch passes `batch_size` as a static
    argument, so it triggers at most one recompile.
- **Seed vmap** is the outer axis.
- **Env vmap** is an inner `vmap` over E envs inside the rollout.
  - E comes from config (`env_num_envs`). The default is `ceil(batch_size / max_steps)`, so Pendulum TRPO uses
    4000/200 = 20 envs and finishes in a single round, exactly like today.
- **Host-only objects**, untouched apart from their torch imports:
  - `SharedConf`, `ModelConfiguration` (wiring, run dirs, `store`/`store_best`),
  - `DataLogger`, `CliPrinter`, plots, Hydra.
- **Printing inside loops** becomes returned metrics that the host prints after the call. Examples:
  - the MPO table printed every 10 steps,
  - the CG and line-search lines,
  - the per-trajectory `TrajectoryCollection.print`.

## 5. Module mapping

Same files, same class names. Methods that used to mutate now return new values. Static configuration (`cov_type`,
`squash`, `mode`, layer sizes, activations, `split_*`, `layer_norm`) is stored with `eqx.field(static=True)`, so the
Python `if`s on it stay plain `if`s.

| File | Torch today | JAX form |
|---|---|---|
| `neural_modules/neural_network.py` | `MultilayerPerceptron(nn.Module, Syncable)` | `eqx.Module` holding a list of `eqx.nn.Linear`. Same init options: biases are zero, `xavier_gaussian` is `jax.nn.initializers.glorot_normal` with a non-truncated normal to match `xavier_normal_`, `orthogonal` uses gain √2, plus `gaussian`, `uniform`, `high` and `none`. Per-output activation lists and the first-layer parameter-free LayerNorm (eps 1e-5) are kept. |
| `policy/continuous/{univariate,diagonal}_gaussian_policy.py` | `nn.Module` | `eqx.Module`. `forward`, `log_prob`, `sample_actions(key, ...)`, `mode`, `kl_divergence`, `entropy` and `likelihood_ratio` keep their formulas, constants (`SIGMA_MIN`, `SQUASH_U_MAX = 5`, `SQUASH_BOUND = tanh(5)`) and shapes. `cov_type` ∈ {single, fixed, state, full}. `fixed` and `action_center/scale` are excluded from gradients through `eqx.partition` filters. |
| `policy/continuous/continuous_policy.py`, `policy/policy.py` | abstract bases | abstract `eqx.Module` bases with the same abstract methods |
| `policy/continuous/select_policy.py` | factory | the same factory, taking a key |
| `value_functions/fc_{v,q}_networks.py` | `nn.Module`, `VanillaGradient` | `eqx.Module`. `v_val`, `q_val`, `q_val_multi` (keeps the `repeat_interleave` row order, `i*K + j`), `v_val(key, states, exp_actions=10, policy)` and `tr_obj`. |
| `data_structures/abstract_conf/syncable.py` | `clone` / `sync` with deepcopy | `sync(online, target, step, sync_steps) = jnp.where(step % sync_steps == 0, online, target)`, applied leaf-wise with `jax.tree.map`. `clone` becomes the identity, because pytrees are immutable. |
| `algorithms/mixed/vanilla_gradient.py` | Adam mixin | `perform_mult_sgd(state, key, num_steps, tc, batch_size, fields)` keeps today's name and argument order, with `state` and `key` added (R1, R2, R6). It is a `lax.scan` over `num_steps` Optax Adam updates. Optax defaults `b1=.9, b2=.999, eps=1e-8` equal the PyTorch defaults. |
| `algorithms/mixed/conjugate_gradient.py` | Python `while` | `lax.while_loop`, carrying `(k, x, r, p, rdotr)`, with the same `max_k` and `rdotr <= tol` stop and the same `A(v)` callable interface |
| `algorithms/mixed/trust_region_gradient.py` | `perform_tr_sgd`, `line_search` | `jax.flatten_util.ravel_pytree` replaces `flat_params`. The FVP is a forward-over-reverse HVP (`jax.jvp(jax.grad(kl))`) plus damping. The line search is a `lax.while_loop` over `lr = 1, 1/2, ...` for up to 10 steps, with the same acceptance test, and restores `x` on failure. |
| `algorithms/mixed/non_parametric_trust_region_optimizatioon.py` | 200 fixed Adam steps on `log η` | `lax.fori_loop` with a fresh Adam per call, as today, run in float64 on CPU (section 7) |
| `algorithms/continuous/policy_gradient/*` | algorithm classes | `TRPOAlgorithm.perform_tr_sgd(state, batch)` and `MPOAlgorithm.perform_mult_steps(state, key)`. The MPO step order is kept: max step, then min step on the same batch, then the logging recompute. `shuffle_up_stats`, the split objective and the split constraints, `exp(λ)` multipliers with λ initialised to 1, the persistent Adam states, and `sample_v` drawing from a key are all kept. |
| `operator/*` | per-trajectory Python loops | `transform_all(buffer, ...) -> buffer`, vmapped over episodes and run as a reverse `lax.scan` over time with the episode mask. TD, GAE and Retrace (truncated at B = `mem_bootstrap_steps`) keep their exact recurrences. Resample, ranking (log-softmax of `add_q/η`) and the filter work the same. |
| `data_structures/temporal_difference_data/*` | pandas, lists, numpy | `EpisodeBuffer`, `Batch` and `ReplayBuffer` pytrees (section 6). `Trajectory` and `TrajectoryBuilder` become thin views over one padded row. They are kept for the eval scripts and tests. |
| `env/mcmc_env_wrapper.py` | gym wrapper | `MCMCEnvWrapper` over a `JaxEnv` protocol (section 6.3). `execute_policy` is a jitted vmapped rollout. `execute_policy_once` is the same rollout with E = 1. |
| `misc/running_centered_filter.py`, the obs-norm path | numpy Welford, Gymnasium `NormalizeObservation` | pure update functions over a stats pytree with identical formulas (section 7) |
| `misc/flat_params.py` | flat parameter copy | `ravel_pytree` helpers |
| `env/acrobot_swingup.py` | dm_control adapter | removed. Playground's `AcrobotSwingup` is used instead. |
| `run/*.py`, `scripts/*.py` | torch I/O | the same CLIs. Checkpoints use `eqx.tree_serialise_leaves` per seed. Eval and GIFs run on the JAX envs (section 6.4). |

Dead code is ported as-is or dropped as-is. Nothing is fixed silently. Every item below is noted in the code.
- `PolicyOptimizationAlgorithm.obj` (unused) is dropped.
- The unused `ret_add_acts` and `num_add_actions` stay as config keys.
- `rnk_max_steps` is still not passed, so the dual keeps its default of 200 steps.
- The parametric MPO mode is kept and still needs `"a"`.

## 5a. Architecture review: JAX compatibility

### Required changes (R): jit, vmap or grad cannot work without these

Each one is shown in its smallest form.

| ID | Current pattern | Why it breaks under jit/vmap | Minimal change, names kept |
|---|---|---|---|
| R1 | `self.mc.get("policy", target=True)` is called inside traced methods (algorithms, operators, Q `v_val`). `mc` is a mutable host dict. | The batched per-seed models must come in as traced arguments. A host dict holds one unbatched object. | `mc` keeps its host role: wiring, run directories, store. The traced code receives a state pytree that has the **same accessor**, `state.get("policy", target=True)`, with the same keys (`policy`, `v_network`, `q_network`, `tc`, ...). So a call site changes only from `self.mc.get(...)` to `state.get(...)`. |
| R2 | `VanillaGradient` and the MPO algorithm own `torch.optim.Adam` objects, holding parameters and state, inside the module or algorithm. | Optimiser state has to be a separate pytree, threaded in and out. It cannot be a field of the module being differentiated. | Same method names (`perform_mult_sgd`, `perform_adam_step`). `opt_state` moves into the algorithm state under keys named after today's attributes (`opt_outer`, `opt_inner`, `v_network.opt`, ...). |
| R3 | `Syncable.clone()` uses `deepcopy`. `mc.sync()` mutates targets with `overall_step % sync_steps`. | It mutates in place, and the branch depends on a traced counter. | Target models become separate leaves in the state. `sync` is a `jnp.where(overall_step % sync_steps == 0, online, target)` tree-map. `overall_step` moves into the state. |
| R4 | `Trajectory` and `TrajectoryCollection` are Python lists of pandas and numpy objects with ragged lengths. `Batch` is numpy. | Static shapes, and pytrees only. | These are padded pytrees (section 6) with the **same attribute and property names**: `states`, `next_states`, `all_states`, `actions`, `rewards`, `dones`, `log_likelihoods`, `traj["field"]`, `batch[field]`, `to_batch`, `sample`, `merge`. They gain a `mask`. |
| R5 | `transform(trajectory)` mutates its annotations, and `transform_all` loops over a list. | In-place writes, and a Python loop over traced data. | `transform(trajectory) -> trajectory` stays per episode, with the same body converted to a reverse `lax.scan`. `TrajectoryOperator.transform_all` becomes `vmap(self.transform)` over the padded episode axis. This is the one base-class edit, and every operator subclass is otherwise untouched. |
| R6 | Global `torch.manual_seed` and `np.random`. | vmap over seeds needs a separate random stream per seed. | Stochastic functions (`sample_actions`, `sample`, `sample_v`, rollout, `v_val`) take an extra leading `key` argument. Nothing else changes. |
| R7 | Python `while` or `if` on tensor values: the CG break, the line search, the dual loop, `step % 10` logging. | Traced values can't drive Python control flow. | The same function, rewritten in place with `lax.while_loop` or `lax.fori_loop`. The loop-variable names stay. |
| R8 | `RunningCenteredFilter` and obs-norm update numpy state on every env step. `MCMCEnvWrapper` counters (`total_steps`, `epoch`, `best_reward`, `max_traj_len`, `created_trajectories`) are Python attributes. | Host mutation inside the rollout | Same class names and fields. The stats become pytree leaves carried through the rollout scan, and the counters live in the state. The thread lock is dropped, because nothing is concurrent any more. |
| R9 | `cli.print`, `DataLogger.log`, `TrajectoryCollection.print` and the CG and line-search prints run inside the update loops. | Host I/O in traced code would run once at trace time. | Traced code returns a `metrics` dict, and the host prints the same lines after the jitted call. The print functions keep their format strings. |
| R10 | `TrustRegionGradient.__init__(policy, ...)` and `FullyVNeuralNet(..., VanillaGradient)` mix stateful mixin `__init__`s into modules through multiple inheritance. | Equinox modules are frozen dataclasses, and mixins must not add mutable fields. | The mixins keep their class names and methods. Their hyperparameters (`tr_eps`, `cg_max_k`, ...) become static fields. The live objects (policy, optimiser) come through the state (R1, R2). |

Not required and therefore unchanged:
- reading `SharedConf` inside methods, because conf values are Python constants at trace time;
- `isinstance` and static config branches such as `split_objective`, `cov_type` and `mode`;
- optional `states=None` or `suff_stats=None` arguments, which are static at trace time;
- the string-based activation selection;
- the `cli.header` structure of the run loops;
- recomputing target statistics several times in MPO, which XLA eliminates as common subexpressions.

### Optional improvements (P)

**Accepted on 2026-10-22: all of P1–P6.**
- P2 defaults to `epochs_per_call: 1`.
- P4 changes only the printed M-step values: they are now the ones before the min step. The training math is unchanged.
- P5 writes checkpoints from a background thread, and the writer is joined before the run ends.

| ID | Proposal | Benefit | Logic impact |
|---|---|---|---|
| P1 | `donate_argnums` on the state in the jitted epoch | Avoids copying the MPO replay buffer (about 50k×20 action samples per seed) every epoch, roughly halving peak memory. | none |
| P2 | `epochs_per_call` (default 1): `lax.scan` over k epochs per jitted call, then log and checkpoint per epoch from the stacked metrics. | Fewer host round-trips, which matters for short Pendulum epochs on MPS. | none for metrics. `store("last")` happens every k epochs. |
| P3 | A shared `misc/typing.py` with jaxtyping aliases (`Obs = Float[Array, "*batch obs"]`, `Act`, `SuffStats`, `Key`) and `DTYPE = jnp.float32` | Consistent shapes across about 40 signatures, and one place to change dtype | none, one new file |
| P4 | Drop the logging-only third recompute in `MPOAlgorithm.perform_step`, and log the values the min step already computed. | About 30% less M-step compute | The printed values become pre-min-step instead of post-min-step. The training math is unchanged. |
| P5 | Make `ModelConfiguration.store` and `store_best` asynchronous: hand the host copy to a writer thread. | The epoch loop doesn't block on disk I/O when there are many seeds. | none |
| P6 | A `vmap` over seeds through one generic `run_seeds(run_fn)` helper shared by `run_trpo` and `run_mpo_non_parametric`, instead of duplicating the batching code in both | Less duplicated code in the two run loops | none, one new helper |

Explicitly **not** proposed, because each would change logic or go against the minimal-diff policy:
- fixing the rollout's filtered-versus-unfiltered state quirk;
- passing `rnk_max_steps` to the dual;
- deleting unused config keys;
- replacing the `ModelConfiguration` service locator with explicit wiring.

## 6. Data and environments

### 6.1 Padded episode buffer (one epoch of rollouts)

Shapes are `[R*E, T]`, with T = `max_steps`, R = rounds and E = parallel envs.

```
states      Float[Array, "eps T+1 obs"]   # includes the final next-state for bootstrapping
actions     Float[Array, "eps T act"]
log_likelihoods, rewards, raw_rewards, dones  Float[Array, "eps T"]   # dones = terminated only
mask        Bool[Array, "eps T"]          # step valid (before episode end)
length      Int[Array, "eps"]
valid_eps   Bool[Array, "eps"]            # this round was actually needed
annotations dict[str, Array]              # "tv", "td", "a", "add_act", "add_q", "grad_ranking", "tq", ...
```

- **Rounds.** A `lax.while_loop` runs rounds of E episodes until the collected valid steps reach `batch_size` or
  `R_max = ceil(batch_size / E)` rounds have run. `R_max` is the static upper bound and covers 1-step episodes.
  - Rounds that aren't needed are masked out, but they still cost compute.
  - Every env in a round finishes its episode, so the episode count is rounded up to a multiple of E. This is the only
    rollout deviation. With E = 1 the rollout is identical to today's.
- **Episode end.** Each env steps until `terminated` or until `max_steps` is reached. Steps after the end are masked,
  and the env state is frozen with `jnp.where`.
  - A `max_steps` cut and an env time-limit truncation are both stored as not done. TD and Retrace therefore bootstrap
    from `states[:, length]` exactly as today.
- **Reset seeding.** Reset keys come from the state's PRNG key, which keeps successive resets independent.
- **Logging.**
  - The logged return is the mean and std of the raw episodic returns over valid episodes.
  - `steps` counts the valid transitions.
  - `best_reward`, `max_traj_len` and `total_steps` are carried in the state.
- **Batch building.** `to_batch(fields)` flattens to `[eps*T]` and carries `mask` along. Every mean and std then uses
  the mask, so the TRPO advantage normalisation and surrogate means equal the unpadded torch values.

### 6.2 Replay buffer (MPO)

- **Layout.** The buffer is a ring of `C` episode slots with the layout from 6.1, plus a per-slot `insert_id`.
  - `C = ceil(tc_size / min_episode_len_bound)` is static, with a configurable bound (`tc_max_episodes`). The default
    is `ceil(tc_size / max_steps) + R_max*E`, which suits envs whose episodes are not much shorter than `max_steps`.
  - The design checks for overflow and raises on the host instead of silently dropping data.
- **Capping keeps the exact rule.** New episodes are inserted, then the oldest are dropped until the kept transitions
  total `tc_size` or fewer, and the newest episode is always kept.
  - Implementation: sort slots by `insert_id` descending, take a cumulative sum of lengths, and invalidate slots whose
    cumsum exceeds `tc_size`, except the newest.
- **Sampling** matches `TrajectoryCollection.sample`: uniform over valid transitions, with replacement.
  - Implementation: `jax.random.choice` over flat indices with probability `mask / mask.sum()`.
  - Sorting the indices is skipped because order does not affect the means.
- **Operators over the whole memory** (resample, ranking, Retrace) vmap over all slots and mask invalid ones.
- **The temperature dual.** It sees `add_q` of the valid transitions only, through a masked mean in the dual, with
  `K = res_add_acts`.

### 6.3 `JaxEnv` protocol and adapters

```python
class JaxEnv(Protocol):
    observation_dim: int; action_dim: int
    action_low: Float[Array, "act"]; action_high: Float[Array, "act"]; obs_high: Float[Array, "obs"]
    def reset(self, key: PRNGKeyArray) -> tuple[EnvState, Float[Array, "obs"]]: ...
    def step(self, state: EnvState, action: Float[Array, "act"]
             ) -> tuple[EnvState, Float[Array, "obs"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]: ...
             # (state, obs, reward, terminated, truncated)
```

Each adapter maps a backend onto Gymnasium's terminated and truncated semantics.

| Config `env_name` | Backend | Notes |
|---|---|---|
| `Pendulum-v1` | gymnax `Pendulum-v1`, `step_env` without auto-reset | gymnax reports its 200-step limit as `done`. The adapter maps it to `truncated`. Pendulum never terminates. |
| `MountainCarContinuous-v0` | gymnax `MountainCarContinuous-v0` | `terminated` = goal reached, computed from the state. The time limit is `truncated`. |
| `InvertedPendulum-v5` | Brax `inverted_pendulum`, `backend="mjx"` | Brax `done` is termination. The time limit is handled by `max_steps`. |
| `InvertedDoublePendulum-v5` | Brax `inverted_double_pendulum`, MJX | same as above |
| `HalfCheetah-v5` | Brax `halfcheetah`, MJX | never terminates |
| `AcrobotSwingup-v0` | Playground `AcrobotSwingup` (MJX) | 1000 steps, reward in [0, 1] |
| `TestPointMass-v0` (tests) | pure-JAX reimplementation of the conftest env | A = 2, asymmetric bounds, 30-step truncation |

The `env_name` keys stay the same, so configs and run directories don't change.
- The Brax and gymnax envs are faithful ports, but not byte-identical to Gymnasium v5. Observation layout, reset noise
  and reward constants can differ slightly.
- Each adapter therefore gets a test that pins its observation and action dims, bounds, reward range and done
  semantics.
- The README `success` thresholds are re-validated in the final benchmark phase.

### 6.4 Eval, replay and GIFs without Gymnasium

- `eval_balance`, `eval_returns` and `run_policy` load a checkpoint and roll out the policy mode on the JAX env.
  Reset keys follow the old `10000 + i` seeds as `PRNGKey(10000 + i)`.
- Rendering:
  - MuJoCo envs use `mujoco.Renderer` on `mjx.get_data(...)`.
  - gymnax envs use gymnax's matplotlib visualiser.

## 7. Numerics and dtypes

- **float32 everywhere.** This matches torch, where `torch.Tensor(np.float64)` becomes float32.
  - Env observations are float32 too (today they are float64 until the policy sees them). The difference is below the
    policy's float32 resolution and is listed in section 10.
- **The temperature dual is float64 on CPU.** The process enables `jax_enable_x64`, every array dtype is explicit
  float32, and only the dual casts `q` to float64, as today.
  - On `mps` the dual runs in float32, because Metal has no float64. This is deviation D4, logged once at startup.
- **Squash.**
  - `y = clip((a - center)/scale, ±tanh(5))`, `u = atanh(y)`, `log_jac = log(scale·(1 - y²))`.
  - Sampling clamps `u` to ±5, and `mode = center + scale·tanh(mu)`.
  - The univariate policy uses `center = 0` and `scale = max|high|`.
  - The full-covariance policy keeps a packed row-major lower triangle, `softplus + SIGMA_MIN` on the diagonal, and the
    `policy_sigma·I` start.
- **KL.**
  - `m-projection` = KL(target‖current) and `i-projection` = KL(current‖target), averaged with `est="mean"` over the
    valid states. The target is wrapped in `stop_gradient`.
  - The full-covariance KL uses `jax.scipy.linalg.solve_triangular`.
- **L2 regularisation.**
  - V: `l2 * ||w||₂` (not squared) over all V params, with gradient.
  - MPO: `l2_reg * ||w||` under `stop_gradient`, because torch's `.data` detaches it.
- **`RunningCenteredFilter`.** Welford, with the n = 1 and `var = mean²` edge cases, then `clip((x-μ)/(σ+1e-8), ±5)`.
  It is fed every preprocessed obs and every transformed reward. The rollout-sees-unfiltered-states quirk is kept as-is.
- **`obs_norm`** reproduces Gymnasium `NormalizeObservation` and `RunningMeanStd`:
  - stats start at mean 0, var 1 and count 1e-4, with parallel-variance updates on every obs including resets;
  - output is `(obs - mean)/sqrt(var + 1e-8)`;
  - eval uses frozen stats;
  - `obs_rms.npz` is stored per seed.
- **Env preprocessing.** `obs / where(isfinite(high) & high > 0, high, 1)` when `env_normalize`, and the affine training
  reward `reward_scale·r + reward_offset`. `raw_return` is computed from the raw rewards.

## 8. Config, CLI and I/O

- **New keys:**
  - `seeds` (a list, default `[seed]`),
  - `env_num_envs` (default `auto`),
  - `tc_max_episodes` (default `auto`),
  - `jax_platform` (`cpu` or `mps`, default `cpu`).
- `seed` keeps working as a single seed.
- **Run directories.** Each seed gets its own directory, as today. `run.conf` stores the real seed.
  - This fixes `ModelConfiguration.store`, which overwrites `seed` with `np.random.randint(1e6)` (deviation D3). The
    overwrite also consumed the RNG and broke `compare.py`'s keying by seed.
- **Checkpoints.**
  - Per seed: `model/{last,best}/{policy,v_network|q_network}.eqx`, written with `eqx.tree_serialise_leaves` from the
    unbatched slice of the seed.
  - `store("last")` keeps its current position, after the rollout and before the update, so stored models lag one
    update exactly as today.
- **Logs.** `data/sreward.txt`, `data/steps.txt` and `data/eta.txt` keep their formats.
- **Hydra.** Composition (`hydra_conf.py`, `config/tune`, `config/tune_all`) is unchanged.
  - `--multirun` over `seed` still works.
  - `seeds=[...]` is the fast path.

## 9. Testing and parity

1. **Golden fixtures.** These are recorded before torch is removed.
   - `tests/fixtures/record_torch_golden.py` runs on the base commit's torch code with fixed inputs and saves
     `tests/fixtures/torch_golden.npz`.
   - The recorded cases:
     - MLP forward for each init and activation, with weights exported and re-imported into Equinox;
     - log_prob, sample-free KL, mode and entropy for univariate, diagonal and full covariance, squashed and unsquashed;
     - TD, GAE and Retrace on hand-built trajectories, including truncation and termination;
     - CG on SPD systems;
     - one full TRPO `perform_tr_sgd` from fixed params and a fixed batch, giving the new params and accept flag;
     - the temperature dual;
     - one MPO `perform_step` with fixed weights and a fixed batch;
     - `RunningCenteredFilter` and the obs-norm sequences.
   - The fixture file and the recorder are committed. Torch stays only as a dev dependency for re-recording.
2. **Parity tests** load the fixtures and compare at rtol 1e-5 on float32. The dual is compared at 1e-10 with x64.
3. **Ported unit tests.** These are rewritten with keys and functional updates; their assertions stay the same:
   - `test_policies`, `test_diagonal_policy` (density integrates to 1, change of variables, Monte-Carlo KL,
     1-D diagonal equals univariate), `test_conjugate_gradient`, `test_operators`, `test_obs_norm`, `test_config`,
     `test_hydra_config`, `test_imports`;
   - `test_env_wrapper`, now checking that only `terminated` is stored as done, on the padded rollout.
4. **New tests:**
   - padded rollout semantics: rounds, masks, a ≥ batch_size guarantee, and E = 1 equivalence;
   - replay capping against the old list-based rule on random episode lengths;
   - one env-contract test per adapter;
   - vmap over seeds gives the same result as running each seed alone (`vmap(epoch)(stack(s)) == stack(epoch(s_i))`);
   - jit stability, so that a second epoch triggers no recompile (checked with `jax.log_compiles`).
5. **Smoke tests.** 2 seeds and 2 epochs of TRPO and MPO on Pendulum and PointMass. The returns have shape
   `[seeds, epochs, 2]` and checkpoints exist.
6. **Removed tests.** `test_acrobot_swingup` (dm_control) is replaced by the Playground adapter test.
   `test_sb3_td3` stays under the `baselines` marker.
7. **Benchmark script.** `scripts/bench_jax.py` reports env-steps/s for torch on the base commit against JAX on CPU and
   MPS, for seed counts of 1, 4 and 16.

## 10. Deliberate deviations (complete list)

| ID | Deviation | Why |
|---|---|---|
| D1 | Random streams differ from torch and numpy. Every draw uses explicit keys. | Different RNG designs, and vmap needs per-seed keys. |
| D2 | Episodes per epoch are rounded up to a multiple of E. E = 1 is exact. | Static shapes for the vmapped envs |
| D3 | `run.conf` stores the real seed, not `np.random.randint(1e6)`. | The overwrite was a bug that broke `compare.py`. |
| D4 | On `mps` the temperature dual runs in float32. CPU keeps float64. | Metal has no float64. |
| D5 | Env dynamics come from gymnax, Brax-MJX and Playground ports instead of Gymnasium v5 and dm_control. | Vectorisable, jittable envs |
| D6 | Observations are float32 from the env on. | Metal has no float64, and the policy already consumes float32. |

Anything else that changes behaviour counts as a bug against this spec.

## 11. Phases

| Phase | Content | Exit check |
|---|---|---|
| **S1 spike, throwaway** | Resolve jax 0.11 with equinox, optax, gymnax, brax, mjx, playground and jax-mps on Python 3.12. Run an MJX HalfCheetah step, `while_loop`, scatter, Cholesky and `solve_triangular` under `JAX_PLATFORMS=mps`. Time vmapped rollouts on CPU and MPS. | A go or no-go per backend and per env, written down. MPS falls back to CPU-only if ops are missing. |
| P0 | Golden fixture recorder, run on torch | `torch_golden.npz` committed |
| P1 | Dependencies and Python pin, jaxtyping and beartype test hook, `ravel_pytree` helpers, CG | `test_conjugate_gradient` passes, parity holds |
| P2 | MLP and the univariate, diagonal and full-covariance policies; V and Q nets; Syncable | Policy and net parity, ported policy tests pass |
| P3 | `JaxEnv` protocol, the adapters, `MCMCEnvWrapper` padded rollout, obs preprocessing and obs-norm | Env contract and rollout tests pass |
| P4 | TD, GAE, `TrustRegionGradient`, TRPO, V fit, and the `run_trpo` epoch as jit plus vmap | TRPO step parity, Pendulum TRPO multi-seed smoke run |
| P5 | Replay buffer, the resample, ranking and Retrace operators, the temperature dual, MPO, Q fit, and the `run_mpo` epoch | MPO step parity, Pendulum MPO multi-seed smoke run |
| P6 | Checkpoints, `store`/`store_best`, eval, replay, GIF scripts, `compare` and `plot` compatibility | Eval scripts run on JAX checkpoints |
| P7 | Remove torch from `src`, the README and benchmark run, re-validate the success thresholds | `test_imports` passes torch-free, benchmark table in the README |

## 12. Risks

- **jax-mps op coverage.** MJX relies on `while_loop`, scatter and gather, and linear solves. S1 decides. The fallback
  is CPU-only with no code change, selected through `jax_platform`.
- **Ecosystem pins.** brax, mjx and playground releases may require jax 0.12 or later. The fallback is the newest
  releases that still accept jax 0.11, found and recorded in S1.
- **Padded compute waste.** On envs that terminate early (the inverted pendulums early in training), most padded steps
  are masked out. That is acceptable, because throughput still dominates torch, and E is configurable.
- **Replay capacity.** Very short episodes could overflow the static slot count `C`. The design raises on the host and
  is configurable through `tc_max_episodes`.
- **Compile time.** The whole epoch is one program, which means a long first compile on MJX. Python's own `max_steps`
  loop is removed, and `lax.scan` keeps the program size independent of T.
