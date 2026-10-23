# JAX port spike S1: stack compatibility and Metal op coverage

Date: 2026-10-22. Machine: Apple M3 Max, macOS 26.7, uv 0.10.4, Python 3.12 (uv venv, throwaway, not in the repo).
Probe: a throwaway script (not committed). Run once with `JAX_PLATFORMS=cpu` and once with `JAX_PLATFORMS=mps`.

## Resolved versions

The first resolution attempt succeeded with no packages removed and no manual pins. `brax` resolves with jax 0.11.

| package | version |
|---|---|
| jax | 0.11.2 |
| jaxlib | 0.11.2 |
| jax-mps | 0.11.0 |
| equinox | 0.13.8 |
| optax | 0.2.8 |
| jaxtyping | 0.3.11 |
| gymnax | 1.0.1 |
| brax | 0.14.2 |
| mujoco | 3.16.0 |
| mujoco-mjx | 3.16.0 |
| playground (import `mujoco_playground`) | 0.2.0 |

Install command: `uv pip install "jax>=0.11,<0.12" "jax-mps>=0.11,<0.12" equinox optax jaxtyping beartype gymnax brax mujoco mujoco-mjx playground`.
Manual pins: none.

### Probe adaptations

Two minimal changes to the probe so that it measures the same thing. Both were needed for the "16-seed MLP adam step" probe, which failed on CPU with
`TypeError: ... custom_jvp ... is not a valid JAX type`. The cause was probe code, not the stack: `eqx.nn.MLP` stores `jax.nn.relu` (a `custom_jvp` object) as a non-array leaf.

1. `eqx.filter_vmap(in_axes=(0, 0, None))` became `in_axes=(eqx.if_array(0), eqx.if_array(0), None)`, so that axis 0 applies only to array leaves.
2. `timed()` now calls `jax.block_until_ready(eqx.filter(fn(*args), eqx.is_array))`, because the returned model also holds non-array leaves.

The port must use the `eqx.if_array` / `eqx.filter` idioms when vmapping over modules.

## Op coverage

| probe | cpu | mps | error text |
|---|---|---|---|
| while_loop | OK | OK | |
| scan reverse | OK | OK | |
| scatter .at[].set | OK | OK | |
| cholesky | OK | OK | |
| solve_triangular batched | OK | OK | |
| logsumexp | OK | OK | |
| random.normal | OK | OK | |
| jvp(grad) hvp | OK | OK | |
| argsort+cumsum | OK | OK | |
| 16-seed MLP adam step | OK | OK | |
| gymnax Pendulum-v1 20x200 | OK | OK | |
| gymnax MountainCarContinuous-v0 20x200 | OK | OK | |
| brax-mjx inverted_pendulum 16x1000 | OK | FAIL | `Unsupported device: MPS:0` (raised when building the env) |
| brax-mjx inverted_double_pendulum 16x1000 | OK | FAIL | `Unsupported device: MPS:0` |
| brax-mjx halfcheetah 16x1000 | OK | FAIL | `Unsupported device: MPS:0` |
| playground AcrobotSwingup 16x1000 | OK | FAIL | `Unknown backend cpu. Available backends are ['mps']` |

The MPS failures for MJX envs occur at environment construction or load, before any op is traced. They are a device or backend-selection limitation (MJX and warp look for a cuda or cpu device),
not a missing op, so MJX op coverage on Metal is untested. The MPS runs print "Platform 'mps' is experimental" and a tip about `JAX_MPS_ASYNC_DISPATCH=1`, which was not enabled.

## Throughput

Seconds for one post-compile call (second invocation), single run, so only order of magnitude is meaningful.

| rollout probe | cpu (s) | mps (s) |
|---|---|---|
| gymnax Pendulum-v1, 20 envs x 200 steps | 0.0002 | 0.0267 |
| gymnax MountainCarContinuous-v0, 20 envs x 200 steps | 0.0002 | 0.0243 |
| brax-mjx inverted_pendulum, 16 x 1000 | 0.2458 | n/a (env construction fails) |
| brax-mjx inverted_double_pendulum, 16 x 1000 | 0.1667 | n/a |
| brax-mjx halfcheetah, 16 x 1000 | 3.6148 | n/a |
| playground AcrobotSwingup, 16 x 1000 | 3.1945 | n/a |
| 16-seed MLP adam step (reference) | 0.0031 | 0.0014 |

For small workloads, MPS is slower than CPU on the gymnax rollouts (about 100x, dominated by per-op dispatch in the scanned loop) and only about 2x faster on the MLP step. MPS gives no throughput benefit at these sizes.

## Decisions

- MPS usable for (a) networks only: **yes**. Every network-side and policy-side op (while_loop, reverse scan, scatter, cholesky, batched solve_triangular, logsumexp, random, hvp, sort/cumsum, vmapped equinox+optax training) works on MPS with jax-mps 0.11.0.
  This includes cholesky, solve_triangular and scatter, so no feature has to be marked unsupported on MPS for these ops. It is not yet faster than CPU for the small probes.
- MPS usable for (b) MJX envs (brax-mjx and playground): **no**. All four fail at env construction with a device or backend error. Run MJX envs with `jax_platform: cpu`.
  Pure-JAX gymnax envs do run on MPS, but slower than CPU at this size.
- Env ids available (CPU, verified): gymnax `Pendulum-v1`, `MountainCarContinuous-v0`; brax (mjx backend) `inverted_pendulum`, `inverted_double_pendulum`, `halfcheetah`; playground `AcrobotSwingup`.
- Pins for `pyproject.toml` in Task 3: `jax>=0.11,<0.12`, with `jaxlib` resolved transitively (0.11.2 at spike time). Extras: `mps` adds `jax-mps>=0.11,<0.12`.
  Core and env dependencies: `equinox>=0.13`, `optax>=0.2.8`, `jaxtyping`, `beartype`, `gymnax>=1.0.1`, `brax>=0.14.2`, `mujoco>=3.16`, `mujoco-mjx>=3.16`, `playground>=0.2.0`. Python `>=3.12` (required by jax-mps).
  `jax>=0.12` is deliberately excluded.
- Because no policy op fails on MPS, the document does not mark `jax_platform: mps` unsupported for cholesky, solve_triangular or scatter features. It does mark it unsupported for MJX-backed envs.
