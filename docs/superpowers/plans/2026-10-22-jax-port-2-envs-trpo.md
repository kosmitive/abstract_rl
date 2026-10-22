# JAX Port, Plan 2: Environments, Rollouts and TRPO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train TRPO end to end in JAX. Many seeds run as one jitted, vmapped program per epoch, on pure-JAX
environments (gymnax, Brax-MJX, Playground). Logs, plots, run directories and checkpoints are written per seed. MPO is
out of scope: it stays torch and stays skipped.

**Architecture:**
- Environments sit behind a small `JaxEnv` adapter protocol.
- `MCMCEnvWrapper.execute_policy` is a `lax.while_loop` rollout over E lockstep lanes. Each lane collects whole
  episodes back to back into a padded `TrajectoryCollection` (`[E, L]`, with episode-end flags and a validity mask).
- The operators keep a per-lane `transform` and gain `transform_all = vmap(transform)`.
- `TrustRegionGradient` becomes CG with a jvp-of-grad FVP and a `lax.while_loop` line search.
- `run_trpo` keeps its host epoch loop around one `jit(vmap(epoch))` call. `run/run_seeds.py` handles per-seed run
  directories, logging and asynchronous checkpointing.

**Tech Stack:** Python 3.12, jax 0.11.2, equinox 0.13.8, optax 0.2.8, jaxtyping and beartype, gymnax 1.0.1, brax 0.14.2,
mujoco and mujoco-mjx 3.16.0, playground 0.2.0. These versions come from
`docs/superpowers/specs/2026-10-22-jax-port-spike.md`.

**Spec:** `docs/superpowers/specs/2026-10-22-jax-port-design.md`. Read sections 1a, 4, 5, 5a, 6 and 7–10 first. Also
read Plan 1's carry-over list at the end of `docs/superpowers/plans/2026-10-22-jax-port-1-foundation.md`.

## Global Constraints

- **Minimal diff (spec 1a).** No renames of files, classes, methods, attributes, config keys or locals. Bodies are
  converted line by line. Comments and docstrings stay, edited only where they describe torch behaviour.
- **Allowed signature changes.** Explicit inputs and outputs may be added: a `key`, a `state: ModelState`, a returned
  `(state, ...)` or `metrics`. Host-only parameters with no traced meaning (`render`, `rew_field_name`) may be removed.
  Each removal is listed in its task.
- **Dtype.** `DTYPE = jnp.float32`. `jax_enable_x64` is never enabled. Tests run with `JAX_PLATFORMS=cpu`, which the
  conftest pins.
- **Masked data.** Every padded buffer marks its valid entries with a bool `mask`. Reductions use `jnp.where`. A
  `Batch` built with a mask zeroes the masked rows. Dones are stored as float32, never bool.
- **Every rollout produces at least one valid transition.** The rollout loop guarantees it, because each lane starts an
  episode.
- **Online/target buffers.** Before the jitted epoch is called with `donate_argnums`, the state's target arrays must be
  distinct buffers from the online ones (Plan 1 carry-over F5).
- **Gradients through state.** `likelihood_ratio` reads the policy from `state`, so gradients are taken through
  `state.set("policy", p)`. Never take them with respect to `self` (Plan 1 carry-over F6).
- **Optimiser states** are registered as `mc.add_main("<mc_key>_opt", ...)` before `mc.state()`.
- **jaxtyping.** Host-input arguments use `Float[np.ndarray | Array, "..."]`. Single-dimension shapes take a leading
  space (`" n"`). Every new module goes into `JAXTYPED_MODULES` in `tests/conftest.py`. `uv run ruff check src tests`
  stays clean.
- **Commit messages.** Sentence case, imperative, ending with a period. Each ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- **Commands.** Run everything with `uv run ...` from `/Users/kosmitive/git/abstract_rl`. A full suite run is
  `uv run pytest -o addopts="" -q 2>&1 | tail -3`.

## Review Focus

1. **Lane boundaries.** A lane that finishes an episode while the epoch is still short of `batch_size` resets and
   continues. Once the target is reached, no lane starts a new episode, but every lane finishes the one it is in. A
   reasonable user expects whole episodes and `steps >= batch_size`. Covered by Task 5,
   `test_rollout_collects_whole_episodes_until_batch_size`.
2. **Truncation is not termination.** Both an env time limit and the `max_steps` cut end an episode with `done = 0`,
   and TD bootstraps from the observed next state. Covered by Task 5 (`test_truncation_is_not_done`) and by Task 6's
   parity tests.
3. **Seeds are independent and reproducible.** Two seeds in one vmapped run equal the same seeds run alone, and each
   seed's run directory records its real seed. Covered by Task 9, `test_vmapped_seeds_match_single_seed_runs` and
   `test_each_seed_has_its_own_run_directory`.
4. **No recompilation per epoch.** The epoch is traced once per distinct `batch_size`. Covered by Task 9,
   `test_epoch_is_traced_once`.
5. **Observation normalisation stats only count valid observations.** Inactive lanes must not update `obs_rms`.
   Covered by Task 5, `test_obs_norm_counts_only_valid_observations`.

## File Structure

| File | Responsibility |
|---|---|
| `src/abstract_rl/env/jax_envs.py` (new) | The `JaxEnv` protocol, `Bounds`, the registry (`make_env`, `register_env`), the gymnax, Brax-MJX and Playground adapters |
| `src/abstract_rl/misc/running_mean_std.py` (new) | Port of Gymnasium's `RunningMeanStd` (obs_norm) with masked batch updates |
| `src/abstract_rl/misc/running_centered_filter.py` | `RunningCenteredFilter` as a pytree with masked sequential Welford updates |
| `src/abstract_rl/data_structures/temporal_difference_data/trajectory.py` | `Trajectory`: one lane `[L]` with ends and mask, plus `set` |
| `src/abstract_rl/data_structures/temporal_difference_data/trajectory_collection.py` | `TrajectoryCollection`: lanes `[E, L]`, plus `to_batch`, `sample` and `episode_stats` |
| `src/abstract_rl/data_structures/temporal_difference_data/trajectory_builder.py` | Deleted, because the rollout writes the buffers directly |
| `src/abstract_rl/env/mcmc_env_wrapper.py` | The lane rollout, preprocessing and env bounds |
| `src/abstract_rl/operator/{trajectory,td,gae,filter}_operator.py` | Per-lane transforms, with `transform_all` as a vmap |
| `src/abstract_rl/algorithms/mixed/trust_region_gradient.py`, `.../policy_gradient/trust_region_policy_optimization.py` | The TRPO step |
| `src/abstract_rl/data_structures/abstract_conf/model_configuration.py` | Equinox `store`, the D3 seed fix |
| `src/abstract_rl/run/run_seeds.py` (new) | `SeedRuns`: per-seed configurations, batched builds, the state with distinct targets, asynchronous stores |
| `src/abstract_rl/run/run_trpo.py` | The jitted, vmapped epoch and the host loop |
| `src/abstract_rl/scripts/bench_jax.py` (new) | The throughput benchmark |

---

### Task 1: Env dependencies, the `JaxEnv` protocol, gymnax adapters and the JAX PointMass

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `src/abstract_rl/env/jax_envs.py`
- Modify: `tests/conftest.py`, which registers a JAX `TestPointMass-v0` next to the Gymnasium one
- Create: `tests/test_jax_envs.py`

**Interfaces:**
- Produces:
  - `Bounds(low: np.ndarray, high: np.ndarray)`, a frozen dataclass;
  - `JaxEnv`, with the attributes `observation_dim: int`, `action_dim: int`, `action_low`, `action_high` and
    `obs_high` (float32 numpy `[A]` / `[Ds]`), and the methods
    `reset(key) -> (env_state, obs Float[Array, " Ds"])` and
    `step(env_state, action Float[Array, " A"], key) -> (env_state, obs, reward Float[Array, ""], terminated Bool[Array, ""], truncated Bool[Array, ""])`;
  - `register_env(env_id: str, factory: Callable[[], JaxEnv]) -> None` and `make_env(env_id: str) -> JaxEnv`
    (unknown ids raise `KeyError` with the list of known ids);
  - `GymnaxEnv(env_id)` and `PointMassEnv()` (the latter defined in `tests/conftest.py`).

- [ ] **Step 1: Add the env dependencies**

Add to `[project] dependencies` in `pyproject.toml`: `"gymnax>=1.0.1"`, `"brax>=0.14.2"`, `"mujoco>=3.16"`,
`"mujoco-mjx>=3.16"` and `"playground>=0.2.0"`. Then run
`uv sync --extra mps --extra mujoco --extra baselines --extra dm-control`.
Expected: the lock resolves with jax 0.11.2, the same as the spike.

- [ ] **Step 2: Write the failing tests**

`tests/test_jax_envs.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.env.jax_envs import make_env

KEY = jax.random.key(0)


def _roll(env, steps, action=None):
    st, obs = env.reset(KEY)
    out = []
    a = jnp.zeros(env.action_dim) if action is None else action
    for k in range(steps):
        st, obs, r, term, trunc = jax.jit(env.step)(st, a, jax.random.key(k))
        out.append((obs, r, term, trunc))
    return out


def _contract(env, obs_dim, act_dim):
    assert env.observation_dim == obs_dim and env.action_dim == act_dim
    assert env.action_low.shape == (act_dim,) and env.action_high.shape == (act_dim,)
    assert env.obs_high.shape == (obs_dim,)
    st, obs = jax.jit(env.reset)(KEY)
    assert obs.shape == (obs_dim,) and obs.dtype == jnp.float32
    st, obs, r, term, trunc = jax.jit(env.step)(st, jnp.zeros(act_dim), KEY)
    assert obs.shape == (obs_dim,) and r.shape == () and term.shape == () and trunc.shape == ()
    # the env is vmappable over lanes
    sts, obss = jax.vmap(env.reset)(jax.random.split(KEY, 3))
    jax.vmap(env.step)(sts, jnp.zeros((3, act_dim)), jax.random.split(KEY, 3))


def test_pendulum_contract_and_time_limit_is_a_truncation():
    env = make_env("Pendulum-v1")
    _contract(env, 3, 1)
    np.testing.assert_allclose(env.obs_high, [1.0, 1.0, 8.0])
    np.testing.assert_allclose(env.action_low, [-2.0]) and np.testing.assert_allclose(env.action_high, [2.0])
    steps = _roll(env, 200)
    assert not any(bool(t) for _, _, t, _ in steps)
    assert [bool(tr) for _, _, _, tr in steps] == [False] * 199 + [True]
    assert all(-16.3 <= float(r) <= 0.0 for _, r, _, _ in steps)


def test_mountaincar_contract_and_goal_is_a_termination():
    env = make_env("MountainCarContinuous-v0")
    _contract(env, 2, 1)
    np.testing.assert_allclose(env.action_low, [-1.0]) and np.testing.assert_allclose(env.action_high, [1.0])
    # full throttle in the direction of the velocity swings up and reaches the goal within the time limit
    st, obs = env.reset(KEY)
    terminated = False
    for k in range(999):
        a = jnp.where(obs[1] >= 0, 1.0, -1.0).reshape(1)
        st, obs, r, term, trunc = jax.jit(env.step)(st, a, jax.random.key(k))
        if bool(term):
            terminated = True
            assert not bool(trunc) and float(r) > 90.0
            break
    assert terminated


def test_point_mass_contract():
    env = make_env("TestPointMass-v0")
    _contract(env, 4, 2)
    np.testing.assert_allclose(env.action_low, [-1.0, -2.0])
    steps = _roll(env, 30)
    assert [bool(tr) for _, _, _, tr in steps] == [False] * 29 + [True]


def test_unknown_env_lists_the_known_ids():
    with pytest.raises(KeyError, match="Pendulum-v1"):
        make_env("Nope-v0")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_jax_envs.py -q`
Expected: FAIL with `ModuleNotFoundError: abstract_rl.env.jax_envs`.

- [ ] **Step 4: Write `jax_envs.py`**

```python
"""Pure-JAX environments behind one small protocol, keyed by the Gymnasium ids used in ``config/env``.

Every adapter maps its backend onto Gymnasium's ``terminated`` / ``truncated`` semantics: a time limit is a
truncation, only a real terminal state is a termination (it is stored as ``done`` and not bootstrapped).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from abstract_rl.misc.typing import DTYPE


@dataclass(frozen=True)
class Bounds:
    """Box bounds of an observation or action space (``low`` / ``high`` like ``gymnasium.spaces.Box``)."""

    low: np.ndarray
    high: np.ndarray


class JaxEnv(Protocol):
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_high: np.ndarray

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]: ...

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]: ...


_REGISTRY: dict[str, Callable[[], JaxEnv]] = {}


def register_env(env_id: str, factory: Callable[[], JaxEnv]) -> None:
    """Register (or replace) the factory of an environment id."""
    _REGISTRY[env_id] = factory


def make_env(env_id: str) -> JaxEnv:
    """Build the JAX environment of a Gymnasium id."""
    if env_id not in _REGISTRY:
        raise KeyError(f"unknown environment {env_id}, known: {sorted(_REGISTRY)}")
    return _REGISTRY[env_id]()


def _flat(x: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, np.float32), shape).reshape(-1).astype(np.float32)


class GymnaxEnv:
    """gymnax environment, stepped with ``step_env`` (no auto reset); its ``done`` is split into the two flags."""

    def __init__(self, env_id: str) -> None:
        import gymnax

        self._env, self._params = gymnax.make(env_id)
        self._env_id = env_id
        act, obs = self._env.action_space(self._params), self._env.observation_space(self._params)
        self._action_shape = tuple(act.shape)
        self.action_dim = int(np.prod(act.shape))
        self.observation_dim = int(np.prod(obs.shape))
        self.action_low, self.action_high = _flat(act.low, act.shape), _flat(act.high, act.shape)
        self.obs_high = _flat(obs.high, obs.shape)

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        obs, env_state = self._env.reset_env(key, self._params)
        return env_state, jnp.asarray(obs, DTYPE).reshape(-1)

    def _terminated(self, env_state: Any) -> Bool[Array, ""]:
        if self._env_id == "MountainCarContinuous-v0":
            return (env_state.position >= self._params.goal_position) & (
                env_state.velocity >= self._params.goal_velocity
            )
        return jnp.asarray(False)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        obs, env_state, reward, done, _ = self._env.step_env(key, env_state, action.reshape(self._action_shape), self._params)
        terminated = jnp.asarray(self._terminated(env_state)).reshape(())
        truncated = jnp.asarray(done, bool).reshape(()) & ~terminated
        return env_state, jnp.asarray(obs, DTYPE).reshape(-1), jnp.asarray(reward, DTYPE).reshape(()), terminated, truncated


register_env("Pendulum-v1", lambda: GymnaxEnv("Pendulum-v1"))
register_env("MountainCarContinuous-v0", lambda: GymnaxEnv("MountainCarContinuous-v0"))
```

If gymnax's `step_env` needs the action as a scalar for these envs, or returns a reward or obs of a different shape,
adapt only the reshapes in `step`. The contract test pins the result. The same applies if gymnax's
MountainCarContinuous state or params name the goal fields differently. Read
`gymnax/environments/classic_control/continuous_mountain_car.py` in the venv for the real names.

- [ ] **Step 5: Register the JAX PointMass in the conftest**

Add this to `tests/conftest.py` after the Gymnasium `PointMass` registration. Keep the Gymnasium one, because the torch
fixture recorder imports it.

```python
from typing import NamedTuple

import jax
import jax.numpy as jnp

from abstract_rl.env.jax_envs import register_env


class _PointMassState(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    t: jax.Array


class PointMassEnv:
    """JAX twin of the Gymnasium PointMass above: same dynamics, reward, bounds and 30-step truncation."""

    observation_dim, action_dim = 4, 2
    action_low = np.array([-1.0, -2.0], np.float32)
    action_high = np.array([1.0, 2.0], np.float32)
    obs_high = np.full(4, np.inf, np.float32)

    def reset(self, key):
        pos = jax.random.uniform(key, (2,), jnp.float32, -1.0, 1.0)
        st = _PointMassState(pos, jnp.zeros(2, jnp.float32), jnp.asarray(0, jnp.int32))
        return st, jnp.concatenate([st.pos, st.vel])

    def step(self, st, action, key):
        a = jnp.clip(action, self.action_low, self.action_high)
        vel = 0.9 * st.vel + 0.1 * a
        pos = st.pos + 0.1 * vel
        st = _PointMassState(pos, vel, st.t + 1)
        reward = -(pos @ pos) - 0.01 * (a @ a)
        return st, jnp.concatenate([pos, vel]), reward.astype(jnp.float32), jnp.asarray(False), st.t >= 30


register_env("TestPointMass-v0", PointMassEnv)
```

Add `"abstract_rl.env.jax_envs"` to `JAXTYPED_MODULES`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_jax_envs.py -q`
Expected: 4 passed. Then run the full suite. Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/abstract_rl/env/jax_envs.py tests/conftest.py tests/test_jax_envs.py
git commit -m "Add the JAX environment protocol with gymnax adapters and a JAX PointMass.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Brax-MJX and Playground adapters

**Files:**
- Modify: `src/abstract_rl/env/jax_envs.py` (append)
- Modify: `tests/test_jax_envs.py` (append)

**Interfaces:**
- Consumes: the protocol and registry from Task 1.
- Produces: `BraxEnv(name)` and `PlaygroundEnv(name)`, registered under `InvertedPendulum-v5`,
  `InvertedDoublePendulum-v5`, `HalfCheetah-v5` and `AcrobotSwingup-v0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jax_envs.py`:

```python
@pytest.mark.parametrize(
    "env_id, obs_dim, act_dim",
    [("InvertedPendulum-v5", None, 1), ("InvertedDoublePendulum-v5", None, 1), ("HalfCheetah-v5", None, 6),
     ("AcrobotSwingup-v0", None, 1)],
)
def test_mjx_contract(env_id, obs_dim, act_dim):
    env = make_env(env_id)
    _contract(env, env.observation_dim, act_dim)
    assert np.all(env.action_low < env.action_high)
    assert np.all(np.isinf(env.obs_high))


def test_inverted_pendulum_falls_and_terminates():
    env = make_env("InvertedPendulum-v5")
    st, obs = env.reset(KEY)
    step = jax.jit(env.step)
    for k in range(200):
        st, obs, r, term, trunc = step(st, jnp.full(1, env.action_high[0]), jax.random.key(k))
        if bool(term):
            break
    assert bool(term) and not bool(trunc)


def test_halfcheetah_never_terminates():
    env = make_env("HalfCheetah-v5")
    steps = _roll(env, 50, jnp.full(6, 0.5))
    assert not any(bool(t) or bool(tr) for _, _, t, tr in steps)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_jax_envs.py -q -k "mjx or inverted or halfcheetah"`
Expected: FAIL with `KeyError: unknown environment InvertedPendulum-v5`.

- [ ] **Step 3: Append the adapters**

```python
def _ctrl_bounds(ctrlrange: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Actuator bounds of a MuJoCo model; unlimited actuators (an all-zero range) fall back to [-1, 1]."""
    ctrl = np.asarray(ctrlrange, np.float32).reshape(action_dim, 2)
    if not np.all(ctrl[:, 0] < ctrl[:, 1]):
        return np.full(action_dim, -1.0, np.float32), np.full(action_dim, 1.0, np.float32)
    return ctrl[:, 0].copy(), ctrl[:, 1].copy()


class BraxEnv:
    """Brax environment on the MJX backend. Brax ``done`` is a termination; time limits are applied by ``max_steps``."""

    def __init__(self, name: str) -> None:
        from brax import envs

        self._env = envs.get_environment(name, backend="mjx")
        self.observation_dim = int(self._env.observation_size)
        self.action_dim = int(self._env.action_size)
        self.action_low, self.action_high = _ctrl_bounds(self._env.sys.actuator_ctrlrange, self.action_dim)
        self.obs_high = np.full(self.observation_dim, np.inf, np.float32)

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        env_state = self._env.reset(key)
        return env_state, jnp.asarray(env_state.obs, DTYPE)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        env_state = self._env.step(env_state, action)
        return (env_state, jnp.asarray(env_state.obs, DTYPE), jnp.asarray(env_state.reward, DTYPE).reshape(()),
                jnp.asarray(env_state.done).reshape(()) > 0.5, jnp.asarray(False))


class PlaygroundEnv:
    """MuJoCo Playground environment (MJX). Its ``done`` is a termination; time limits are applied by ``max_steps``."""

    def __init__(self, name: str) -> None:
        from mujoco_playground import registry

        self._env = registry.load(name)
        self.observation_dim = int(np.prod(np.asarray(self._env.observation_size)))
        self.action_dim = int(self._env.action_size)
        self.action_low, self.action_high = _ctrl_bounds(self._env.mj_model.actuator_ctrlrange, self.action_dim)
        self.obs_high = np.full(self.observation_dim, np.inf, np.float32)

    def reset(self, key: PRNGKeyArray) -> tuple[Any, Float[Array, " Ds"]]:
        env_state = self._env.reset(key)
        return env_state, jnp.asarray(env_state.obs, DTYPE).reshape(-1)

    def step(
        self, env_state: Any, action: Float[Array, " A"], key: PRNGKeyArray
    ) -> tuple[Any, Float[Array, " Ds"], Float[Array, ""], Bool[Array, ""], Bool[Array, ""]]:
        env_state = self._env.step(env_state, action)
        return (env_state, jnp.asarray(env_state.obs, DTYPE).reshape(-1), jnp.asarray(env_state.reward, DTYPE).reshape(()),
                jnp.asarray(env_state.done).reshape(()) > 0.5, jnp.asarray(False))


register_env("InvertedPendulum-v5", lambda: BraxEnv("inverted_pendulum"))
register_env("InvertedDoublePendulum-v5", lambda: BraxEnv("inverted_double_pendulum"))
register_env("HalfCheetah-v5", lambda: BraxEnv("halfcheetah"))
register_env("AcrobotSwingup-v0", lambda: PlaygroundEnv("AcrobotSwingup"))
```

If `env.sys.actuator_ctrlrange` or `mj_model.actuator_ctrlrange` has another name in the installed versions, use the
matching MuJoCo model field. Record the name used in the report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_jax_envs.py -q`
Expected: all passed. MJX compiles take a few seconds each. Then run the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/abstract_rl/env/jax_envs.py tests/test_jax_envs.py
git commit -m "Add the Brax-MJX and Playground environment adapters.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `RunningCenteredFilter` and `RunningMeanStd` as masked pytrees

**Files:**
- Modify: `src/abstract_rl/misc/running_centered_filter.py` (whole file)
- Create: `src/abstract_rl/misc/running_mean_std.py`
- Create: `tests/test_filters.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces, `RunningCenteredFilter(name: str, dim: int, std_dev_noise=1e-8, clip=5)`, an `eqx.Module`:
  - the fields `_n` (int32 scalar), `_m` `[dim]`, `_s` `[dim]` (float32), and the static `_name`, `_dim`, `_clip` and
    `_std_dev_noise`;
  - `register(xs: [N, dim] or [dim], mask: Bool[N] | None = None) -> RunningCenteredFilter` (sequential Welford over
    the valid rows, in order);
  - `__call__(x, update: bool = False)`, which normalises with the current statistics and clips to ±clip
    (`update=True` raises `ValueError`, because updating returns a new filter through `register`);
  - the properties `mean`, `var` and `std`, unchanged in meaning.
- Produces, `RunningMeanStd`, an `eqx.Module` with `mean`, `var`, `count`:
  - `RunningMeanStd.create(dim)` with mean 0, var 1, count 1e-4 (Gymnasium's defaults);
  - `update(xs [N, dim], mask Bool[N] | None = None) -> RunningMeanStd` (parallel-variance batch update over the valid
    rows; no change if none are valid);
  - `normalize(x) -> (x - mean) / sqrt(var + 1e-8)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_filters.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from golden import golden

from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.misc.running_mean_std import RunningMeanStd


def test_running_centered_filter_matches_torch_sequence():
    g = golden("filter")
    f = RunningCenteredFilter("states", 3)
    for i, x in enumerate(g["xs"]):
        f = f.register(jnp.asarray(x))
        np.testing.assert_allclose(f(jnp.asarray(x)), g["outs"][i], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(f.mean, g["means"][i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f.std, g["stds"][i], rtol=1e-4, atol=1e-6)


def test_running_centered_filter_masked_rows_are_skipped_in_order():
    xs = jnp.asarray(np.random.default_rng(0).normal(size=(6, 2)), jnp.float32)
    mask = jnp.array([True, False, True, True, False, True])
    masked = RunningCenteredFilter("s", 2).register(xs, mask)
    plain = RunningCenteredFilter("s", 2).register(xs[np.asarray(mask)])
    np.testing.assert_allclose(masked.mean, plain.mean, rtol=1e-6)
    np.testing.assert_allclose(masked.var, plain.var, rtol=1e-5)
    assert int(masked._n) == 4
    jax.jit(lambda f, x, m: f.register(x, m))(RunningCenteredFilter("s", 2), xs, mask)


def test_running_mean_std_matches_gymnasium_sequence():
    g = golden("obsnorm")
    rms = RunningMeanStd.create(3)
    for i, x in enumerate(g["xs"]):
        rms = rms.update(jnp.asarray(x)[None])
        np.testing.assert_allclose(rms.mean, g["means"][i], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(rms.var, g["vars"][i], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(rms.normalize(jnp.asarray(x)), g["outs"][i], rtol=1e-4, atol=1e-5)


def test_running_mean_std_batch_update_equals_sequential_and_ignores_masked_rows():
    xs = jnp.asarray(np.random.default_rng(1).normal(size=(5, 3)) * 3 + 1, jnp.float32)
    seq = RunningMeanStd.create(3)
    for x in xs[:4]:
        seq = seq.update(x[None])
    batch = RunningMeanStd.create(3).update(xs, jnp.array([True, True, True, True, False]))
    np.testing.assert_allclose(batch.mean, seq.mean, rtol=1e-5)
    np.testing.assert_allclose(batch.var, seq.var, rtol=1e-4)
    np.testing.assert_allclose(batch.count, seq.count, rtol=1e-6)
    none = RunningMeanStd.create(3).update(xs, jnp.zeros(5, bool))
    np.testing.assert_allclose(none.mean, 0.0) and np.testing.assert_allclose(none.var, 1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_filters.py -q`
Expected: FAIL. `register` returns `None`, and `running_mean_std` is missing.

- [ ] **Step 3: Convert `running_centered_filter.py`**

```python
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from abstract_rl.misc.typing import DTYPE

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
# https://github.com/ikostrikov/pytorch-trpo/blob/master/running_state.py


class RunningCenteredFilter(eqx.Module):
    _n: Int[Array, ""]
    _m: Float[Array, " dim"]
    _s: Float[Array, " dim"]
    _name: str = eqx.field(static=True)
    _dim: int = eqx.field(static=True)
    _clip: float = eqx.field(static=True)
    _std_dev_noise: float = eqx.field(static=True)

    def __init__(self, name: str, dim: int, std_dev_noise: float = 1e-8, clip: int = 5) -> None:

        # build running variables
        self._n = jnp.asarray(0, jnp.int32)
        self._dim = dim
        self._m = jnp.zeros(dim, DTYPE)
        self._s = jnp.zeros(dim, DTYPE)
        self._name = name
        self._clip = float(clip)
        self._std_dev_noise = std_dev_noise

    def register(self, xs: Float[np.ndarray | Array, "..."], mask: Bool[Array, " N"] | None = None) -> "RunningCenteredFilter":
        """Welford update with every (valid) row of ``xs``, in order. Returns the updated filter."""
        xs = jnp.asarray(xs, DTYPE).reshape([-1, self._dim])
        mask = jnp.ones(xs.shape[0], bool) if mask is None else mask

        def one(carry: tuple[Any, Any, Any], row: tuple[Any, Any]) -> tuple[tuple[Any, Any, Any], None]:
            n, m, s = carry
            x, valid = row
            n_new = n + 1
            # trivial
            first = n_new == 1
            # update
            old_mean = m
            m_new = jnp.where(first, x, old_mean + (x - old_mean) / n_new)
            s_new = jnp.where(first, s, s + (x - old_mean) * (x - m_new))
            return (jnp.where(valid, n_new, n), jnp.where(valid, m_new, m), jnp.where(valid, s_new, s)), None

        (n, m, s), _ = jax.lax.scan(one, (self._n, self._m, self._s), (xs, mask))
        return eqx.tree_at(lambda f: (f._n, f._m, f._s), self, (n, m, s))

    def __call__(self, x: Float[np.ndarray | Array, "..."], update: bool = False) -> Float[Array, "..."]:
        if update:
            raise ValueError("register returns the updated filter, call it before normalizing")
        x = (jnp.asarray(x, DTYPE) - self.mean) / (self.std + self._std_dev_noise)
        x = jnp.clip(x, -self._clip, self._clip)
        return x

    @property
    def mean(self) -> Float[Array, " dim"]:
        return self._m

    @property
    def var(self) -> Float[Array, " dim"]:
        return jnp.where(self._n > 1, self._s / jnp.maximum(self._n - 1, 1), self._m * self._m)

    @property
    def std(self) -> Float[Array, " dim"]:
        return jnp.sqrt(self.var)
```

- [ ] **Step 4: Write `running_mean_std.py`**

```python
"""Observation normalisation statistics of Gymnasium's ``NormalizeObservation`` (``RunningMeanStd``) as a pytree."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from abstract_rl.misc.typing import DTYPE


class RunningMeanStd(eqx.Module):
    mean: Float[Array, " dim"]
    var: Float[Array, " dim"]
    count: Float[Array, ""]

    @classmethod
    def create(cls, dim: int, epsilon: float = 1e-4) -> "RunningMeanStd":
        return cls(jnp.zeros(dim, DTYPE), jnp.ones(dim, DTYPE), jnp.asarray(epsilon, DTYPE))

    def update(self, xs: Float[np.ndarray | Array, "N dim"], mask: Bool[Array, " N"] | None = None) -> "RunningMeanStd":
        """Parallel mean / variance update with the valid rows (Gymnasium ``update_mean_var_count_from_moments``)."""
        xs = jnp.asarray(xs, DTYPE)
        mask = jnp.ones(xs.shape[0], bool) if mask is None else mask
        w = mask.astype(DTYPE)[:, None]
        batch_count = jnp.sum(mask).astype(DTYPE)
        safe = jnp.maximum(batch_count, 1.0)
        batch_mean = jnp.sum(w * jnp.where(mask[:, None], xs, 0.0), axis=0) / safe
        batch_var = jnp.sum(w * jnp.where(mask[:, None], (xs - batch_mean) ** 2, 0.0), axis=0) / safe

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        any_valid = batch_count > 0
        return RunningMeanStd(
            jnp.where(any_valid, new_mean, self.mean), jnp.where(any_valid, new_var, self.var), tot_count
        )

    def normalize(self, x: Float[np.ndarray | Array, "..."], epsilon: float = 1e-8) -> Float[Array, "..."]:
        return (jnp.asarray(x, DTYPE) - self.mean) / jnp.sqrt(self.var + epsilon)
```

- [ ] **Step 5: Run the tests to verify they pass**

Add `"abstract_rl.misc.running_centered_filter"` and `"abstract_rl.misc.running_mean_std"` to `JAXTYPED_MODULES`.

Run: `uv run pytest tests/test_filters.py -q`
Expected: 4 passed. Then run the full suite.

- [ ] **Step 6: Commit**

```bash
git add src/abstract_rl/misc/running_centered_filter.py src/abstract_rl/misc/running_mean_std.py tests/test_filters.py tests/conftest.py
git commit -m "Port the running filters to masked pytrees.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Lane `Trajectory` and `TrajectoryCollection`

**Files:**
- Modify: `src/abstract_rl/data_structures/temporal_difference_data/trajectory.py` (whole file)
- Modify: `src/abstract_rl/data_structures/temporal_difference_data/trajectory_collection.py` (whole file)
- Delete: `src/abstract_rl/data_structures/temporal_difference_data/trajectory_builder.py`
- Create: `tests/test_trajectories.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces, `Trajectory` (one lane, `L` steps), an `eqx.Module`:
  - constructor
    `Trajectory(states [L,Ds], next_states [L,Ds], actions [L,A], rewards [L,1], raw_rewards [L], dones [L,1], log_likelihoods [L,1], ends Bool[L], mask Bool[L], annotations=None)`;
  - the properties `states`, `next_states`, `actions`, `rewards`, `raw_rewards`, `dones`, `log_likelihoods`, `ends`,
    `mask`;
  - `__getitem__`, `__contains__` and `__len__` (= L);
  - `set(key, value) -> Trajectory` and `replace(**fields) -> Trajectory`.
  - `all_states`, `traj_id`, `env`, the pandas frame and the reward helpers are removed. A lane holds several episodes,
    and their ends are marked by `ends`.
- Produces, `TrajectoryCollection(env, trajectories: Trajectory)`, an `eqx.Module`. The trajectories carry a leading
  lane axis `[E]`. It provides:
  - `__len__` (= E), `trajectories()` (the batched `Trajectory`) and `with_trajectories(t) -> TrajectoryCollection`;
  - `to_batch(copy_fields=None) -> Batch`, with lanes flattened to `[E*L]` and `mask` set;
  - `sample(key, batch_size, copy_fields=None) -> Batch`, uniform over valid transitions with replacement, no mask;
  - `episode_stats() -> dict[str, Array]` with `returns` `[E*L]` (raw), `lengths` `[E*L]` and `valid` `Bool[E*L]`, one
    slot per possible episode.
  - `merge`, `extend`, `print` and `max_l` are removed. They served the MPO replay memory (Plan 3) and host printing;
    the run loop prints from `episode_stats`.

- [ ] **Step 1: Write the failing tests**

`tests/test_trajectories.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


def _lanes():
    # lane 0: episodes of length 2 and 3, then padding; lane 1: one episode of length 4, then padding
    ends = jnp.array([[0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]], bool)
    mask = jnp.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], bool)
    e, ell = ends.shape
    raw = jnp.arange(e * ell, dtype=jnp.float32).reshape(e, ell)
    z = jnp.zeros((e, ell, 1))
    s = jnp.arange(e * ell * 3, dtype=jnp.float32).reshape(e, ell, 3)
    t = Trajectory(s, s + 1, jnp.zeros((e, ell, 2)), raw[..., None], raw, z, z, ends, mask, {"a": raw})
    return TrajectoryCollection(None, t)


def test_to_batch_flattens_lanes_and_masks_padding():
    tc = _lanes()
    b = tc.to_batch(["a"])
    assert b.states.shape == (12, 3) and b["a"].shape == (12,) and b.mask.shape == (12,)
    assert int(b.mask.sum()) == 9
    assert float(b["a"][5]) == 0.0  # the padded row is zeroed by Batch


def test_sample_only_draws_valid_transitions():
    tc = _lanes()
    b = jax.jit(lambda c, k: c.sample(k, 2000, ["a"]))(tc, jax.random.key(0))
    assert b.mask is None and b.states.shape == (2000, 3)
    drawn = set(np.asarray(b["a"]).astype(int).tolist())
    assert drawn == {0, 1, 2, 3, 4, 6, 7, 8, 9}


def test_episode_stats_sum_raw_rewards_per_episode():
    st = jax.jit(lambda c: c.episode_stats())(_lanes())
    valid = np.asarray(st["valid"])
    assert valid.sum() == 3
    np.testing.assert_allclose(sorted(np.asarray(st["returns"])[valid]), sorted([0 + 1, 2 + 3 + 4, 6 + 7 + 8 + 9]))
    np.testing.assert_allclose(sorted(np.asarray(st["lengths"])[valid]), [2, 3, 4])


def test_trajectory_set_returns_a_new_lane():
    t = _lanes().trajectories()
    t2 = t.set("tv", jnp.ones((2, 6)))
    assert "tv" in t2 and "tv" not in t
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_trajectories.py -q`
Expected: FAIL, because the `Trajectory` constructor signature differs.

- [ ] **Step 3: Rewrite `trajectory.py`**

```python
from typing import Any

import equinox as eqx


class Trajectory(eqx.Module):
    """
    One lane of a rollout: L consecutive steps holding one or more whole episodes. ``ends`` marks the last step of every
    episode (terminated, truncated by the environment or cut at max_steps); ``dones`` marks true terminations only;
    ``mask`` marks the valid steps (the lane is padded after its last episode). A trajectory collection holds the lanes
    of one rollout with a leading lane axis.
    """

    _states: Any
    _next_states: Any
    _actions: Any
    _rewards: Any
    _raw_rewards: Any
    _dones: Any
    _log_likelihoods: Any
    _ends: Any
    _mask: Any
    _annotations: dict[str, Any]

    def __init__(
        self,
        states: Any,
        next_states: Any,
        actions: Any,
        rewards: Any,
        raw_rewards: Any,
        dones: Any,
        log_likelihoods: Any,
        ends: Any,
        mask: Any,
        annotations: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new trajectory from the (padded) step arrays of one lane."""
        self._states = states
        self._next_states = next_states
        self._actions = actions
        self._rewards = rewards
        self._raw_rewards = raw_rewards
        self._dones = dones
        self._log_likelihoods = log_likelihoods
        self._ends = ends
        self._mask = mask
        self._annotations = dict(annotations) if annotations is not None else {}

    def __len__(self) -> int:
        return int(self._mask.shape[-1])

    def set(self, key: str, item: Any) -> "Trajectory":
        """A new trajectory with the annotation ``key`` set (trajectories are immutable)."""
        return eqx.tree_at(lambda t: t._annotations, self, {**self._annotations, key: item})

    def replace(self, **fields: Any) -> "Trajectory":
        """A new trajectory with some step arrays replaced, e.g. ``replace(states=...)``."""
        names = tuple(f"_{k}" for k in fields)
        return eqx.tree_at(lambda t: tuple(getattr(t, n) for n in names), self, tuple(fields.values()))

    def __getitem__(self, key: str) -> Any:
        return self._annotations[key]

    def __contains__(self, value: str) -> bool:
        return value in self._annotations

    # --- standard reinforcement learning fields

    @property
    def states(self) -> Any:
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states

    @property
    def actions(self) -> Any:
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def dones(self) -> Any:
        """
        True terminations of the trajectory (truncations are not done).
        :return: The done flags.
        """
        return self._dones

    @property
    def log_likelihoods(self) -> Any:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

    @property
    def rewards(self) -> Any:
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def raw_rewards(self) -> Any:
        """Rewards of the environment, before the training transform (for the logged return)."""
        return self._raw_rewards

    @property
    def next_states(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._next_states

    @property
    def ends(self) -> Any:
        """Last step of every episode."""
        return self._ends

    @property
    def mask(self) -> Any:
        """Valid steps."""
        return self._mask
```

- [ ] **Step 4: Rewrite `trajectory_collection.py`**

Keep the MIT license header, then:

```python
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory


def _flat(x: Any) -> Any:
    return x.reshape((-1,) + x.shape[2:])


class TrajectoryCollection(eqx.Module):
    """Represents a simple trajectory collection: the lanes [E] of one rollout, each a padded Trajectory of L steps."""

    _trajectories: Trajectory

    def __init__(self, env: Any, trajectories: Trajectory) -> None:
        self._trajectories = trajectories

    def __len__(self) -> int:
        return int(self._trajectories.mask.shape[0])

    def trajectories(self) -> Trajectory:
        return self._trajectories

    def with_trajectories(self, trajectories: Trajectory) -> "TrajectoryCollection":
        return eqx.tree_at(lambda c: c._trajectories, self, trajectories)

    def to_batch(self, copy_fields: list[str] | None = None) -> Batch:
        """Convert the trajectory collection to a batch.

        :param copy_fields: The fields to copy into the batch.
        :return: The final created batch, padded steps are masked.
        """
        t = self._trajectories
        fields = copy_fields if copy_fields is not None else []
        annots = {a: _flat(t[a]) for a in fields}
        return Batch(
            _flat(t.states), _flat(t.actions), _flat(t.log_likelihoods), _flat(t.rewards), _flat(t.next_states),
            _flat(t.dones), annots, mask=_flat(t.mask),
        )

    def sample(self, key: PRNGKeyArray, batch_size: int, copy_fields: list[str] | None = None) -> Batch:
        """Uniformly sample ``batch_size`` valid transitions with replacement."""
        t = self._trajectories
        fields = copy_fields if copy_fields is not None else []
        mask = _flat(t.mask)
        p = mask.astype(jnp.float32) / jnp.sum(mask)
        tindices = jax.random.choice(key, mask.shape[0], (batch_size,), replace=True, p=p)

        def take(x: Any) -> Any:
            return _flat(x)[tindices]

        annots = {a: take(t[a]) for a in fields}
        return Batch(take(t.states), take(t.actions), take(t.log_likelihoods), take(t.rewards), take(t.next_states),
                     take(t.dones), annots)

    def episode_stats(self) -> dict[str, Array]:
        """Raw return and length of every episode: one slot per possible episode, ``valid`` marks the real ones."""
        t = self._trajectories
        e, length = t.mask.shape
        ends = t.ends & t.mask
        episode = jnp.cumsum(ends, axis=1) - ends + jnp.arange(e)[:, None] * length
        seg = episode.reshape(-1)
        raw = jnp.where(t.mask, t.raw_rewards, 0.0).reshape(-1)
        returns = jax.ops.segment_sum(raw, seg, num_segments=e * length)
        lengths = jax.ops.segment_sum(t.mask.reshape(-1).astype(jnp.int32), seg, num_segments=e * length)
        valid = jax.ops.segment_sum(ends.reshape(-1).astype(jnp.int32), seg, num_segments=e * length) > 0
        return {"returns": returns, "lengths": lengths, "valid": valid}
```

Delete `trajectory_builder.py`. Its only users are the torch env wrapper, which Task 5 replaces, and the torch fixture
recorder, which runs from a worktree of 4f7828e. Add the two modules to `JAXTYPED_MODULES`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trajectories.py -q`
Expected: 4 passed.

Then run the full suite. `tests/test_imports.py` must still pass, because the deleted module is no longer imported.
`mcmc_env_wrapper.py` still imports `TrajectoryBuilder` until Task 5. Remove that import line in this task, and leave
the rest of the old wrapper body in place, unreachable, for Task 5 to replace.

- [ ] **Step 6: Commit**

```bash
git add -A src/abstract_rl/data_structures/temporal_difference_data src/abstract_rl/env/mcmc_env_wrapper.py tests/test_trajectories.py tests/conftest.py
git commit -m "Store rollouts as padded lanes of whole episodes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The lane rollout in `MCMCEnvWrapper`

**Files:**
- Modify: `src/abstract_rl/env/mcmc_env_wrapper.py` (whole file)
- Modify: `tests/test_env_wrapper.py` and `tests/test_obs_norm.py` (whole files, without skip markers)
- Modify: `docs/superpowers/specs/2026-10-22-jax-port-design.md` (section 6.1 and D2)
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: `make_env`/`Bounds` (Task 1), the filters (Task 3), `Trajectory`/`TrajectoryCollection` (Task 4) and the
  policies (Plan 1).
- Produces, `MCMCEnvWrapper(mc)`. It keeps `discount()`, `observation_space` / `action_space` (now `Bounds`),
  `observation_dim` and `action_dim`. The new and changed methods are:
  - `init_state() -> dict[str, Any]`, returning `obs_rms`, `state_filter` and `reward_filter`, which are registered
    with `mc.add_main` by the run loop;
  - `num_envs(batch_size, max_steps) -> int`, which reads the conf key `env_num_envs` (`auto` means
    `ceil(batch_size / max_steps)`);
  - `execute_policy(state, key, policy, max_steps, batch_size, exploration=True) -> (state, TrajectoryCollection, metrics)`.
    It removes `render` and `rew_field_name` (host-only). The metrics are `steps` (int), `episode_returns` (masked
    `[E*L]`), `episode_valid`, `episode_lengths` and `max_traj_len`. The state's `obs_rms`, `state_filter` and
    `reward_filter` are updated.
  - `execute_policy_once(state, key, max_steps, policy, opt=False) -> (state, Trajectory)`, which runs one lane and one
    episode.
  - The host counters `total_steps`, `epoch` and `best_reward` move to the run loop.

- [ ] **Step 1: Revise the spec**

In section 6.1, replace the "Rounds" bullet and its sub-bullets with:

"**Lanes.** E environments run in lockstep inside one `lax.while_loop`. Each lane collects whole episodes back to back
into a buffer `[E, L]` with `L = ceil(batch_size / E) + max_steps`. When a lane ends an episode while the epoch has
fewer than `batch_size` valid transitions, it resets and continues. Otherwise it stops. A lane never cuts an episode,
and no lane starts an episode after the target is reached. `ends` marks the last step of every episode and `mask` the
valid steps."

Replace the D2 row of section 10 with:

"D2 | An epoch overshoots `batch_size` by at most the episodes still running in the other E−1 lanes; torch overshoots
by at most one episode. E = 1 is identical to torch. | Static shapes for the vmapped envs, with memory linear in
`batch_size + E·max_steps`."

- [ ] **Step 2: Write the failing tests**

Replace `tests/test_env_wrapper.py`:

```python
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.env.jax_envs import register_env
from abstract_rl.policy.continuous.select_policy import select_policy

KEY = jax.random.key(0)


class _CountState(NamedTuple):
    t: jax.Array


class _TerminatesAfterThree:
    """Terminates on the third step; a 1-D action space with symmetric bounds."""

    observation_dim, action_dim = 1, 1
    action_low, action_high = np.array([-1.0], np.float32), np.array([1.0], np.float32)
    obs_high = np.array([np.inf], np.float32)

    def reset(self, key):
        return _CountState(jnp.asarray(0, jnp.int32)), jnp.zeros(1)

    def step(self, st, action, key):
        st = _CountState(st.t + 1)
        return st, st.t.astype(jnp.float32).reshape(1), jnp.asarray(1.0), st.t >= 3, jnp.asarray(False)


register_env("TestTerminates-v0", _TerminatesAfterThree)


def _setup(make_pendulum_mc, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, KEY)
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    return env, policy, state


def test_truncation_is_not_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 200, policy)
    assert int(traj.mask.sum()) == 200  # Pendulum-v1 truncates after 200 steps
    assert not np.any(np.asarray(traj.dones)) and bool(traj.ends[199])


def test_max_steps_cut_is_not_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 10, policy)
    assert int(traj.mask.sum()) == 10 and not np.any(np.asarray(traj.dones)) and bool(traj.ends[9])


def test_termination_is_done(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_name="TestTerminates-v0", env_num_envs=1)
    _, traj = env.execute_policy_once(state, KEY, 50, policy)
    assert int(traj.mask.sum()) == 3
    np.testing.assert_allclose(np.asarray(traj.dones)[:3, 0], [0.0, 0.0, 1.0])


def test_rollout_collects_whole_episodes_until_batch_size(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_name="TestTerminates-v0", env_num_envs=4)
    _, tc, m = jax.jit(lambda s, k: env.execute_policy(s, k, policy, 50, 10))(state, KEY)
    t = tc.trajectories()
    steps = int(m["steps"])
    assert steps >= 10 and steps % 3 == 0  # whole 3-step episodes only
    assert steps < 10 + 4 * 3  # at most the running episodes of the other lanes on top
    assert int(np.asarray(m["episode_valid"]).sum()) == steps // 3
    # every lane ends its last valid step with an episode end
    for lane in range(4):
        valid = np.asarray(t.mask[lane])
        if valid.any():
            assert bool(t.ends[lane][np.nonzero(valid)[0][-1]])


def test_single_lane_matches_sequential_semantics(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=1)
    _, tc, m = env.execute_policy(state, KEY, policy, 200, 450)
    assert int(m["steps"]) == 600 and int(np.asarray(m["episode_valid"]).sum()) == 3  # like torch: 3 whole episodes
```

Replace `tests/test_obs_norm.py`:

```python
"""``obs_norm`` is Gymnasium's NormalizeObservation: the stats see every valid observation (resets and steps)."""

import jax
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.policy.continuous.select_policy import select_policy


def test_obs_norm_counts_only_valid_observations(make_pendulum_mc):
    mc = make_pendulum_mc(obs_norm=True, env_name="TestPointMass-v0", env_num_envs=3)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    state, tc, m = env.execute_policy(state, jax.random.key(1), policy, 30, 40)
    episodes = int(np.asarray(m["episode_valid"]).sum())
    # one reset observation per episode plus one observation per step, starting from Gymnasium's count of 1e-4
    np.testing.assert_allclose(float(state.get("obs_rms").count), int(m["steps"]) + episodes + 1e-4, rtol=1e-6)
    # stored states are normalized: roughly zero mean, unit scale
    states = np.asarray(tc.trajectories().states)[np.asarray(tc.trajectories().mask)]
    assert np.all(np.abs(states.mean(axis=0)) < 1.0)


def test_without_obs_norm_the_stats_stay_untouched(make_pendulum_mc):
    mc = make_pendulum_mc(env_num_envs=1)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    env = mc.env
    state = ModelState.create(env.init_state(), {"policy": (policy, policy)})
    state, _, _ = env.execute_policy(state, jax.random.key(1), policy, 200, 200)
    np.testing.assert_allclose(float(state.get("obs_rms").count), 1e-4)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_env_wrapper.py tests/test_obs_norm.py -q`
Expected: FAIL. `init_state` and the new `execute_policy` signature don't exist yet.

- [ ] **Step 4: Rewrite `mcmc_env_wrapper.py`**

```python
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.jax_envs import Bounds, make_env
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.misc.running_mean_std import RunningMeanStd
from abstract_rl.misc.typing import DTYPE

if TYPE_CHECKING:
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration


def _lane_where(cond: Array, a: Any, b: Any) -> Any:
    """Per lane select between two pytrees with a leading lane axis."""
    return jax.tree.map(lambda x, y: jnp.where(cond.reshape(cond.shape + (1,) * (x.ndim - 1)), x, y), a, b)


class MCMCEnvWrapper:
    """
    Environment wrapper for JAX environments. Adds support for executing whole trajectories based on a policy in E
    lockstep lanes, instead of only giving a step based interface to the outside.
    """

    def __init__(self, mc: "ModelConfiguration") -> None:
        """
        Initialize a new environment.
        :param mc: The model configuration with everything important.
        """
        conf = mc["conf"]
        self.mc = mc
        self.num_threads = conf.get("num_threads", 1)
        self.render = conf.get("render", False)

        # save config
        self.conf = conf
        self.num_epochs = conf.get_root("num_epochs")
        self.env_name = conf["name"]
        self.env = make_env(self.env_name)
        # running observation statistics of NormalizeObservation, part of the traced state (init_state)
        self._obs_norm = bool(conf.to_dict().get("obs_norm", False))
        self._num_envs = conf.get("num_envs", "auto")
        self._seed = conf.get_root("seed")

        self._normalize = conf.get("normalize", False)
        self._discount = conf["discount"]

        self.cli = CliPrinter().instance
        self.obs_sp = Bounds(-self.env.obs_high, self.env.obs_high)
        self.act_sp = Bounds(self.env.action_low, self.env.action_high)

        # observations are scaled by the (finite) upper bound of the observation space if normalize is set
        high = np.asarray(self.obs_sp.high, dtype=np.float64)
        self._obs_scale = np.where(np.isfinite(high) & (high > 0), high, 1.0).astype(np.float32)

        # affine reward transform r' = reward_scale * r + reward_offset used for training only (evaluation scripts
        # report the raw environment return)
        self._reward_scale = conf.get("reward_scale", 1.0)
        self._reward_offset = conf.get("reward_offset", 0.0)

    def init_state(self) -> dict[str, Any]:
        """The traced statistics of this environment: observation normalisation and the two running filters."""
        return {
            "obs_rms": RunningMeanStd.create(self.observation_dim),
            "state_filter": RunningCenteredFilter("states", self.observation_dim),
            "reward_filter": RunningCenteredFilter("rewards", 1),
        }

    def discount(self) -> float:
        """
        Discount factor of the environment.
        :return: The discount factor used.
        """
        return self._discount

    def num_envs(self, batch_size: int, max_steps: int) -> int:
        """Number of lockstep lanes E: ``env_num_envs``, or ``ceil(batch_size / max_steps)`` for ``auto``."""
        if self._num_envs == "auto" or self._num_envs is None:
            return max(1, -(-batch_size // max_steps))
        return int(self._num_envs)

    def _preprocess(self, obs: Array, obs_rms: RunningMeanStd) -> Array:
        obs = obs_rms.normalize(obs) if self._obs_norm else jnp.asarray(obs, DTYPE)
        return obs / self._obs_scale if self._normalize else obs

    def execute_policy(
        self,
        state: ModelState,
        key: PRNGKeyArray,
        policy: Any,
        max_steps: int,
        batch_size: int,
        exploration: bool = True,
    ) -> tuple[ModelState, TrajectoryCollection, dict[str, Array]]:
        """
        Executes the policy for complete episodes (each at most max_steps long) in E lanes until at least batch_size
        transitions were collected. A lane resets after an episode while fewer than batch_size transitions are
        collected, otherwise it stops; no episode is cut.
        :param state: The traced model state (observation statistics and filters are read and updated).
        :param key: The random key of resets and action samples.
        :param policy: The policy to evaluate.
        :param max_steps: The maximum number of steps per episode.
        :param batch_size: The minimum number of transitions to collect.
        :param exploration: Sample from the policy if True, otherwise use its mode.
        :return: The new state, the trajectory collection and the rollout metrics.
        """
        e = self.num_envs(batch_size, max_steps)
        length = -(-batch_size // e) + max_steps
        sd, ad = self.observation_dim, self.action_dim
        obs_rms, state_filter, reward_filter = state.get("obs_rms"), state.get("state_filter"), state.get("reward_filter")

        key, k_reset = jax.random.split(key)
        env_state, raw = jax.vmap(self.env.reset)(jax.random.split(k_reset, e))
        active = jnp.ones(e, bool)
        if self._obs_norm:
            obs_rms = obs_rms.update(raw, active)
        cs = self._preprocess(raw, obs_rms)
        state_filter = state_filter.register(cs, active)

        def zeros(*shape: int, dtype: Any = DTYPE) -> Array:
            return jnp.zeros((e, length) + shape, dtype)

        buf = {
            "states": zeros(sd), "next_states": zeros(sd), "actions": zeros(ad), "rewards": zeros(1),
            "raw_rewards": zeros(), "dones": zeros(1), "log_likelihoods": zeros(1),
            "ends": zeros(dtype=bool), "mask": zeros(dtype=bool),
        }
        carry = (jnp.asarray(0), key, env_state, cs, jnp.zeros(e, jnp.int32), active, jnp.asarray(0), buf,
                 obs_rms, state_filter, reward_filter)

        def cond(c: Any) -> Array:
            i, _, _, _, _, active, *_ = c
            return (i < length) & jnp.any(active)

        def body(c: Any) -> Any:
            i, key, env_state, cs, t, active, count, buf, obs_rms, state_filter, reward_filter = c
            key, k_act, k_step, k_reset = jax.random.split(key, 4)

            # sample action and calc log likelihood
            suff_stats = policy.forward(cs)
            a = policy.mode(suff_stats=suff_stats) if not exploration else policy.sample_actions(k_act, suff_stats=suff_stats)
            a = a.reshape(e, ad)
            ll = policy.log_prob(a, suff_stats=suff_stats).reshape(e)

            stepped, raw_nxt, r, terminated, truncated = jax.vmap(self.env.step)(env_state, a, jax.random.split(k_step, e))
            if self._obs_norm:
                obs_rms = obs_rms.update(raw_nxt, active)
            nxt = self._preprocess(raw_nxt, obs_rms)
            state_filter = state_filter.register(nxt, active)
            r_train = self._reward_scale * r + self._reward_offset
            reward_filter = reward_filter.register(r_train[:, None], active)

            t = t + 1
            # only true terminations are stored as done, so truncated episodes can be bootstrapped
            end = terminated | truncated | (t >= max_steps)
            step_vals = {
                "states": cs, "next_states": nxt, "actions": a, "rewards": r_train[:, None], "raw_rewards": r,
                "dones": terminated.astype(DTYPE)[:, None], "log_likelihoods": ll[:, None], "ends": end, "mask": active,
            }
            buf = {k: _lane_where(active, buf[k].at[:, i].set(v), buf[k]) for k, v in step_vals.items()}
            count = count + jnp.sum(active)

            # finished lanes start a new episode while the batch is not complete, otherwise they stop
            finished = active & end
            cont = finished & (count < batch_size)
            active = active & ~(finished & (count >= batch_size))
            reset_state, reset_raw = jax.vmap(self.env.reset)(jax.random.split(k_reset, e))
            if self._obs_norm:
                obs_rms = obs_rms.update(reset_raw, cont)
            reset_cs = self._preprocess(reset_raw, obs_rms)
            state_filter = state_filter.register(reset_cs, cont)
            env_state = _lane_where(cont, reset_state, _lane_where(active, stepped, env_state))
            cs = jnp.where(cont[:, None], reset_cs, nxt)
            t = jnp.where(cont, 0, t)
            return (i + 1, key, env_state, cs, t, active, count, buf, obs_rms, state_filter, reward_filter)

        _, _, _, _, _, _, count, buf, obs_rms, state_filter, reward_filter = jax.lax.while_loop(cond, body, carry)
        trajectories = Trajectory(
            buf["states"], buf["next_states"], buf["actions"], buf["rewards"], buf["raw_rewards"], buf["dones"],
            buf["log_likelihoods"], buf["ends"], buf["mask"],
        )
        tc = TrajectoryCollection(self, trajectories)
        stats = tc.episode_stats()
        metrics = {
            "steps": count,
            "episode_returns": stats["returns"],
            "episode_valid": stats["valid"],
            "episode_lengths": stats["lengths"],
            "max_traj_len": jnp.max(jnp.where(stats["valid"], stats["lengths"], 0)),
        }
        state = state.set("obs_rms", obs_rms).set("state_filter", state_filter).set("reward_filter", reward_filter)
        return state, tc, metrics

    def execute_policy_once(
        self, state: ModelState, key: PRNGKeyArray, max_steps: int, policy: Any, opt: bool = False
    ) -> tuple[ModelState, Trajectory]:
        """
        Execute a policy once for the maximum number of steps or until the environment terminates.
        :return: The new state and the single-lane trajectory of the episode.
        """
        saved = self._num_envs
        self._num_envs = 1
        try:
            state, tc, _ = self.execute_policy(state, key, policy, max_steps, 1, exploration=not opt)
        finally:
            self._num_envs = saved
        return state, jax.tree.map(lambda x: x[0], tc.trajectories())

    def close(self) -> None:
        """JAX environments hold no resources."""

    @property
    def observation_space(self) -> Bounds:
        """
        Bounds for the observation space.
        :return: Bound of the observation space.
        """

        return self.obs_sp

    @property
    def action_space(self) -> Bounds:
        """
        Bounds for the action space.
        :return: Bound of the action space.
        """
        return self.act_sp

    @property
    def observation_dim(self) -> int:
        """
        Dimension of observation space.
        :return: Dimension of the observation space.
        """
        return int(self.env.observation_dim)

    @property
    def action_dim(self) -> int:
        """
        Dimension of action space.
        :return: Dimension of the action space.
        """
        return int(self.env.action_dim)
```

In `config/alg/trpo.yaml` and `config/alg/mpo.yaml`, add below `env_render: false`:

```yaml
# lockstep environment lanes of a rollout (auto: ceil(batch_size / max_steps))
env_num_envs: auto
```

Add `"abstract_rl.env.mcmc_env_wrapper"` to `JAXTYPED_MODULES`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_env_wrapper.py tests/test_obs_norm.py -q`
Expected: 7 passed. Then run the full suite: the policy tests built through `make_pendulum_mc` must still pass, now on
the gymnax env.

- [ ] **Step 6: Commit**

```bash
git add src/abstract_rl/env/mcmc_env_wrapper.py config/alg tests/test_env_wrapper.py tests/test_obs_norm.py tests/conftest.py docs/superpowers/specs/2026-10-22-jax-port-design.md
git commit -m "Roll out whole episodes in lockstep lanes inside one while loop.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The TD, GAE and Filter operators

**Files:**
- Modify: `src/abstract_rl/operator/trajectory_operator.py`, `td_operator.py`, `gae_operator.py`, `filter_operator.py` (whole files)
- Create: `tests/test_advantage_operators.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `TrajectoryOperator.transform(state, trajectory) -> Trajectory` (per lane) and
    `transform_all(state, trajectories: TrajectoryCollection) -> TrajectoryCollection`, which vmaps over lanes.
  - `TemporalDifferenceOperator` sets `"tv"` `[L]` and `"td"` `[L, 1]`.
  - `GeneralizedAdvantagesEstimationOperator` sets `"a"` `[L]`.
  - `FilterOperator` replaces states, next states and rewards once, using the state's filters.
  - Model lookups move from `__init__` to `transform` as `state.get(...)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_advantage_operators.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import golden, load_named, sub

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


def _ops(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = load_named(FullyVNeuralNet(mc, jax.random.key(0)), sub(golden("adv_v"), "params"))
    with mc["conf"].ns("tdop"):
        td = TemporalDifferenceOperator(mc)
    with mc["conf"].ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
    return ModelState.create({}, {"v_network": (v, v)}), td, gae


def _lane(g, pad=0):
    s = jnp.asarray(g["all_states"], jnp.float32)
    n = len(s) - 1
    p = lambda x: jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], x.dtype)])  # noqa: E731
    ends = jnp.zeros(n, bool).at[-1].set(True)
    return Trajectory(p(s[:-1]), p(s[1:]), p(jnp.zeros((n, 1))), p(jnp.asarray(g["rewards"], jnp.float32)),
                      p(jnp.asarray(g["rewards"][:, 0], jnp.float32)), p(jnp.asarray(g["dones"], jnp.float32)),
                      p(jnp.zeros((n, 1))), p(ends), p(jnp.ones(n, bool)))


@pytest.mark.parametrize("case", ["truncated", "terminated"])
def test_td_and_gae_match_torch(make_pendulum_mc, case):
    state, td, gae = _ops(make_pendulum_mc)
    g = golden(f"adv_{case}")
    n = len(g["rewards"])
    t = gae.transform(state, td.transform(state, _lane(g, pad=3)))
    np.testing.assert_allclose(t["tv"][:n], g["tv"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["td"][:n], g["td"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["a"][:n], g["a"], rtol=1e-5, atol=1e-5)


def test_episodes_in_one_lane_do_not_leak_into_each_other(make_pendulum_mc):
    state, td, gae = _ops(make_pendulum_mc)
    g1, g2 = golden("adv_terminated"), golden("adv_truncated")
    l1, l2 = _lane(g1), _lane(g2)
    both = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), l1, l2)
    n1 = len(g1["rewards"])
    t = gae.transform(state, td.transform(state, both))
    np.testing.assert_allclose(t["a"][:n1], g1["a"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["a"][n1:], g2["a"], rtol=1e-5, atol=1e-5)


def test_transform_all_vmaps_over_lanes(make_pendulum_mc):
    state, td, gae = _ops(make_pendulum_mc)
    g = golden("adv_truncated")
    lanes = jax.tree.map(lambda x: jnp.stack([x, x]), _lane(g))
    tc = jax.jit(lambda s, c: gae.transform_all(s, td.transform_all(s, c)))(state, TrajectoryCollection(None, lanes))
    np.testing.assert_allclose(tc.trajectories()["a"][1], g["a"], rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_advantage_operators.py -q`
Expected: FAIL, because `transform` takes one argument.

- [ ] **Step 3: Convert the operators**

`trajectory_operator.py`:

```python
import jax

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


class TrajectoryOperator:
    """
    A simple evaluation operator interface.
    """

    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory (one lane) with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectory: trajectory to transform.
        :return: The transformed trajectory.
        """

        raise NotImplementedError

    def transform_all(self, state: ModelState, trajectories: TrajectoryCollection) -> TrajectoryCollection:
        """
        Transform all lanes of a collection with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectories: trajectories to transform.
        """

        return trajectories.with_trajectories(
            jax.vmap(lambda trajectory: self.transform(state, trajectory))(trajectories.trajectories())
        )
```

`td_operator.py`. The class docstring and `__repr__` are unchanged, `__init__` no longer reads `v_net`, and
`transform` becomes:

```python
    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state (gives the target v network).
        :param trajectory: trajectory to transform.
        """

        v_net = state.get("v_network", True)

        # obtain rewards
        discount = self.env.discount()
        t_states = trajectory.states
        t_rewards = trajectory.rewards
        dones = trajectory.dones
        v = v_net.forward(t_states)
        mask = 1 - dones

        # bootstrap from the observed next state if an episode was truncated (e.g. time limit) instead of treating the
        # end of the trajectory as terminal; inside an episode the next value is v[t + 1]
        next_v = v_net.forward(trajectory.next_states) * mask
        td_errors = t_rewards + discount * next_v - v

        def step(tv_next: Array, xs: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
            r, m, last_v, end = xs
            tv_next = jnp.where(end, last_v, tv_next)
            tv = r + discount * m * tv_next
            return tv, tv

        _, tv = jax.lax.scan(step, jnp.zeros((), t_rewards.dtype), (t_rewards[:, 0], mask[:, 0], next_v[:, 0], trajectory.ends), reverse=True)

        return trajectory.set("tv", tv).set("td", td_errors)
```

Its imports become `import jax`, `import jax.numpy as jnp`, `from jaxtyping import Array`, `ModelState`, `Trajectory`
and `TrajectoryOperator`; `torch` is gone.

`gae_operator.py`. `__init__` stores `self.gae_lambda = float(self.conf["lambda"])` and no longer reads `v_net`, and
`transform` becomes:

```python
    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectory: trajectory to transform.
        """

        # obtain rewards
        discount = self.env.discount()

        # use v approximation to get advantages
        td_errors = trajectory["td"][:, 0]
        dones = trajectory.dones
        mask = 1 - dones[:, 0]

        # get combined ratio
        gae_gamma_lambda = discount * self.gae_lambda

        def step(a_next: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            td, m, end = xs
            # an advantage never crosses the end of an episode
            a_next = jnp.where(end, 0.0, a_next)
            a = td + m * gae_gamma_lambda * a_next
            return a, a

        _, a = jax.lax.scan(step, jnp.zeros((), td_errors.dtype), (td_errors, mask, trajectory.ends), reverse=True)
        return trajectory.set("a", a)
```

`filter_operator.py`. `transform` becomes:

```python
    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param state: The traced model state (gives the running filters).
        :param trajectory: trajectory to transform.
        """
        if self.filter_states and "unfiltered_states" not in trajectory:
            state_filter = state.get("state_filter")
            trajectory = trajectory.set("unfiltered_states", trajectory.states)
            trajectory = trajectory.replace(states=state_filter(trajectory.states), next_states=state_filter(trajectory.next_states))

        if self.filter_rewards and "unfiltered_rewards" not in trajectory:
            reward_filter = state.get("reward_filter")
            trajectory = trajectory.set("unfiltered_rewards", trajectory.rewards)
            trajectory = trajectory.replace(rewards=reward_filter(trajectory.rewards))
        return trajectory
```

Add the four operator modules to `JAXTYPED_MODULES`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_advantage_operators.py -q`
Expected: 4 passed. Then run the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/abstract_rl/operator tests/test_advantage_operators.py tests/conftest.py
git commit -m "Port the TD, GAE and filter operators to per-lane reverse scans.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `TrustRegionGradient` and `TRPOAlgorithm`

**Files:**
- Modify: `src/abstract_rl/algorithms/mixed/trust_region_gradient.py` (whole file)
- Modify: `src/abstract_rl/algorithms/continuous/policy_gradient/trust_region_policy_optimization.py` (whole file)
- Modify: `src/abstract_rl/algorithms/continuous/policy_gradient/policy_optimization_algorithm.py`. Remove the unused
  torch `obj` (spec 5) and keep the class with its `__init__`.
- Create: `tests/test_trpo.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `TrustRegionGradient.__init__(model: str, tr_eps, cg_max_k=10, cg_residual_tol=1e-6, dvp_damping=1e-5, mode="min")`,
    where `model` is the ModelState key of the optimised model;
  - `perform_tr_sgd(state, curst, batch, tr_batch) -> (state, metrics)`;
  - `line_search(state, batch, x, fullstep, exp_improv_rate, max_steps=10, accept_ratio=0.1, max_kl_ratio=1.5) -> (accepted, step, lr, obj, x_final, rows)`;
  - `tr_obj(state, batch)` and `tr_distance(state, batch)`, both abstract.
  - The metrics are `{"accepted", "step", "lr", "rows": Float[10, 8]}`. The rows are the torch line-search columns,
    NaN for steps that never ran.
  - `TRPOAlgorithm(mc)` implements `tr_distance` as the mean m-projection KL of `state.get("policy")` and `tr_obj` as
    its `likelihood_ratio`.

- [ ] **Step 1: Write the failing tests**

`tests/test_trpo.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import golden, load_named, named_parameters, sub

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy


def _setup(make_pendulum_mc, case, env):
    mc = make_pendulum_mc(env_name=env)
    with mc["conf"].ns("policy"):
        policy = load_named(select_policy(mc, jax.random.key(0)), sub(golden(case), "before/params"))
    with mc["conf"].ns("trpo"):
        alg = TRPOAlgorithm(mc)
    g = golden(case)
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(jnp.asarray(g["states"]), jnp.asarray(g["actions"]), jnp.asarray(g["lls"]), z, jnp.asarray(g["states"]), z,
                  {"a": jnp.asarray(g["a"])})
    return alg, ModelState.create({}, {"policy": (policy, policy)}), batch, g


@pytest.mark.parametrize("case, env", [("trpo_uni", "Pendulum-v1"), ("trpo_diag", "TestPointMass-v0")])
def test_trpo_step_matches_torch(make_pendulum_mc, case, env):
    alg, state, batch, g = _setup(make_pendulum_mc, case, env)
    new_state, m = jax.jit(lambda s, b: alg.perform_tr_sgd(s, 0, b, b))(state, batch)
    got = named_parameters(new_state.get("policy"))
    before, after = sub(g, "before/params"), sub(g, "after/params")
    torch_accepted = any(not np.allclose(before[k], after[k]) for k in after)
    assert bool(m["accepted"]) == torch_accepted
    for k, expected in after.items():
        np.testing.assert_allclose(got[k], expected, rtol=1e-3, atol=1e-5, err_msg=k)
    # the target is untouched
    np.testing.assert_allclose(named_parameters(new_state.get("policy", True))["network.layer_0.weight"],
                               before["network.layer_0.weight"])


def test_failed_line_search_keeps_the_parameters(make_pendulum_mc):
    alg, state, batch, g = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    # zero advantages give no expected improvement, every step is rejected
    zero = Batch(batch.states, batch.actions, batch.log_likelihoods, batch.rewards, batch.next_states, batch.dones,
                 {"a": jnp.zeros_like(batch["a"]) + jnp.arange(len(batch)) * 0.0})
    new_state, m = alg.perform_tr_sgd(state, 0, zero, zero)
    assert not bool(m["accepted"])
    np.testing.assert_allclose(named_parameters(new_state.get("policy"))["cov"], named_parameters(state.get("policy"))["cov"])


def test_objective_gradient_flows_through_the_state(make_pendulum_mc):
    alg, state, batch, _ = _setup(make_pendulum_mc, "trpo_uni", "Pendulum-v1")
    import equinox as eqx

    grads = eqx.filter_grad(lambda p: alg.tr_obj(state.set("policy", p), batch))(state.get("policy"))
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree.leaves(grads))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_trpo.py -q`
Expected: FAIL. The torch `perform_tr_sgd` signature and its torch types differ.

- [ ] **Step 3: Convert `trust_region_gradient.py`**

```python
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to


class TrustRegionGradient:
    """Add trust region gradient based optimization to any class by inheriting this one."""

    def __init__(
        self,
        model: str,
        tr_eps: float,
        cg_max_k: int = 10,
        cg_residual_tol: float = 1e-6,
        dvp_damping: float = 1e-5,
        mode: str = "min",
    ) -> None:
        """Should be self explanatory

        :param model: The model state key of the model to optimize
        :param tr_eps: The trust region to use.
        :param cg_max_k: The maximum number of iterations for the conjugate gradient iteration.
        :param cg_residual_tol: The residual tolerance when executing the conjugate gradient.
        :param dvp_damping: The damping to use for the FIM
        :param mode: The mode in which to run the optimization.
        """
        self._model = model
        self.tr_eps = tr_eps
        self.cg_max_k = cg_max_k
        self.cg_residual_tol = cg_residual_tol
        self.dvp_damping = dvp_damping
        self.mode = mode

        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def _at(self, state: ModelState, model: Any, flat: Float[Array, " n"]) -> ModelState:
        return state.set(self._model, set_flat_params_to(model, flat))

    def perform_tr_sgd(self, state: ModelState, curst: int, batch: Any, tr_batch: Any) -> tuple[ModelState, dict[str, Array]]:

        # get v objective and the gradient
        model = state.get(self._model)
        x = get_flat_params_from(model)
        old_obj, g = jax.value_and_grad(lambda flat: self.tr_obj(self._at(state, model, flat), batch))(x)

        # fisher vector products using direct mehtod (forward over reverse hessian vector product)
        def fvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
            grad_f = jax.grad(lambda flat: self.tr_distance(self._at(state, model, flat), tr_batch))
            r_hvp = jax.jvp(grad_f, (x,), (v,))[1]
            r_vp = r_hvp + self.dvp_damping * v
            return r_vp

        # calc natural gradient and set gradient for line search
        s = conjugate_gradient(fvp, -g, self.cg_max_k, self.cg_residual_tol)
        sdvps = (s * fvp(s)).sum()
        nat_metric = 0.5 * sdvps
        real_eps = self.tr_eps
        lm = jnp.sqrt(nat_metric / real_eps)

        # finish by line search
        neggdotstepdir = (-g * s).sum()
        accepted, step, lr, _, x_final, rows = self.line_search(state, batch, x, s / lm, neggdotstepdir / lm)
        state = self._at(state, model, x_final)
        return state, {"accepted": accepted, "step": step, "lr": lr, "rows": rows}

    def line_search(
        self,
        state: ModelState,
        batch: Any,
        x: Float[Array, " n"],
        fullstep: Float[Array, " n"],
        exp_improv_rate: Float[Array, ""],
        max_steps: int = 10,
        accept_ratio: float = 0.1,
        max_kl_ratio: float = 1.5,
    ) -> tuple[Bool[Array, ""], Int[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, " n"], Float[Array, "s 8"]]:
        """Perform a line search.

        :param batch: The batch to use for the objective.
        :param fullstep: The fullstep, e.g the learing rate and the gradient
        :param exp_improv_rate: The expected improvement rate for the line search.
        :param max_steps: The maximum number of steps for the line search.
        :param accept_ratio: The acceptance ratio between actual and expected improvement rate.
        :param max_kl_ratio: Steps whose trust region distance exceeds max_kl_ratio * tr_eps are rejected, the
            quadratic approximation of the distance can underestimate it.
        :return: accepted, step, lr, objective, the final flat parameters (x if no step was accepted) and the printed
            table rows ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv", "distance").
        """
        model = state.get(self._model)
        old_obj = jax.lax.stop_gradient(self.tr_obj(self._at(state, model, x), batch))
        rows = jnp.full((max_steps, 8), jnp.nan, old_obj.dtype)

        # start with the learning rate continuously decay until the objective got better
        def cond(c: Any) -> Array:
            step, _, accepted, _, _ = c
            return (step < max_steps) & ~accepted

        def body(c: Any) -> Any:
            step, lr, _, _, rows = c
            # calc difference and revert this difference
            x_new = x + lr * fullstep
            new_state = self._at(state, model, x_new)

            # update objective and learnrate
            new_obj = self.tr_obj(new_state, batch)
            act_improv = old_obj - new_obj
            exp_improv = lr * exp_improv_rate
            ratio = act_improv / exp_improv
            distance = self.tr_distance(new_state, batch)
            rows = rows.at[step].set(jnp.stack([step.astype(old_obj.dtype), old_obj, new_obj, lr, ratio, act_improv, exp_improv, distance]))

            accepted = (ratio > accept_ratio) & (act_improv > 0) & (distance <= max_kl_ratio * self.tr_eps)
            next_step = jnp.where(accepted, step, step + 1)
            next_lr = jnp.where(accepted, lr, lr / 2)
            return next_step, next_lr, accepted, jnp.where(accepted, new_obj, old_obj), rows

        init = (jnp.asarray(0), jnp.asarray(1.0, old_obj.dtype), jnp.asarray(False), old_obj, rows)
        step, lr, accepted, obj, rows = jax.lax.while_loop(cond, body, init)

        # if line search was not successful the parameters stay at x
        x_final = jnp.where(accepted, x + lr * fullstep, x)
        return accepted, step, lr, obj, x_final, rows

    def tr_distance(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """The training distance."""
        raise NotImplementedError

    def tr_obj(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """The training objective."""
        raise NotImplementedError
```

- [ ] **Step 4: Convert `trust_region_policy_optimization.py`**

```python
from typing import Any

from jaxtyping import Array, Float

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.trust_region_gradient import TrustRegionGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState


class TRPOAlgorithm(PolicyOptimizationAlgorithm, TrustRegionGradient):
    """
    (class docstring unchanged)
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        PolicyOptimizationAlgorithm.__init__(self, mc)
        conf = mc.get("conf")

        # trpo vars
        trpo_eps = conf["tr_eps"]
        trpo_cg_max_k = conf["cg_max_k"]
        trpo_cg_residual_tol = conf["cg_res_tol"]
        trpo_dvp_damping = conf["dvp_damping"]

        TrustRegionGradient.__init__(
            self, "policy", trpo_eps, trpo_cg_max_k, trpo_cg_residual_tol, trpo_dvp_damping, "max"
        )

    def tr_distance(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """Pass back KL divergence for training divergence.

        :param state: The traced state holding the (candidate) policy and its target.
        :param batch: The batch to use.
        :return: The KL divergence.
        """
        return state.get("policy").kl_divergence(state, batch, est="mean", mode="m-projection")

    def tr_obj(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """Pass back the likelihood ratio gradient.

        :param state: The traced state holding the (candidate) policy.
        :param batch: The batch to use.
        :return: The likelihood ratio gradient
        """
        return state.get("policy").likelihood_ratio(state, batch)  # - self.trpo_entropy * self.policy.entropy(batch)
```

In `policy_optimization_algorithm.py`, delete the `obj` method and its `torch`/`Iterable`/`Trajectory` imports, and
keep `__init__`. Add the three modules to `JAXTYPED_MODULES`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trpo.py -q`
Expected: 4 passed. If a parameter misses rtol 1e-3, compare the intermediate `g`, `s` and `lm` against a torch
re-computation. Do it in a worktree of 4f7828e, not by loosening the tolerance. Report the measured error.

Then run the full suite. The torch MPO algorithm imports `PolicyOptimizationAlgorithm`, and it must still import.

- [ ] **Step 6: Commit**

```bash
git add src/abstract_rl/algorithms tests/test_trpo.py tests/conftest.py
git commit -m "Port the TRPO step to a jitted conjugate gradient and line search.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Equinox checkpoints, `SeedRuns`, run-config keys and the platform switch

**Files:**
- Modify: `src/abstract_rl/data_structures/abstract_conf/model_configuration.py` (`store`, plus removing `load`'s
  torch use)
- Create: `src/abstract_rl/run/run_seeds.py`
- Modify: `config/config.yaml`, `config/alg/trpo.yaml`, `config/alg/mpo.yaml`,
  `src/abstract_rl/scripts/hydra_conf.py` (`run_dict`), `src/abstract_rl/scripts/run.py` (`train`) and
  `src/abstract_rl/misc/mixed.py` (`default_settings`)
- Create: `tests/test_run_seeds.py`
- Modify: `tests/test_hydra_config.py` (append), `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `ModelConfiguration.store(name, models: dict[str, Any] | None = None)`. It writes `run.conf` with the real `seed`
    (D3), then `model/<name>/<key>.eqx` via `eqx.tree_serialise_leaves` for each model (default: the online syncables),
    then `model/obs_rms.npz` when `models` has an `obs_rms` entry.
  - `ModelConfiguration.load` reads `.eqx` files into the registered models with `eqx.tree_deserialise_leaves`.
  - `store_best`, unchanged.
- Produces, `SeedRuns(conf, alg: str)`:
  - `seeds: list[int]` (conf `seeds`, else `[seed]`), `mcs: list[ModelConfiguration]` (one run directory and
    `DataLogger` each), `mc` (= `mcs[0]`, used for wiring) and `keys` (typed keys `[S]`);
  - `build(fn: Callable[[key], T]) -> T` (vmapped over the seed keys) and `broadcast(tree) -> tree` (adds the seed axis);
  - `state() -> ModelState` (`mc.state()` with target arrays copied and `overall_step` `[S]`);
  - `seed_slice(tree, i) -> tree` and `store(name, models_per_seed: dict[str, Any])`, which runs asynchronously in one
    writer thread in submission order;
  - `store_best(i, ret, epoch)`, queued after the stores, and `close()`, which joins the writer.
- Produces the config keys `seeds: null` and `jax_platform: cpu` (root), and `epochs_per_call: 1` (alg).

- [ ] **Step 1: Write the failing tests**

`tests/test_run_seeds.py`:

```python
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_seeds import SeedRuns


def test_each_seed_has_its_own_run_directory(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [3, 7]}, "trpo")
    assert runs.seeds == [3, 7] and len({mc.run_root for mc in runs.mcs}) == 2
    with runs.mc["conf"].ns("policy"):
        policies = runs.build(lambda k: select_policy(runs.mc, k))
    assert policies.network.layers[0].weight.shape[0] == 2
    assert not np.allclose(policies.network.layers[0].weight[0], policies.network.layers[0].weight[1])
    runs.store("last", {"policy": policies})
    runs.store_best(0, -5.0, 0)
    runs.close()
    for i, mc in enumerate(runs.mcs):
        conf = json.loads(open(f"{mc.run_root}/run.conf").read())
        assert conf["seed"] == runs.seeds[i]  # D3: the real seed, not a random overwrite
        with runs.mc["conf"].ns("policy"):
            fresh = select_policy(runs.mc, jax.random.key(0))
        loaded = eqx.tree_deserialise_leaves(f"{mc.run_root}/model/last/policy.eqx", fresh)
        np.testing.assert_allclose(loaded.network.layers[0].weight, policies.network.layers[0].weight[i])
    assert (runs.mcs[0].model_root + "/best.json") and json.loads(open(runs.mcs[0].model_root + "/best.json").read())["epoch"] == 0


def test_state_has_distinct_target_buffers(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [1, 2]}, "trpo")
    with runs.mc["conf"].ns("policy"):
        policies = runs.build(lambda k: select_policy(runs.mc, k))
        runs.mc.add_syncable("policy", policies, 1)
    st = runs.state()
    assert st.overall_step.shape == (2,)
    online, target = st.get("policy"), st.get("policy", True)
    assert online.network.layers[0].weight is not target.network.layers[0].weight
    # donation of the whole state must not hit "donate the same buffer twice"
    jax.jit(lambda s: s.sync(), donate_argnums=0)(st)
    runs.close()


def test_broadcast_adds_the_seed_axis(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [1, 2, 3]}, "trpo")
    out = runs.broadcast({"x": jnp.zeros(4)})
    assert out["x"].shape == (3, 4)
    runs.close()
```

Append to `tests/test_hydra_config.py`:

```python
def test_jax_run_keys():
    conf = run_dict(load(["env=pendulum", "alg=trpo", "seeds=[1,2,3]"]))
    assert conf["seeds"] == [1, 2, 3] and conf["jax_platform"] == "cpu" and conf["epochs_per_call"] == 1
    assert run_dict(load(["env=pendulum", "alg=trpo"]))["seeds"] is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_run_seeds.py tests/test_hydra_config.py -q`
Expected: FAIL. `abstract_rl.run.run_seeds` is missing, and so is `seeds` in the config.

- [ ] **Step 3: Update the config and `run_dict`**

Add to `config/config.yaml`, below `seed: 1`:

```yaml
# several seeds trained at once in one vmapped program (null: just `seed`), e.g. seeds=[1,2,3,4]
seeds: null
# JAX backend of a run: cpu (default) or mps (Metal, needs the `mps` extra; MJX environments only run on cpu)
jax_platform: cpu
```

Add to `config/alg/trpo.yaml` and `config/alg/mpo.yaml`:

```yaml
# epochs per jitted call (logging and checkpoints happen once per call)
epochs_per_call: 1
```

In `run_dict`, after `conf["seed"] = cfg.seed`:

```python
    conf["seeds"] = list(cfg.seeds) if cfg.get("seeds") is not None else None
    conf["jax_platform"] = cfg.get("jax_platform", "cpu")
```

In `scripts/run.py`, make `train` select the platform before anything imports jax:

```python
def train(alg: str, conf: dict[str, Any]) -> Any:
    """Run one training. Returns the logged ``[mean, std]`` return of every seed and epoch."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", conf.get("jax_platform", "cpu"))
    if alg == "mpo":
        ...  # unchanged
```

In `misc/mixed.py`, `default_settings` drops `torch.set_num_threads`. Keep the parameter, so callers don't change, and
the numpy print options. Remove the `torch` import.

- [ ] **Step 4: Convert `ModelConfiguration.store` and `load`**

Add `import equinox as eqx`. Remove `import torch` and `np.random.randint`. Then:

```python
    def load(self) -> int:
        """Load the model configuration."""

        conf = self["conf"]
        self.overall_step = conf["overall_step"]

        for k in self.syncables:
            s, model, target = self.syncables[k]
            model = eqx.tree_deserialise_leaves(join(self.run_root, "model/last", f"{k}.eqx"), model)
            self.syncables[k] = (s, model, target)

        return self.overall_step

    def store(self, name: str, models: dict[str, Any] | None = None) -> None:
        """Store the model configuration and the (unbatched) models of this run, by default the online syncables."""

        # store the config as a json file
        dconf = self["conf"].to_dict()
        dconf["overall_step"] = self.overall_step

        r = json.dumps(dconf, indent=4, sort_keys=True)
        with open(join(self.run_root, "run.conf"), "w") as file_conf:
            file_conf.write(r)

        models = dict(models) if models is not None else {k: v[1] for k, v in self.syncables.items()}
        obs_rms = models.pop("obs_rms", None)
        if obs_rms is not None:
            np.savez(join(self.model_root, "obs_rms.npz"), mean=np.asarray(obs_rms.mean), var=np.asarray(obs_rms.var),
                     count=np.asarray(obs_rms.count, np.float64))
        model_path = join(self.model_root, name)
        if not exists(model_path):
            mkdir(model_path)
        for k, model in models.items():
            eqx.tree_serialise_leaves(join(model_path, f"{k}.eqx"), model)
```

- [ ] **Step 5: Write `run/run_seeds.py`**

```python
"""Several seeds of one run configuration, trained as one vmapped program (spec P6)."""

import copy
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.misc.data_logger import DataLogger

T = TypeVar("T")


class SeedRuns:
    """One run directory, model configuration and logger per seed; models are built and trained batched over seeds."""

    def __init__(self, conf: dict[str, Any], alg: str) -> None:
        seeds = conf.get("seeds") or [conf["seed"]]
        self.seeds = [int(s) for s in seeds]
        self.mcs: list[ModelConfiguration] = []
        for s in self.seeds:
            c = copy.deepcopy(conf)
            c["seed"] = s
            c.setdefault("alg", alg)
            mc = ModelConfiguration(c)
            mc.add_main("log", DataLogger(mc.data_dir))
            self.mcs.append(mc)
        self.mc = self.mcs[0]
        self.keys = jax.vmap(jax.random.key)(jnp.asarray(self.seeds, jnp.uint32))
        # checkpoints are written by one background thread, in submission order (spec P5)
        self._writer = ThreadPoolExecutor(max_workers=1)

    def build(self, fn: Callable[[Any], T]) -> T:
        """``fn(key)`` for every seed, batched along a leading seed axis."""
        return eqx.filter_vmap(fn)(self.keys)

    def broadcast(self, tree: T) -> T:
        """The same value for every seed, with a leading seed axis."""
        n = len(self.seeds)
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)

    def state(self) -> ModelState:
        """The batched traced state of the wiring configuration, with target arrays in their own buffers."""
        st = self.mc.state()
        syncables = {k: (o, jax.tree.map(jnp.copy, t)) for k, (o, t) in st.syncables.items()}
        st = eqx.tree_at(lambda s: s.syncables, st, syncables)
        return eqx.tree_at(lambda s: s.overall_step, st, jnp.zeros(len(self.seeds), jnp.int32))

    @staticmethod
    def seed_slice(tree: T, i: int) -> T:
        """The unbatched value of seed ``i``."""
        return jax.tree.map(lambda x: x[i], tree)

    def store(self, name: str, models_per_seed: dict[str, Any]) -> None:
        """Store the batched models of every seed into its own run directory (asynchronously)."""
        host = jax.device_get(models_per_seed)
        for i, mc in enumerate(self.mcs):
            models = {k: self.seed_slice(v, i) for k, v in host.items()}
            self._writer.submit(mc.store, name, models)

    def store_best(self, i: int, ret: float, epoch: int) -> None:
        """Keep the last stored models of seed ``i`` as its best ones if ``ret`` is a new best return."""
        self._writer.submit(self.mcs[i].store_best, ret, epoch)

    def close(self) -> None:
        """Wait for all pending checkpoint writes."""
        self._writer.shutdown(wait=True)
```

Add `"abstract_rl.run.run_seeds"` and `"abstract_rl.data_structures.abstract_conf.model_configuration"` to
`JAXTYPED_MODULES`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_seeds.py tests/test_hydra_config.py tests/test_config.py -q`
Expected: all passed. Then run the full suite.

- [ ] **Step 7: Commit**

```bash
git add src/abstract_rl/data_structures/abstract_conf/model_configuration.py src/abstract_rl/run/run_seeds.py config \
    src/abstract_rl/scripts/hydra_conf.py src/abstract_rl/scripts/run.py src/abstract_rl/misc/mixed.py \
    tests/test_run_seeds.py tests/test_hydra_config.py tests/conftest.py
git commit -m "Run seeds side by side with per-seed run directories and Equinox checkpoints.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `run_trpo` as a jitted, vmapped epoch

**Files:**
- Modify: `src/abstract_rl/run/run_trpo.py` (whole file)
- Modify: `tests/test_training_smoke.py` (re-enable the two TRPO smoke tests and update their checkpoint glob)
- Create: `tests/test_run_trpo.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: all earlier tasks.
- Produces:
  - `run_trpo(conf) -> np.ndarray` of shape `[S, epochs_run, 2]` (mean and std of the raw return per seed and epoch);
  - the module-level `EPOCH_TRACES: list[int]`, to which one `batch_size` entry is appended per trace.

- [ ] **Step 1: Write the failing tests**

`tests/test_run_trpo.py`:

```python
import json

import numpy as np

from abstract_rl.run import run_trpo as run_trpo_module
from abstract_rl.run.run_trpo import run_trpo


def _conf(trpo_conf, **overrides):
    return {**trpo_conf, "env_name": "TestPointMass-v0", "max_steps": 30, "policy_batch_size": 120, "val_opt_steps": 5,
            **overrides}


def test_returns_per_seed_and_epoch(trpo_conf, tmp_path):
    returns = run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=3))
    assert returns.shape == (2, 3, 2) and np.all(np.isfinite(returns))
    runs = sorted(tmp_path.glob("TestPointMass-v0/trpo/*"))
    assert len(runs) == 2
    assert sorted(json.loads((r / "run.conf").read_text())["seed"] for r in runs) == [1, 2]
    for r in runs:
        assert (r / "model/last/policy.eqx").exists() and (r / "model/best/policy.eqx").exists()
        assert len((r / "data/sreward.txt").read_text().split()) == 3 * 2


def test_epoch_is_traced_once(trpo_conf):
    run_trpo_module.EPOCH_TRACES.clear()
    run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=3))
    assert run_trpo_module.EPOCH_TRACES == [120]


def test_vmapped_seeds_match_single_seed_runs(trpo_conf):
    both = run_trpo(_conf(trpo_conf, seeds=[1, 2], num_epochs=2))
    alone = run_trpo(_conf(trpo_conf, seeds=[2], num_epochs=2))
    np.testing.assert_allclose(both[1], alone[0], rtol=1e-3, atol=1e-3)


def test_env_step_budget_shortens_the_last_epoch(trpo_conf):
    returns = run_trpo(_conf(trpo_conf, num_epochs=10, max_env_steps=300))
    assert returns.shape[1] == 3  # 120 + 120 + a shortened third epoch reach the 300 step budget


def test_epochs_per_call(trpo_conf):
    returns = run_trpo(_conf(trpo_conf, num_epochs=4, epochs_per_call=2))
    assert returns.shape == (1, 4, 2) and np.all(np.isfinite(returns))
```

In `tests/test_training_smoke.py`, remove the skip marks from `test_trpo_pendulum_smoke` and
`test_trpo_with_two_dimensional_actions_smoke`. In both, change `model/last/policy.data` to `model/last/policy.eqx`.
Update `_check_returns` and `_check_multi_dimensional` to accept the seed axis by checking `returns[0]`. The MPO and
eval tests keep their Plan 3 skips.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_run_trpo.py -q`
Expected: FAIL. The torch `run_trpo` calls the old APIs.

- [ ] **Step 3: Rewrite `run_trpo.py`**

```python
from functools import partial
from os.path import join
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_seeds import SeedRuns
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet

# batch sizes the jitted epoch was traced for (a second epoch with the same batch size must not retrace)
EPOCH_TRACES: list[int] = []


# title of the script
def run_trpo(conf: Any) -> np.ndarray:
    """
    Runs the passed config on policy, every seed of ``seeds`` at once.
    :param conf: The run configuration.
    :return: The logged [mean, std] return of every seed and epoch, shape [seeds, epochs, 2].
    """

    # create a model configuration per seed, the first one wires the shared components
    conf = dict(conf)
    conf.setdefault("alg", "trpo")
    runs = SeedRuns(conf, "trpo")
    mc = runs.mc
    conf = mc["conf"]
    default_settings(conf.get_root("torch_threads") if "torch_threads" in conf else 1)
    cli = CliPrinter().instance

    num_epochs = conf["num_epochs"]
    max_steps = conf["max_steps"]
    epochs_per_call = int(conf.get("epochs_per_call", 1))

    # create an environment wrapper
    with conf.ns("env"):
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)
    for k, v in env.init_state().items():
        mc.add_main(k, runs.broadcast(v))
    mc.add_main("key", runs.keys)

    # define namespace for v
    with conf.ns("val"):
        sync_steps = conf["sync_steps"]
        v_opt_steps = conf["opt_steps"]
        v_batch_size = conf["batch_size"]
        v_network = runs.build(lambda key: FullyVNeuralNet(mc, jax.random.fold_in(key, 1)))
        mc.add_syncable("v_network", v_network, sync_steps)
        mc.add_main("v_network_opt", jax.vmap(lambda v: v.init_opt())(v_network))

    # policy neural_modules
    with conf.ns("policy"):
        policy_batch_size = conf["batch_size"]
        sync_steps = conf["sync_steps"]
        policy = runs.build(lambda key: select_policy(mc, jax.random.fold_in(key, 2)))
        mc.add_syncable("policy", policy, sync_steps)

    with conf.ns("filter"):
        filter = FilterOperator(mc)
        mc.add_main("fil", filter)

    # create algo
    with conf.ns("tdop"):
        td_op = TemporalDifferenceOperator(mc)
        mc.add_main("td", td_op)

    with conf.ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
        mc.add_main("gae", gae)

    alg_name = conf["alg"]
    with conf.ns(alg_name):
        alg = TRPOAlgorithm(mc)
        mc.add_main("alg", alg)

    def epoch(state: ModelState, batch_size: int) -> tuple[ModelState, dict[str, Any]]:
        """One epoch of one seed: rollout, advantages, trust region step on the policy, fit of v, sync."""
        key, k_rollout, k_v = jax.random.split(state.get("key"), 3)
        pre = {"policy": state.get("policy"), "v_network": state.get("v_network"), "obs_rms": state.get("obs_rms")}

        # test policy
        state, tc, rollout = env.execute_policy(state, k_rollout, state.get("policy"), max_steps, batch_size, exploration=True)
        tc = filter.transform_all(state, tc)

        # calc value fns and policies: estimate v values by sample return
        tc = td_op.transform_all(state, tc)
        tc = gae.transform_all(state, tc)
        batch = tc.to_batch(["a"])
        state, p_step = alg.perform_tr_sgd(state, 0, batch, batch)

        # v-step
        state, v_step = state.get("v_network").perform_mult_sgd(state, k_v, v_opt_steps, tc, v_batch_size, ["tv"])

        # copy target_networks
        state = state.sync()
        state = state.set("key", key)
        return state, {"rollout": rollout, "p_step": p_step, "v_step": v_step, "pre": pre}

    def chunk(state: ModelState, batch_size: int, num: int) -> tuple[ModelState, dict[str, Any]]:
        EPOCH_TRACES.append(batch_size)
        state, metrics = jax.lax.scan(lambda s, _: epoch(s, batch_size), state, None, length=num)
        return state, metrics

    compiled: dict[tuple[int, int], Any] = {}

    def run_chunk(state: ModelState, batch_size: int, num: int) -> tuple[ModelState, dict[str, Any]]:
        if (batch_size, num) not in compiled:
            compiled[(batch_size, num)] = jax.jit(jax.vmap(partial(chunk, batch_size=batch_size, num=num)), donate_argnums=0)
        return compiled[(batch_size, num)](state)

    # names for rewards
    stoch_reward = "sreward"

    # create a new reward plot and the logged fields of every seed
    rew_plots = [RewardValueFunctionPlot() for _ in runs.mcs]
    for seed_mc in runs.mcs:
        seed_mc["log"].create_field(stoch_reward, 2)
        # number of environment steps collected in each epoch (for plotting learning curves over samples)
        seed_mc["log"].create_field("steps", 1)

    state = runs.state()
    best_reward = np.full(len(runs.seeds), np.finfo(np.float64).min)

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = conf.get("max_env_steps", None)
    epoch_steps = policy_batch_size
    total_steps = np.zeros(len(runs.seeds), np.int64)
    e = 0
    while e < num_epochs:
        # initial print of the episode
        if e % 100 == 0:
            cli.new_epoch(mc.run_root, e)
        cli.big_header(f"epoch {e}")
        num = min(epochs_per_call, num_epochs - e)
        policy_batch_size = epoch_steps
        if max_env_steps:
            done_steps = int(total_steps.max())
            if done_steps >= max_env_steps:
                break
            policy_batch_size = min(epoch_steps, max_env_steps - done_steps)
            if policy_batch_size < epoch_steps:
                num = 1

        state, metrics = run_chunk(state, policy_batch_size, num)
        metrics = jax.device_get(metrics)

        for c in range(num):
            with cli.header("stochastic evaluation", 1):
                for i, seed_mc in enumerate(runs.mcs):
                    rollout = jax.tree.map(lambda x, i=i, c=c: x[i, c], metrics["rollout"])
                    returns = rollout["episode_returns"][rollout["episode_valid"]]
                    # the logged return is the one of the environment, not of the (optionally scaled) training reward
                    mean_return, std_return = float(np.mean(returns)), float(np.std(returns))
                    best_reward[i] = max(best_reward[i], mean_return)
                    seed_mc["log"].log({stoch_reward: [mean_return, std_return], "steps": [int(rollout["steps"])]})
                    total_steps[i] += int(rollout["steps"])
                    cli.print(f"seed {runs.seeds[i]:5d} | episodes {len(returns):4d} | steps {int(rollout['steps']):7d} | "
                              f"return {mean_return:12.4f} +- {std_return:10.4f}")

            with cli.header("store model & generate plots", 1):
                for i, seed_mc in enumerate(runs.mcs):
                    # make and store reward plot
                    res_log = seed_mc["log"].get([stoch_reward], e + 1)
                    rew_plots[i].update(*res_log, int(np.sum(metrics["rollout"]["episode_valid"][i, c])))  # type: ignore[call-arg]
                    rew_plots[i].save(join(seed_mc.plot_root, "performance.png"))

            with cli.header("p-step", True):
                rows = metrics["p_step"]["rows"][0, c]
                for row in rows[~np.isnan(rows[:, 0])]:
                    cli.print("%10i | %14.10f | %14.10f | %10.5f | %14.10f | %14.10f | %14.10f | %10.6f" % tuple(row))
                if not bool(metrics["p_step"]["accepted"][0, c]):
                    cli.line().print("FAILED")
            e += 1

        # save last model (the models the last rollout ran with) and keep it as best model if its return is the highest
        last = num - 1
        for seed_mc in runs.mcs:
            seed_mc.overall_step = e
        runs.store("last", jax.tree.map(lambda x: x[:, last], metrics["pre"]))
        for i, seed_mc in enumerate(runs.mcs):
            runs.store_best(i, float(seed_mc["log"].get([stoch_reward], e)[0][e - 1, 0]), e - 1)
        cli.empty().print("last model saved")

    runs.close()
    for plot in rew_plots:
        plot.close()
    return np.stack([seed_mc["log"].get([stoch_reward], e)[0] for seed_mc in runs.mcs])
```

`store_best` compares the return of the *last* epoch of each chunk. With `epochs_per_call: 1`, that is exactly torch's
per-epoch behaviour.

Add `"abstract_rl.run.run_trpo"` to `JAXTYPED_MODULES`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_trpo.py tests/test_training_smoke.py -q`
Expected: the new tests and the two TRPO smoke tests pass, and the MPO/eval smokes stay skipped. Compiles take tens of
seconds.

Then run the full suite and `uv run ruff check src tests`.

- [ ] **Step 5: Commit**

```bash
git add src/abstract_rl/run/run_trpo.py tests/test_run_trpo.py tests/test_training_smoke.py tests/conftest.py
git commit -m "Train TRPO for many seeds at once in one jitted, vmapped epoch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: TRPO throughput benchmark and README

**Files:**
- Create: `src/abstract_rl/scripts/bench_jax.py`
- Modify: `README.md` (new section "JAX: many seeds at once")

**Interfaces:**
- Consumes: `run_trpo`.
- Produces:
  `python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6 --out data/bench.json`, which
  prints and writes env-steps per second, excluding the compiling first call.

- [ ] **Step 1: Write the benchmark**

```python
"""Throughput of the JAX TRPO training loop: environment steps per second summed over seeds.

``uv run python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6``. The first epoch call
compiles and is reported separately.
"""

import argparse
import json
import tempfile
import time
from typing import Any

from abstract_rl.scripts.hydra_conf import load, run_dict


def bench(env: str, seeds: int, epochs: int) -> dict[str, Any]:
    from abstract_rl.run import run_trpo as module

    conf = run_dict(load([f"env={env}", "alg=trpo"]))
    conf.update({"seeds": list(range(1, seeds + 1)), "data_root": tempfile.mkdtemp()})
    times: list[float] = []
    original = module.SeedRuns.store

    def timed_store(self: Any, name: str, models: Any) -> None:
        times.append(time.perf_counter())
        original(self, name, models)

    module.SeedRuns.store = timed_store  # type: ignore[method-assign]
    try:
        start = time.perf_counter()
        conf["num_epochs"] = epochs
        returns = module.run_trpo(conf)
    finally:
        module.SeedRuns.store = original  # type: ignore[method-assign]
    steps_per_epoch = conf["policy_batch_size"] * seeds
    first = times[0] - start
    steady = (times[-1] - times[0]) / max(len(times) - 1, 1)
    return {"env": env, "seeds": seeds, "first_epoch_s": first, "epoch_s": steady,
            "env_steps_per_s": steps_per_epoch / steady, "final_return_mean": float(returns[:, -1, 0].mean())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="pendulum")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    results = [bench(args.env, s, args.epochs) for s in args.seeds]
    for r in results:
        print(f"{r['env']:12s} seeds {r['seeds']:3d} | first call {r['first_epoch_s']:7.2f}s | epoch {r['epoch_s']:7.3f}s "
              f"| {r['env_steps_per_s']:10.0f} env-steps/s")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for Pendulum and HalfCheetah**

```bash
uv run python -m abstract_rl.scripts.bench_jax --env pendulum --seeds 1 4 16 --epochs 6 --out /tmp/bench_pendulum.json
uv run python -m abstract_rl.scripts.bench_jax --env halfcheetah --seeds 1 4 --epochs 3 --out /tmp/bench_halfcheetah.json
```

Expected: one line per seed count, with finite env-steps/s.

- [ ] **Step 3: Measure the torch reference on the same machine**

```bash
git worktree add /tmp/abstract_rl_torch 4f7828e
cd /tmp/abstract_rl_torch && uv sync && /usr/bin/time -p uv run python -c "
from abstract_rl.scripts.hydra_conf import load, run_dict
from abstract_rl.run.run_trpo import run_trpo
c = run_dict(load(['env=pendulum', 'alg=trpo'])); c['num_epochs'] = 6; c['data_root'] = '/tmp/torch_bench'
run_trpo(c)"
cd /Users/kosmitive/git/abstract_rl && git worktree remove --force /tmp/abstract_rl_torch
```

Torch env-steps/s = 6 × `policy_batch_size` / real seconds. That figure includes startup. Note this in the README
table.

- [ ] **Step 4: Add the README section**

Add a section "JAX: many seeds at once" after the quick-start commands. It covers:
1. the command `uv run python -m abstract_rl.scripts.run env=pendulum alg=trpo seeds=[1,2,3,4]`;
2. the `mps` extra and `jax_platform`. Metal runs the networks; the MJX envs need `cpu`.
3. a table built from Steps 2 and 3, with the columns `env`, `seeds`, `JAX env-steps/s (cpu)`,
   `torch env-steps/s (1 seed)`, `speedup` and `first call (compile)`;
4. one sentence stating that MPO is still the torch implementation.

Use the measured numbers only. Write "n/a" if a measurement failed, with the reason.

- [ ] **Step 5: Run the checks and commit**

Run: `uv run ruff check src tests && uv run pytest -o addopts="" -q 2>&1 | tail -3`
Expected: ruff is clean, and the suite has no failures.

```bash
git add src/abstract_rl/scripts/bench_jax.py README.md
git commit -m "Benchmark the JAX TRPO loop and document running many seeds at once.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
