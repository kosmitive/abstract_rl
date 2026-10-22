# JAX Port, Plan 3: MPO, Evaluation and Removing Torch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port MPO to the jitted, vmapped JAX design TRPO already uses: replay memory, the Resample, Retrace and
VariationalRanking operators, the temperature-dual E-step, the M-step, and the vmapped run loop. Port the eval, replay
and GIF scripts to the JAX environments with frozen observation statistics. Then remove torch and Gymnasium from the
library.

**Architecture:**
- The MPO replay memory is a `TrajectoryCollection` with a single lane of `C` rows: a flat ring of compacted valid
  transitions. Each row carries `ends` and `mask`, plus `insert_id` and `time` annotations. Because it is one long lane
  with episode boundaries, every lane operator (`transform_all`, `sample`, `to_batch`) works on it unchanged.
- `merge` reproduces torch's capping rule exactly: drop the oldest whole episodes until at most `tc_size` transitions
  remain, and always keep the newest.
- MPO's persistent state lives in `ModelState`: the Lagrange multipliers, both Adam states, `eta` and the memory.
- Eval loads unbatched Equinox models and runs `execute_policy(..., update_stats=False)`.

**Tech Stack:** As in Plans 1 and 2: Python 3.12, jax 0.11.2, equinox, optax, jaxtyping and beartype, gymnax, brax-MJX,
playground and mujoco.

**Spec:** `docs/superpowers/specs/2026-10-22-jax-port-design.md`. Also read the carry-over list at the end of Plan 1 and
Plan 2's final-review findings I-3, I-4 and I-5, quoted in Task 1.

## Global Constraints

These are Plan 2's constraints, unchanged:
- minimal diff, with names, comments and docstrings kept;
- float32, x64 never enabled;
- masked data via `jnp.where`; at least one valid row per memory;
- distinct target buffers before donation; gradients through `state.set(...)`;
- optimiser states as `<mc_key>_opt`;
- jaxtyping single-dimension shapes with a leading space; new modules added to `JAXTYPED_MODULES`;
- `uv run ruff check src tests` clean;
- commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
- tests pinned to CPU;
- explicit-path staging only. Never stage the untracked `docs/superpowers/plans/2026-10-10-modernize-review-fixes.md`
  or `src/abstract_rl/env/.jax_envs.py.swp`, and never use `git stash`.

Plus these:
- **Memory capacity.** `C = tc_size + 2 * R`, where `R = env_sample + E * max_steps` bounds the valid transitions of one
  rollout.
- **MPO uses a fixed lane count E** from the full `env_sample`, including for a budget-shortened last epoch.
- **P4.** The M-step's printed objective and constraints come from the min-step computation; there is no third
  recompute.

## Review Focus

1. **Capping equals torch's list rule.** On random episode lengths, `merge` keeps exactly the newest episodes whose
   total is at most `tc_size` (always at least the newest one), in collection-time order. Covered by Task 2,
   `test_merge_capping_matches_the_list_rule`.
2. **No episode wraps around the ring.** When an insert does not fit before the end, the pointer jumps to 0. An episode
   is never split, so boundary-aware scans stay correct. Covered by Task 2, `test_insert_never_splits_an_episode`.
3. **Frozen eval statistics.** Eval never changes `obs_rms` or the filters. Covered by Task 1,
   `test_frozen_rollout_keeps_statistics`.
4. **The E-step sees only valid rows.** The temperature dual and the log-softmax ranking ignore masked rows. Covered by
   Task 3, `test_eta_ignores_invalid_memory_rows`.
5. **Torch is gone from `src`.** `test_imports` passes with torch uninstalled, so `import torch` appears nowhere under
   `src/abstract_rl` except inside the optional SB3 baseline runner, lazily. Covered by Task 7,
   `test_library_does_not_import_torch`.

---

### Task 1: The `execute_policy` lane count and frozen statistics; spec revision

**Files:**
- Modify: `src/abstract_rl/env/mcmc_env_wrapper.py`
- Modify: `tests/test_env_wrapper.py` (append)
- Modify: `docs/superpowers/specs/2026-10-22-jax-port-design.md` (section 6.2, section 10, section 12)

**Interfaces:**
- Produces:
  - `execute_policy(state, key, policy, max_steps, batch_size, exploration=True, num_envs=None, update_stats=True) -> (state, tc, metrics)`.
    `num_envs=None` uses `self.num_envs(...)`. With `update_stats=False`, `obs_rms` and both filters are neither
    updated nor returned changed.
  - Every rollout row carries the annotation `"time"`, int32 `[E, L]`, set to the loop iteration that wrote it.
  - `execute_policy_once(state, key, max_steps, policy, opt=False, update_stats=True)` calls
    `execute_policy(..., num_envs=1, ...)` and no longer mutates `self._num_envs` (Plan 2 final review I-3).

- [ ] **Step 1: Revise the spec**

Replace section 6.2 with:

"### 6.2 Replay memory (MPO)

- **Layout.** The replay memory is a `TrajectoryCollection` with one lane of `C` rows. It is a flat ring of
  transitions compacted from the rollouts, holding only valid rows. Every row has `ends`, `mask`, the annotation
  `insert_id` (the epoch that inserted it) and the annotation `time` (the rollout step that wrote it).
  `C = tc_size + 2·R` with `R = env_sample + E·max_steps`.
- **Insert.** The valid rows of a rollout are compacted lane by lane, so episodes stay contiguous. They are written
  starting at the ring pointer. If they do not fit before the end, the pointer jumps to 0 first, so no episode wraps.
- **Capping keeps the exact torch rule.** Episodes are ordered by `(insert_id, time of their last step)`, newest first.
  The longest newest prefix whose lengths sum to `tc_size` or less is kept, and the newest episode is always kept. Every
  other row is masked.
- **Sampling** is uniform over valid rows, with replacement.
- **Operators.** Resample, ranking and Retrace run over the whole memory lane and are boundary-aware through `ends` and
  `mask`. The temperature dual only sees valid rows."

Add a D7 row to section 10:

"D7 | When capping drops part of one epoch's episodes, ties between lanes finishing at the same rollout step are broken
by lane order. | Lockstep lanes finish episodes simultaneously; torch collected them one after another."

Add a risk to section 12:

"The E-step resamples every memory row, including the masked ones, because C is about 1.1 × tc_size."

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_env_wrapper.py`:

```python
def test_frozen_rollout_keeps_statistics(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, obs_norm=True, env_num_envs=2)
    state, _, _ = env.execute_policy(state, KEY, policy, 200, 400)
    frozen, tc, _ = env.execute_policy(state, jax.random.key(1), policy, 200, 400, update_stats=False)
    np.testing.assert_allclose(frozen.get("obs_rms").count, state.get("obs_rms").count)
    np.testing.assert_allclose(frozen.get("obs_rms").mean, state.get("obs_rms").mean)
    assert int(frozen.get("state_filter")._n) == int(state.get("state_filter")._n)


def test_explicit_lane_count_and_time_annotation(make_pendulum_mc):
    env, policy, state = _setup(make_pendulum_mc, env_num_envs=4)
    _, tc, _ = env.execute_policy(state, KEY, policy, 200, 200, num_envs=1)
    t = tc.trajectories()
    assert t.mask.shape[0] == 1
    np.testing.assert_array_equal(np.asarray(t["time"][0][:200]), np.arange(200))
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_env_wrapper.py`
Expected: the two new tests FAIL with an unexpected keyword argument.

- [ ] **Step 4: Implement**

In `execute_policy`:
- add the keyword parameters `num_envs: int | None = None` and `update_stats: bool = True`, and set
  `e = num_envs if num_envs is not None else self.num_envs(batch_size, max_steps)`;
- guard every `obs_rms.update(...)` and every `state_filter.register(...)` / `reward_filter.register(...)` with
  `if update_stats:`, a static Python flag;
- add `"time": zeros(dtype=jnp.int32)` to `buf`, write `jnp.full(e, i, jnp.int32)` into it in `step_vals`, and pass
  `annotations={"time": buf.pop("time")}` to the `Trajectory`.

`execute_policy_once` becomes:

```python
    def execute_policy_once(
        self, state: ModelState, key: PRNGKeyArray, max_steps: int, policy: Any, opt: bool = False,
        update_stats: bool = True,
    ) -> tuple[ModelState, Trajectory]:
        """
        Execute a policy once for the maximum number of steps or until the environment terminates.
        :return: The new state and the single-lane trajectory of the episode.
        """
        state, tc, _ = self.execute_policy(
            state, key, policy, max_steps, 1, exploration=not opt, num_envs=1, update_stats=update_stats
        )
        return state, jax.tree.map(lambda x: x[0], tc.trajectories())
```

- [ ] **Step 5: Run the tests and commit**

Run: `uv run pytest -o addopts="" -q tests/test_env_wrapper.py tests/test_obs_norm.py tests/test_run_trpo.py`
Expected: all pass. Then run the full suite.

```bash
git add src/abstract_rl/env/mcmc_env_wrapper.py tests/test_env_wrapper.py docs/superpowers/specs/2026-10-22-jax-port-design.md
git commit -m "Choose the lane count per rollout and allow frozen statistics.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The replay memory, with `TrajectoryCollection.memory` and `merge`

**Files:**
- Modify: `src/abstract_rl/data_structures/temporal_difference_data/trajectory_collection.py` (append)
- Create: `tests/test_replay_memory.py`

**Interfaces:**
- Produces:
  - `TrajectoryCollection.memory(env, capacity: int, obs_dim: int, act_dim: int) -> TrajectoryCollection`, a
    classmethod. It returns one lane of `capacity` rows, all invalid, with the annotations `insert_id` (int32, −1) and
    `time` (int32, 0), and with fields `_ptr` and `_inserted` (int32 scalars).
  - `merge(collection: TrajectoryCollection, max_l: int) -> TrajectoryCollection`. It compacts and inserts the valid
    rows of a rollout collection (its lanes and annotations `time`), then caps to `max_l` transitions.
  - `with_annotations(template: dict[str, Any]) -> TrajectoryCollection` adds zero-filled annotations of the given
    per-row shapes to every lane, once at setup.

- [ ] **Step 1: Write the failing tests**

`tests/test_replay_memory.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


def _rollout(lengths_per_lane, insert_value, length):
    """A rollout collection whose lanes hold whole episodes of the given lengths; states carry an episode tag."""
    e = len(lengths_per_lane)
    states = np.zeros((e, length, 1), np.float32)
    ends = np.zeros((e, length), bool)
    mask = np.zeros((e, length), bool)
    time = np.zeros((e, length), np.int32)
    for lane, lengths in enumerate(lengths_per_lane):
        t = 0
        for k, n in enumerate(lengths):
            states[lane, t : t + n, 0] = insert_value * 1000 + lane * 100 + k
            ends[lane, t + n - 1] = True
            mask[lane, t : t + n] = True
            time[lane, t : t + n] = np.arange(t, t + n)
            t += n
    z = jnp.zeros((e, length, 1))
    traj = Trajectory(jnp.asarray(states), jnp.asarray(states), z, z, jnp.zeros((e, length)), z, z, jnp.asarray(ends),
                      jnp.asarray(mask), {"time": jnp.asarray(time)})
    return TrajectoryCollection(None, traj)


def _episodes(memory):
    t = jax.tree.map(lambda x: np.asarray(x[0]), memory.trajectories())
    tags = t.states[t.mask, 0]
    out, last = [], None
    for tag in tags:
        if tag != last:
            out.append(int(tag))
            last = tag
    return out


def _list_rule(collected, max_l):
    """torch TrajectoryCollection.extend: drop the oldest whole trajectories until the rest fits, keep the newest."""
    k, i = 0, len(collected)
    while i > 0 and k + collected[i - 1][1] <= max_l:
        i -= 1
        k += collected[i][1]
    return collected[min(i, len(collected) - 1) :]


def test_merge_capping_matches_the_list_rule():
    rng = np.random.default_rng(0)
    memory = TrajectoryCollection.memory(None, 400, 1, 1)
    collected = []
    for epoch in range(25):
        lanes = [list(rng.integers(1, 12, size=rng.integers(1, 4))) for _ in range(3)]
        rollout = _rollout(lanes, epoch, 40)
        memory = jax.jit(lambda m, r: m.merge(r, 60))(memory, rollout)
        # collection order: by the time the episode ended, ties by lane
        finished = []
        for lane, lengths in enumerate(lanes):
            t = 0
            for k, n in enumerate(lengths):
                t += n
                finished.append((t - 1, lane, epoch * 1000 + lane * 100 + k, n))
        collected += [(tag, n) for _, _, tag, n in sorted(finished)]
        collected = _list_rule(collected, 60)
        assert sorted(_episodes(memory)) == sorted(tag for tag, _ in collected)
        assert int(memory.trajectories().mask.sum()) == sum(n for _, n in collected)


def test_insert_never_splits_an_episode():
    memory = TrajectoryCollection.memory(None, 50, 1, 1)
    for epoch in range(8):
        memory = memory.merge(_rollout([[7, 7]], epoch, 20), 1000)
        t = jax.tree.map(lambda x: np.asarray(x[0]), memory.trajectories())
        valid = np.nonzero(t.mask)[0]
        # every valid run of rows that shares a tag is contiguous and ends with an end flag
        for tag in set(t.states[t.mask, 0].tolist()):
            rows = valid[t.states[valid, 0] == tag]
            assert np.all(np.diff(rows) == 1) and t.ends[rows[-1]]


def test_sampling_draws_only_kept_rows():
    memory = TrajectoryCollection.memory(None, 200, 1, 1)
    for epoch in range(6):
        memory = memory.merge(_rollout([[10, 10], [5]], epoch, 30), 40)
    batch = memory.sample(jax.random.key(0), 5000, [])
    kept = set(_episodes(memory))
    assert set(np.asarray(batch.states[:, 0]).astype(int).tolist()) <= kept


def test_with_annotations_adds_zero_rows():
    memory = TrajectoryCollection.memory(None, 10, 1, 1).with_annotations({"add_q": jax.ShapeDtypeStruct((4,), jnp.float32)})
    assert memory.trajectories()["add_q"].shape == (1, 10, 4)
    with pytest.raises(KeyError):
        memory.trajectories()["missing"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_replay_memory.py`
Expected: FAIL, because `TrajectoryCollection.memory` does not exist.

- [ ] **Step 3: Implement**

`TrajectoryCollection` gains two optional array fields, `_ptr: Any = None` and `_inserted: Any = None`, declared after
`_trajectories`. Then append:

```python
    @classmethod
    def memory(cls, env: Any, capacity: int, obs_dim: int, act_dim: int) -> "TrajectoryCollection":
        """An empty replay memory: one lane of ``capacity`` invalid rows (MPO, spec section 6.2)."""
        z = jnp.zeros((1, capacity, 1), jnp.float32)
        trajectories = Trajectory(
            jnp.zeros((1, capacity, obs_dim)), jnp.zeros((1, capacity, obs_dim)), jnp.zeros((1, capacity, act_dim)), z,
            jnp.zeros((1, capacity)), z, z, jnp.zeros((1, capacity), bool), jnp.zeros((1, capacity), bool),
            {"insert_id": jnp.full((1, capacity), -1, jnp.int32), "time": jnp.zeros((1, capacity), jnp.int32)},
        )
        collection = cls(env, trajectories)
        return eqx.tree_at(lambda c: (c._ptr, c._inserted), collection, (jnp.asarray(0), jnp.asarray(0)),
                           is_leaf=lambda x: x is None)

    def with_annotations(self, template: dict[str, Any]) -> "TrajectoryCollection":
        """Zero-filled annotations with the given per-row shapes, e.g. the output structure of the MPO operators."""
        t = self._trajectories
        lanes, length = t.mask.shape
        for k, spec in template.items():
            t = t.set(k, jnp.zeros((lanes, length, *spec.shape), spec.dtype))
        return self.with_trajectories(t)

    def merge(self, collection: "TrajectoryCollection", max_l: int) -> "TrajectoryCollection":
        """Merges the valid rows of a rollout collection into this replay memory and drops the oldest episodes until
        at most ``max_l`` transitions remain (the newest episode is always kept).

        :param collection: The rollout to merge, lanes of whole episodes.
        :param max_l: The maximum number of transitions.
        """
        mem = jax.tree.map(lambda x: x[0], self._trajectories)
        new = jax.tree.map(_flat, collection.trajectories())
        capacity = mem.mask.shape[0]
        n_new = new.mask.shape[0]

        # compact the valid rows of the rollout, lane by lane, so episodes stay contiguous
        order = jnp.argsort(~new.mask, stable=True)
        count = jnp.sum(new.mask)
        ptr = jnp.where(self._ptr + count > capacity, 0, self._ptr)
        rows = (ptr + jnp.arange(n_new)) % capacity
        write = jnp.arange(n_new) < count

        def insert(old: Any, value: Any) -> Any:
            value = value[order]
            selected = jnp.where(write.reshape((-1,) + (1,) * (value.ndim - 1)), value, old[rows])
            return old.at[rows].set(selected)

        fields = {
            "states": new.states, "next_states": new.next_states, "actions": new.actions, "rewards": new.rewards,
            "raw_rewards": new.raw_rewards, "dones": new.dones, "log_likelihoods": new.log_likelihoods,
            "ends": new.ends, "mask": new.mask,
        }
        mem = mem.replace(**{k: insert(getattr(mem, k), v) for k, v in fields.items()})
        mem = mem.set("time", insert(mem["time"], new["time"]))
        mem = mem.set("insert_id", insert(mem["insert_id"], jnp.full(n_new, self._inserted, jnp.int32)))

        # capping: newest episodes first (insert id, then the time of their last step), longest prefix within max_l
        ends = mem.ends & mem.mask
        episode = jnp.cumsum(mem.ends) - mem.ends
        length = jax.ops.segment_sum(mem.mask.astype(jnp.int32), episode, num_segments=capacity)
        valid = jax.ops.segment_sum(ends.astype(jnp.int32), episode, num_segments=capacity) > 0
        recency = jnp.where(ends, mem["insert_id"] * capacity * 4 + mem["time"], -1)
        key = jax.ops.segment_max(recency, episode, num_segments=capacity)
        key = jnp.where(valid, key, -1)
        newest_first = jnp.argsort(-key, stable=True)
        cum = jnp.cumsum(length[newest_first] * valid[newest_first])
        keep_sorted = valid[newest_first] & ((cum <= max_l) | (jnp.arange(capacity) == 0))
        keep = jnp.zeros(capacity, bool).at[newest_first].set(keep_sorted)
        mem = mem.replace(mask=mem.mask & keep[episode])

        out = self.with_trajectories(jax.tree.map(lambda x: x[None], mem))
        return eqx.tree_at(lambda c: (c._ptr, c._inserted), out, (ptr + count, self._inserted + 1))
```

The `recency` key orders episodes by insert id, then by the rollout step of their last row. `capacity * 4` is larger
than any `time` value, because a rollout's `L` is at most `capacity`.

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest -o addopts="" -q tests/test_replay_memory.py tests/test_trajectories.py`
Expected: all pass. Then run the full suite.

```bash
git add src/abstract_rl/data_structures/temporal_difference_data/trajectory_collection.py tests/test_replay_memory.py
git commit -m "Keep the MPO replay memory as a compacted ring with the torch capping rule.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The Resample, Retrace and VariationalRanking operators and the policy builder

**Files:**
- Modify: `src/abstract_rl/operator/trajectory_operator.py` (key plumbing), `resample_actions_operator.py`,
  `retrace_operator.py`, `np_rew_operator.py`
- Modify: `src/abstract_rl/policy/non_parametric/non_parametric_variational_policy_builder.py`
- Modify: `src/abstract_rl/operator/{td,gae,filter}_operator.py`: `transform` gains an unused `key=None` parameter
- Create: `tests/test_mpo_operators.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `TrajectoryOperator.transform(state, trajectory, key=None)` and `transform_all(state, trajectories, key=None)`.
    With a key, one key per lane is split and vmapped.
  - `ResampleOperator` sets `add_act` (`[L,K]` or `[L,K,A]`), `add_q` `[L,K]` and `add_pi` `[L,K]`, using the target
    policy and the target Q.
  - `RetraceOperator` sets `q` `[L,1]` and `tq` `[L,1]`, using the current policy, the target Q, `v_val` with the key
    and the truncated Retrace sum within episodes.
  - `VariationalRankingOperator.hook(state, tc) -> state` stores `eta`; its `transform` sets `grad_ranking` `[L,K]`.
  - `NonParametricVariationalPolicyBuilder` provides `init_state() -> {"eta": [1]}`, `calc_eta(state, tc) -> state`
    (dual over the valid memory rows, warm-started from `state["eta"]`, 200 Adam steps, because `rnk_max_steps` is
    read but not passed, as in torch) and `build(state, trajectory) -> NonParametricPolicy`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mpo_operators.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import golden, load_named, sub

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet

MPO_VAL = {"val_struct": [256, 256], "val_act_fn": ["elu", "elu", "none"]}


def _mpo_mc(make_pendulum_mc, **overrides):
    from abstract_rl.scripts.mpo_pendulum import run_config

    keys = ("res_add_acts", "ret_lambda", "ret_add_acts", "mem_bootstrap_steps", "rnk_eps", "rnk_lr", "rnk_init_eta",
            "rnk_max_steps")
    return make_pendulum_mc(alg="mpo", **MPO_VAL, **{k: run_config[k] for k in keys}, **overrides)


def _models(mc):
    with mc["conf"].ns("policy"):
        policy = load_named(select_policy(mc, jax.random.key(0)), sub(golden("retrace_policy"), "params"))
    with mc["conf"].ns("val"):
        q = load_named(FullyQNeuralNet(mc, jax.random.key(0)), sub(golden("retrace_q"), "params"))
    return ModelState.create({}, {"policy": (policy, policy), "q_network": (q, q)})


def _lane(g):
    s = jnp.asarray(g["all_states"], jnp.float32)
    n = len(s) - 1
    ends = jnp.zeros(n, bool).at[-1].set(True)
    return Trajectory(s[:-1], s[1:], jnp.asarray(g["actions"], jnp.float32), jnp.asarray(g["rewards"], jnp.float32),
                      jnp.asarray(g["rewards"][:, 0], jnp.float32), jnp.asarray(g["dones"], jnp.float32),
                      jnp.asarray(g["log_likelihoods"], jnp.float32), ends, jnp.ones(n, bool))


@pytest.mark.parametrize("case", ["truncated", "terminated"])
def test_resample_and_retrace_match_torch(make_pendulum_mc, monkeypatch, case):
    # the torch fixture replaced rsample by the mean; zero noise reproduces it
    monkeypatch.setattr(jax.random, "normal", lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype))
    mc = _mpo_mc(make_pendulum_mc)
    state = _models(mc)
    with mc["conf"].ns("res"):
        res = ResampleOperator(mc)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    g = golden(f"retrace_{case}")
    t = ret.transform(state, res.transform(state, _lane(g), jax.random.key(0)), jax.random.key(1))
    np.testing.assert_allclose(t["add_act"], g["add_act"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["add_q"], g["add_q"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(t["add_pi"], g["add_pi"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t["q"], g["q"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(t["tq"], g["tq"], rtol=1e-4, atol=1e-4)


def test_retrace_does_not_cross_episodes(make_pendulum_mc, monkeypatch):
    monkeypatch.setattr(jax.random, "normal", lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype))
    mc = _mpo_mc(make_pendulum_mc)
    state = _models(mc)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    g1, g2 = golden("retrace_terminated"), golden("retrace_truncated")
    both = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), _lane(g1), _lane(g2))
    t = ret.transform(state, both, jax.random.key(1))
    n1 = len(g1["rewards"])
    np.testing.assert_allclose(t["tq"][:n1], g1["tq"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t["tq"][n1:], g2["tq"], rtol=1e-4, atol=1e-4)


def test_eta_ignores_invalid_memory_rows(make_pendulum_mc):
    mc = _mpo_mc(make_pendulum_mc)
    with mc["conf"].ns("rnk"):
        rnk = VariationalRankingOperator(mc)
    q = jax.random.normal(jax.random.key(0), (1, 64, 20)) * 5
    memory = TrajectoryCollection.memory(None, 80, 3, 1).with_annotations({"add_q": jax.ShapeDtypeStruct((20,), jnp.float32)})
    t = memory.trajectories()
    t = t.set("add_q", t["add_q"].at[:, :64].set(q).at[:, 64:].set(1e6)).replace(mask=t.mask.at[:, :64].set(True))
    memory = memory.with_trajectories(t)
    state = ModelState.create(rnk.b.init_state(), {})
    eta = rnk.hook(state, memory).get("eta")
    from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature

    np.testing.assert_allclose(eta, find_optimal_temperature(q[0], rnk.b.non_parametric_variational_eps, 1.0, rnk.b.non_parametric_variational_lr), rtol=1e-5)
    ranked = rnk.transform_all(ModelState.create({"eta": eta}, {}), memory).trajectories()["grad_ranking"]
    np.testing.assert_allclose(jnp.exp(ranked[0, :64]).sum(axis=1), 1.0, rtol=1e-5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_mpo_operators.py`
Expected: FAIL, because `transform` takes no key and the operators still read torch models in `__init__`.

- [ ] **Step 3: Implement**

`trajectory_operator.py`: `transform(self, state, trajectory, key=None)` (abstract) and:

```python
    def transform_all(self, state: ModelState, trajectories: TrajectoryCollection, key: Any = None) -> TrajectoryCollection:
        """
        Transform all lanes of a collection with the current instance of the evaluation operator.
        :param state: The traced model state.
        :param trajectories: trajectories to transform.
        :param key: Random key of stochastic operators, split into one key per lane.
        """
        lanes = trajectories.trajectories()
        if key is None:
            out = jax.vmap(lambda trajectory: self.transform(state, trajectory))(lanes)
        else:
            keys = jax.random.split(key, len(trajectories))
            out = jax.vmap(lambda trajectory, k: self.transform(state, trajectory, k))(lanes, keys)
        return trajectories.with_trajectories(out)
```

TD, GAE and Filter get the extra `key: Any = None` parameter, which they don't use.

`resample_actions_operator.py`. `__init__` keeps `conf`, `add_act`, `mc` and `discount`, and drops `policy` and `q_net`.
The transform is:

```python
    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the operator.
        """
        policy = state.get("policy", True)
        q_net = state.get("q_network", True)

        # sample actions for sample based
        t_states = trajectory.states
        f_stats = policy.forward(t_states)
        sampled_actions = policy.sample_actions(key, suff_stats=f_stats, num_actions=self.add_act)
        ll = policy.log_prob(sampled_actions, suff_stats=f_stats)
        q_vals = q_net.q_val_multi(t_states, sampled_actions)
        return trajectory.set("add_act", sampled_actions).set("add_q", q_vals).set("add_pi", ll)
```

`retrace_operator.py`. `__init__` keeps `conf`, `mc`, `discount` and `env`, sets `self.ret_lambda = float(...)`, keeps
`ret_bootstrap` and `ret_add_acts`, and drops the models and the torch `train()`. Then:

```python
def _shift(x: Array, k: int) -> Array:
    """x[t + k], zero beyond the end."""
    return jnp.concatenate([x[k:], jnp.zeros((k,) + x.shape[1:], x.dtype)])

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """
        policy = state.get("policy", False)
        q_network = state.get("q_network", True)

        # first rolling estimate of the exp q function
        discount = self.discount
        tl = len(trajectory)

        # obtain q and v value estimates
        states = trajectory.states
        actions = trajectory.actions
        q_vals = q_network.q_val(states, actions).reshape(-1)

        # expectation of the next state under the evaluated (current) policy, the same policy defines the trace
        # coefficients c; terminal states have no value
        nxt_states = trajectory.next_states
        dones = trajectory.dones.reshape(-1)
        nxt_v_vals = q_network.v_val(state, key, nxt_states, policy=policy).reshape(-1) * (1 - dones)

        # get import. sampling coefficients
        c = self._def_bounded_importance_sampling_coefficients(state, trajectory).reshape(-1)

        # calc td errors for all t
        r = trajectory.rewards.reshape(-1)
        td_errors_q = r + discount * nxt_v_vals - q_vals
        # Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{j=t}^{t+B-1} gamma^(j-t) (prod_{i=t+1}^{j} c_i) delta_j, computed
        # vectorized over t by iterating over the offset j - t; the sum stops at the end of the episode of t
        episode = jnp.cumsum(trajectory.ends) - trajectory.ends
        tq = q_vals + td_errors_q
        coeff = jnp.ones(tl)
        for k in range(1, min(self.ret_bootstrap, tl)):
            same_episode = (_shift(episode, k) == episode) & _shift(trajectory.mask, k)
            coeff = coeff * discount * _shift(c, k) * same_episode
            tq = tq + coeff * _shift(td_errors_q, k)

        return trajectory.set("q", q_vals.reshape(-1, 1)).set("tq", tq.reshape(-1, 1))

    def _def_bounded_importance_sampling_coefficients(self, state: ModelState, trajectory: Trajectory) -> Array:
        """
        Calculated the bounded importance sampling coefficients.
        :param trajectory: The trajectory for which the coefficients have to be calculated
        :return: A vector containing the coefficients.
        """
        policy = state.get("policy", False)

        # obtain the bounded importance sampling coefficients coefficients
        beh_lls = trajectory.log_likelihoods.reshape([-1, 1])
        pi_ll = policy.log_prob(trajectory.actions, states=trajectory.states)
        r = jnp.exp(pi_ll - beh_lls)
        c = jnp.minimum(r, 1.0) * self.ret_lambda
        return c
```

`non_parametric_variational_policy_builder.py`. The constructor keeps the conf reads (`eps`, `lr`, `init_eta`,
`max_steps`), drops the models and the logger, and the methods become:

```python
    def init_state(self) -> dict[str, Any]:
        """The traced temperature, warm started from rnk_init_eta."""
        return {"eta": jnp.asarray([self.non_parametric_variational_init_eta], DTYPE)}

    def calc_eta(self, state: ModelState, tc: TrajectoryCollection) -> ModelState:
        """
        Calculate the optimal eta using the find optimal temperature function, over the valid rows of the collection.
        """
        t = tc.trajectories()

        # append all q values and likelihoods
        q_tensor = t["add_q"].reshape((-1, t["add_q"].shape[-1]))
        eta = find_optimal_temperature(
            q_tensor, self.non_parametric_variational_eps, state.get("eta")[0], self.non_parametric_variational_lr,
            mask=t.mask.reshape(-1),
        )
        return state.set("eta", eta)

    def build(self, state: ModelState, trajectory: Trajectory) -> NonParametricPolicy:
        """
        Takes a trajectory or batch data and builds the correct samples.
        :param trajectory: Tje trajectory to transform.
        :return: A non parametric policy.
        """

        exp_samples = trajectory["add_q"] / state.get("eta")
        return NonParametricPolicy(exp_samples)
```

`np_rew_operator.py`. `hook(self, state, tc) -> ModelState` returns `self.b.calc_eta(state, tc)`, and `transform`
becomes:

```python
    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # call the super class
        q = self.b.build(state, trajectory)
        # normalize per state, e.g. log q(a|s) = Q(s,a) / eta - log sum_a' exp(Q(s,a') / eta)
        max_q_samples = jnp.max(q.samples, axis=1, keepdims=True)
        q_norm = jnp.log(jnp.sum(jnp.exp(q.samples - max_q_samples), axis=1, keepdims=True)) + max_q_samples
        q.samples = q.samples - q_norm

        # add a annotation to the trajectory
        return trajectory.set("grad_ranking", q.samples)
```

Add the seven touched modules to `JAXTYPED_MODULES`.

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest -o addopts="" -q tests/test_mpo_operators.py tests/test_advantage_operators.py`
Expected: all pass. Then run the full suite.

```bash
git add src/abstract_rl/operator src/abstract_rl/policy/non_parametric tests/test_mpo_operators.py tests/conftest.py
git commit -m "Port the resample, retrace and variational ranking operators to JAX.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `MPOAlgorithm`

**Files:**
- Modify: `src/abstract_rl/algorithms/continuous/policy_gradient/maximum_a_posteriori_policy_optimization_algorithm.py` (whole file)
- Create: `tests/test_mpo.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces, `MPOAlgorithm(mc)`:
  - `init_state(policy) -> {"lagrange_mults": ones([n]), "opt_outer", "opt_inner"}`, where `n` is 1, or the suff-stat
    dim when the constraint is split;
  - `sample_v(key) -> list`;
  - `shuffle_up_stats(state, batch, policy)`, `main_obj(state, batch, l, policy)`, `main_constraints(state, batch, l, policy, v)`
    and `obj(state, batch, policy, lagrange_mults, v)`;
  - `perform_step(state, step, batch, v) -> (state, metrics)` (max step, then min step; P4 metrics from the min step);
  - `perform_mult_steps(state, key, num_steps, tc, batch_size) -> (state, metrics)`, a `lax.scan`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mpo.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import golden, load_named, named_parameters, sub

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.mpo_pendulum import run_config

MPO_KEYS = [k for k in run_config if k.startswith("mpo_")]
CASES = {
    "mpo_uni": {},
    "mpo_split": {"mpo_coordinate_ascent_objective": True, "mpo_coordinate_ascent_constraint": True,
                  "mpo_trust_regions": [0.01, 0.02]},
    "mpo_diag": {"env_name": "TestPointMass-v0"},
}


@pytest.mark.parametrize("case", CASES)
def test_three_mpo_steps_match_torch(make_pendulum_mc, case):
    mc = make_pendulum_mc(alg="mpo", **{k: run_config[k] for k in MPO_KEYS}, **CASES[case])
    g = golden(case)
    with mc["conf"].ns("policy"):
        fresh = select_policy(mc, jax.random.key(0))
    policy, target = load_named(fresh, sub(g, "step0/params")), load_named(fresh, sub(g, "target/params"))
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    state = ModelState.create(alg.init_state(policy), {"policy": (policy, target)})
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(jnp.asarray(g["states"]), z, z, z, jnp.asarray(g["states"]), z,
                  {"add_act": jnp.asarray(g["add_act"]), "grad_ranking": jnp.asarray(g["grad_ranking"]),
                   "add_pi": jnp.zeros((n, 20))})
    np.testing.assert_allclose(state.get("lagrange_mults"), g["step0/lagrange"].reshape(-1))
    v = alg.sample_v(jax.random.key(0))
    step = jax.jit(lambda s, i: alg.perform_step(s, i, batch, v))
    for s in range(3):
        state, _ = step(state, s)
        got = named_parameters(state.get("policy"))
        for k, expected in sub(g, f"step{s + 1}/params").items():
            np.testing.assert_allclose(got[k], expected, rtol=1e-4, atol=1e-5, err_msg=f"{case} step {s + 1} {k}")
        np.testing.assert_allclose(state.get("lagrange_mults"), g[f"step{s + 1}/lagrange"].reshape(-1), rtol=1e-4, atol=1e-6)


def test_list_trust_regions_are_sampled_per_key(make_pendulum_mc):
    mc = make_pendulum_mc(alg="mpo", **{k: run_config[k] for k in MPO_KEYS}, mpo_trust_regions=[[0.01, 0.05]])
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    a, b = alg.sample_v(jax.random.key(0))[0], alg.sample_v(jax.random.key(1))[0]
    assert 0.01 <= float(a) <= 0.05 and float(a) != float(b)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_mpo.py`
Expected: FAIL. The torch `MPOAlgorithm` builds torch optimisers.

- [ ] **Step 3: Convert the module**

Keep the class docstring, `__repr__`, the attribute names (`outer_lr`, `inner_lr`, `mode`, `trust_regions`,
`split_objective`, `split_constraint`, `l2_reg`, `fmtstr`, `titlestr`) and the method names. The bodies become:

```python
class MPOAlgorithm(PolicyOptimizationAlgorithm):
    (docstring and __repr__ unchanged)

    def __init__(self, mc: ModelConfiguration) -> None:
        PolicyOptimizationAlgorithm.__init__(self, mc)

        # create lagrange multiplier
        policy = self.mc.get("policy", target=False)
        num_lagrange_multipliers = policy.get_suff_stats_dim()

        # inner and outer learning rated, trust regions and batch size shortcuts
        self.outer_lr = self.conf["outer_lr"]
        self.inner_lr = self.conf["inner_lr"]
        self.mode = self.conf["mode"]
        self.trust_regions = self.conf["trust_regions"]
        self.split_objective = self.conf["coordinate_ascent_objective"]
        self.split_constraint = self.conf["coordinate_ascent_constraint"]
        self.cli = CliPrinter().instance
        self.l2_reg = self.conf.get("l2_reg", 0.0)

        self.num_params = 1 if not self.split_constraint else num_lagrange_multipliers
        self.fmtstr = "%10i | %10.5f" + self.num_params * " | %10.5f x %10.5f"
        self.titlestr = "%10s | %10s" + self.num_params * " | %23s"

    def init_state(self, policy: Any) -> dict[str, Any]:
        """Lagrange multipliers (log space, initialised to one like torch) and the persistent Adam states."""
        lagrange_mults = jnp.ones(self.num_params, DTYPE)
        return {
            "lagrange_mults": lagrange_mults,
            # optimizers are persistent over all steps, so adam can accumulate its moment estimates
            "opt_outer": optax.adam(self.outer_lr).init(eqx.filter(policy, eqx.is_inexact_array)),
            "opt_inner": optax.adam(self.inner_lr).init(lagrange_mults),
        }

    def sample_v(self, key: PRNGKeyArray) -> list[Any]:
        """Sample new trust regions.

        :return: the new sampled trust regions.
        """

        samp_trust_regions = self.trust_regions
        v: list[Any] = [None] * len(samp_trust_regions)
        for d in range(len(samp_trust_regions)):
            if isinstance(samp_trust_regions[d], list):
                lo, hi = samp_trust_regions[d]
                v[d] = jax.random.uniform(jax.random.fold_in(key, d), (), DTYPE, lo, hi)
            else:
                v[d] = samp_trust_regions[d]
        return v

    def shuffle_up_stats(self, state: ModelState, batch: Any, policy: Any) -> list[Any]:
        (docstring unchanged)
        # get target and correct policy
        t_states = jnp.asarray(batch.states, DTYPE)
        t_policy = state.get("policy", target=True)

        # target stats
        target_suff_stats = jax.lax.stop_gradient(t_policy.forward(t_states))
        suff_stats = policy.forward(t_states)
        l_suff_stats = [suff_stats]

        # for each dimension
        num_dims = target_suff_stats.shape[1]
        for d in range(num_dims):
            lst_els = []
            if d > 0:
                lst_els.append(target_suff_stats[:, :d])
            lst_els.append(suff_stats[:, d : d + 1])
            if d < num_dims - 1:
                lst_els.append(target_suff_stats[:, d + 1 :])
            l_suff_stats.append(jnp.concatenate(lst_els, axis=1))

        return l_suff_stats

    def main_obj(self, state: ModelState, batch: Any, l_suff_stats: list[Any], policy: Any) -> Any:
        (docstring unchanged)
        # split objective if wanted
        if self.split_objective:
            # build the objective in a coordinate ascent fashion
            # iterate over the dimensions
            num_dims = l_suff_stats[0].shape[1]
            obj = jnp.zeros(())
            for d in range(num_dims):
                part_obj = self.__lagrange_obj(batch, l_suff_stats[d + 1], policy)
                obj = obj + part_obj

        else:
            obj = self.__lagrange_obj(batch, l_suff_stats[0], policy)

        return obj

    def main_constraints(self, state: ModelState, batch: Any, l_suff_stats: list[Any], policy: Any, v: list[Any]) -> list[Any]:
        (docstring unchanged)
        kl = "i-projection" if self.mode == "parametric" else "m-projection"

        # split constraint if wanted
        if self.split_constraint:
            num_dims = l_suff_stats[0].shape[1]
            constraints = []
            for d in range(num_dims):
                c = v[d] - policy.kl_divergence(state, batch, l_suff_stats[d + 1], kl, "mean")
                constraints.append(c)

        else:
            constraints = [(v[0] - policy.kl_divergence(state, batch, l_suff_stats[0], kl, "mean"))]

        return constraints

    def obj(self, state: ModelState, batch: Any, policy: Any, lagrange_mults: Any, v: list[Any]) -> Any:  # type: ignore[override]
        (docstring unchanged)
        l_suff_stats = self.shuffle_up_stats(state, batch, policy)
        obj = self.main_obj(state, batch, l_suff_stats, policy)
        constraints = self.main_constraints(state, batch, l_suff_stats, policy, v)
        for i, c in enumerate(constraints):
            obj = obj + jnp.exp(lagrange_mults[i]) * c

        return obj

    def perform_mult_steps(
        self, state: ModelState, key: PRNGKeyArray, num_steps: int, tc: Any, batch_size: int
    ) -> tuple[ModelState, dict[str, Any]]:
        (docstring unchanged)
        k_v, key = jax.random.split(key)
        v = self.sample_v(k_v)

        # perform a min step
        def body(carry: Any, s: Any) -> Any:
            state, key = carry
            key, sub = jax.random.split(key)
            batch = tc.sample(sub, batch_size, ["add_act", "grad_ranking", "add_pi"])
            state, metrics = self.perform_step(state, s, batch, v)
            return (state, key), metrics

        (state, _), metrics = jax.lax.scan(body, (state, key), jnp.arange(num_steps))
        return state, metrics

    def perform_step(self, state: ModelState, step: Any, batch: Any, v: list[Any]) -> tuple[ModelState, dict[str, Any]]:
        (docstring unchanged)
        # perform a max step on the policy, then a min step on the lagrange multipliers
        state = self.__max_step(state, batch, v)
        state, obj, constraints = self.__min_step(state, batch, v)

        # the logged objective and constraints are the ones of the min step (spec P4)
        return state, {"obj": obj, "lagrange_mults": state.get("lagrange_mults"), "constraints": jnp.stack(constraints)}

    def __lagrange_obj(self, batch: Any, suff_stats: Any, policy: Any) -> Any:
        (docstring unchanged)
        # build regularization for mean
        if self.mode == "parametric":
            act = jnp.asarray(batch.actions, DTYPE)
            adv = jnp.asarray(batch["a"], DTYPE)

            # sample sufficient statistics
            ll = policy.log_prob(act, suff_stats=suff_stats)
            rew_vals = jnp.exp(ll) * adv
            obj = jnp.mean(rew_vals)

        else:
            t_add_act = jnp.asarray(batch["add_act"], DTYPE)
            t_add_q = jnp.asarray(batch["grad_ranking"], DTYPE)

            # weighted maximum likelihood: E_s [ sum_a q(a|s) log pi(a|s) ], where q(a|s) is normalized per state
            ll = policy.log_prob(t_add_act, suff_stats=suff_stats)
            rew_vals = jnp.exp(t_add_q) * ll
            obj = jnp.mean(jnp.sum(rew_vals, axis=1))

        # the torch version regularised with the detached parameters (no gradient)
        w = jax.lax.stop_gradient(get_flat_params_from(policy))
        return obj - self.l2_reg * jnp.linalg.norm(w)

    def __max_step(self, state: ModelState, batch: Any, v: list[Any]) -> ModelState:
        """
        Solves the minimization problem targeting the policy parameters only.
        """
        policy = state.get("policy")
        lagrange_mults = state.get("lagrange_mults")
        opt = optax.adam(self.outer_lr)
        grads = eqx.filter_grad(lambda p: -self.obj(state, batch, p, lagrange_mults, v))(policy)
        updates, opt_state = opt.update(grads, state.get("opt_outer"), eqx.filter(policy, eqx.is_inexact_array))
        return state.set("policy", eqx.apply_updates(policy, updates)).set("opt_outer", opt_state)

    def __min_step(self, state: ModelState, batch: Any, v: list[Any]) -> tuple[ModelState, Any, list[Any]]:
        """
        Solves the minimization problem targeting the lagrange multiplies. Note that they have to be
        positive.
        """
        policy = state.get("policy")
        opt = optax.adam(self.inner_lr)

        def objective(lagrange_mults: Any) -> Any:
            l_suff_stats = self.shuffle_up_stats(state, batch, policy)
            constraints = self.main_constraints(state, batch, l_suff_stats, policy, v)
            return self.obj(state, batch, policy, lagrange_mults, v), constraints

        (obj, constraints), grads = jax.value_and_grad(objective, has_aux=True)(state.get("lagrange_mults"))
        updates, opt_state = opt.update(grads, state.get("opt_inner"))
        state = state.set("lagrange_mults", state.get("lagrange_mults") + updates).set("opt_inner", opt_state)
        return state, obj, constraints
```

The imports are `jax`, `jax.numpy as jnp`, `equinox as eqx`, `optax`, `PRNGKeyArray`, `ModelState`,
`ModelConfiguration`, `CliPrinter`, `get_flat_params_from` and `DTYPE`. `numpy` and `torch` are gone. Add the module to
`JAXTYPED_MODULES`.

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest -o addopts="" -q tests/test_mpo.py`
Expected: 4 passed. If a case misses rtol 1e-4, compare the first step's gradients against the torch fixture's
recomputation in a 4f7828e worktree before changing tolerances, and report the measured error.

```bash
git add src/abstract_rl/algorithms/continuous/policy_gradient/maximum_a_posteriori_policy_optimization_algorithm.py tests/test_mpo.py tests/conftest.py
git commit -m "Port the MPO M-step to a jitted scan of max and min steps.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `run_mpo_non_parametric` as a jitted, vmapped epoch

**Files:**
- Modify: `src/abstract_rl/run/run_mpo_non_parametric.py` (whole file)
- Modify: `tests/test_training_smoke.py` (re-enable the two MPO smoke tests and use `.eqx` checkpoints)
- Create: `tests/test_run_mpo.py`
- Modify: `src/abstract_rl/scripts/bench_jax.py` (`--alg mpo`)
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces: `run_mpo_non_parametric(conf) -> np.ndarray` of shape `[S, epochs, 2]`, the module-level
  `EPOCH_TRACES: list[int]`, and per-seed `data/eta.txt` logs.

- [ ] **Step 1: Write the failing tests**

`tests/test_run_mpo.py`:

```python
import json

import numpy as np

from abstract_rl.run import run_mpo_non_parametric as module
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric


def _conf(mpo_conf, **overrides):
    return {**mpo_conf, "env_name": "TestPointMass-v0", "max_steps": 30, "env_sample": 120, "val_opt_steps": 5,
            "policy_opt_steps": 5, "tc_size": 600, **overrides}


def test_returns_per_seed_and_epoch_and_eta_log(mpo_conf, tmp_path):
    returns = run_mpo_non_parametric(_conf(mpo_conf, seeds=[1, 2], num_epochs=3))
    assert returns.shape == (2, 3, 2) and np.all(np.isfinite(returns))
    runs = sorted(tmp_path.glob("TestPointMass-v0/mpo/*"))
    assert sorted(json.loads((r / "run.conf").read_text())["seed"] for r in runs) == [1, 2]
    for r in runs:
        assert (r / "model/last/policy.eqx").exists() and (r / "model/last/q_network.eqx").exists()
        etas = np.loadtxt(r / "data/eta.txt")
        assert etas.shape == (3,) and np.all(etas > 0)


def test_epoch_is_traced_once(mpo_conf):
    module.EPOCH_TRACES.clear()
    run_mpo_non_parametric(_conf(mpo_conf, num_epochs=3))
    assert module.EPOCH_TRACES == [120]


def test_memory_respects_tc_size(mpo_conf):
    returns = run_mpo_non_parametric(_conf(mpo_conf, num_epochs=8, tc_size=200))
    assert returns.shape == (1, 8, 2)
    assert module.LAST_MEMORY_SIZES and max(module.LAST_MEMORY_SIZES) <= 200 + 30
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_run_mpo.py`
Expected: FAIL, because the torch loop calls the old APIs.

- [ ] **Step 3: Rewrite `run_mpo_non_parametric.py`**

Follow `run/run_trpo.py`'s structure exactly: `SeedRuns`, `env.init_state()` broadcast, `key`, models via
`runs.build(... fold_in(key, i))`, `add_syncable`, the jitted and vmapped `chunk` with `donate_argnums=0`, the
`(batch_size, num)` compile cache, the budget cap, per-seed logging, plots, `store("last", pre)` and `store_best`,
`try/finally: runs.close()`. Keep the torch file's `cli.header` blocks ("stochastic evaluation", "store model &
generate plots", "e-step", "m-step", "v-step") and names. The MPO specifics are:

```python
    # create the memory and get some settings
    with conf.ns("tc"):
        tc_size = conf["size"]
    with conf.ns("env"):
        num_new_steps = conf["sample"]
        num_traj = conf["traj_evaluations"]
    num_envs = env.num_envs(num_new_steps, max_steps)
    capacity = tc_size + 2 * (num_new_steps + num_envs * max_steps)

    # wiring (unchanged order): policy, alg, q network, filter, ret, res, rnk
    ...
    mc.add_main("q_network_opt", jax.vmap(lambda q: q.init_opt())(q_network))
    for k, v in jax.vmap(alg.init_state)(policy).items():
        mc.add_main(k, v)
    mc.add_main("eta", runs.broadcast(rnk.b.init_state()["eta"]))

    def annotate(state, tc, key):
        k_res, k_ret = jax.random.split(key)
        tc = res.transform_all(state, tc, k_res)
        tc = rnk.transform_all(state, tc)
        return ret.transform_all(state, tc, k_ret)

    # the memory holds every annotation of the operators from the start, so the jitted carry keeps its structure
    empty = TrajectoryCollection.memory(env, capacity, env.observation_dim, env.action_dim)
    template = jax.eval_shape(lambda s: annotate(s, empty, jax.random.key(0)), runs.seed_slice(runs.state(), 0))
    template_ann = {k: jax.ShapeDtypeStruct(v.shape[2:], v.dtype)
                    for k, v in template.trajectories()._annotations.items() if k not in ("insert_id", "time")}
    mc.add_main("tc", runs.broadcast(empty.with_annotations(template_ann)))

    def epoch(state, batch_size):
        key, k_rollout, k_annotate, k_m, k_ret, k_q = jax.random.split(state.get("key"), 6)
        pre = {"policy": state.get("policy"), "q_network": state.get("q_network")}

        # run the stochastic policy and save the trajectories in the memory; filters see the new trajectories once
        state, new_tc, rollout = env.execute_policy(
            state, k_rollout, state.get("policy"), max_steps, batch_size, exploration=True, num_envs=num_envs
        )
        pre["obs_rms"] = state.get("obs_rms")
        new_tc = filter.transform_all(state, new_tc)
        tc = state.get("tc").merge(new_tc, tc_size)

        # e-step: resample extra actions and q values, temperature, ranking
        k_res, _ = jax.random.split(k_annotate)
        tc = res.transform_all(state, tc, k_res)
        state = rnk.hook(state, tc)
        tc = rnk.transform_all(state, tc)

        # m-step
        state, m_step = alg.perform_mult_steps(state, k_m, policy_opt_size, tc, policy_batch_size)

        # v-step
        tc = ret.transform_all(state, tc, k_ret)
        state, v_step = state.get("q_network").perform_mult_sgd(state, k_q, q_opt_steps, tc, q_batch_size, ["tq", "q"])

        # copy target_networks
        state = state.set("tc", tc).sync().set("key", key)
        memory_size = jnp.sum(tc.trajectories().mask)
        return state, {"rollout": rollout, "m_step": m_step, "v_step": v_step, "pre": pre, "eta": state.get("eta"),
                       "memory_size": memory_size}
```

Host side:
- log `eta` per seed: create the field `eta` with dimension 1, then `seed_mc["log"].log({"eta": [float(eta)]})`;
- the plot uses `num_traj`, as torch did;
- extend the module-level `LAST_MEMORY_SIZES: list[int]` with every seed's `memory_size` after each call;
- print the M-step table every 10 steps for seed 0 with `alg.titlestr` / `alg.fmtstr` (P4 values);
- print the v-step table as in run_trpo;
- `EPOCH_TRACES.append(batch_size)` at trace time.

In `scripts/bench_jax.py`, add `--alg {trpo,mpo}` (default trpo), dispatching to the matching run function. Steps per
epoch are `env_sample` for MPO.

Re-enable `test_mpo_pendulum_smoke` and `test_mpo_with_two_dimensional_actions_smoke` in
`tests/test_training_smoke.py`, using the `.eqx` glob and the seed axis as in the TRPO smokes. Add
`"abstract_rl.run.run_mpo_non_parametric"` to `JAXTYPED_MODULES`.

- [ ] **Step 4: Run the tests, a learning check and commit**

Run: `uv run pytest -o addopts="" -q tests/test_run_mpo.py tests/test_training_smoke.py`
Expected: all MPO and TRPO smokes pass, and the eval smokes stay skipped until Task 6.

Learning check, recorded in the report and not committed: Pendulum MPO with the default config, `seeds=[1,2]`, 50
epochs (50k steps per seed) on CPU. Torch reached −150 at about 40k steps. Report the return curve. If it does not
improve beyond −600 by epoch 50, investigate before committing.

Run: `uv run python -m abstract_rl.scripts.bench_jax --alg mpo --env pendulum --seeds 1 4 --epochs 4` and record the
throughput.

```bash
git add src/abstract_rl/run/run_mpo_non_parametric.py src/abstract_rl/scripts/bench_jax.py tests/test_run_mpo.py tests/test_training_smoke.py tests/conftest.py
git commit -m "Train MPO for many seeds at once in one jitted, vmapped epoch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Evaluation, replay and GIFs on the JAX environments

**Files:**
- Modify: `src/abstract_rl/scripts/eval_balance.py`, `src/abstract_rl/scripts/eval_returns.py`,
  `src/abstract_rl/run/run_policy.py`
- Create: `src/abstract_rl/env/render.py`
- Modify: `tests/test_training_smoke.py` (re-enable the eval tests, with `returns[0]` for the seed axis)
- Create: `tests/test_eval.py`

**Interfaces:**
- Produces, in `eval_balance.py`:
  - `load_policy(run_dir, model="last") -> (env_wrapper, policy, state)`. It reloads the `ModelConfiguration`, builds
    `MCMCEnvWrapper`, deserialises `model/<model>/policy.eqx` into an unbatched `select_policy(mc, key(0))` template,
    and loads frozen `obs_rms`: `model/obs_rms.npz` for `last` and `model/best/obs_rms.npz` for `best` (Plan 2 final
    review I-5).
  - `run_episode(env_wrapper, policy, state, seed, frames=None) -> (return, thetas)` uses
    `execute_policy_once(state, jax.random.key(seed), max_steps, policy, opt=True, update_stats=False)`. The return is
    the sum of the raw rewards, and the thetas are `atan2(next_states[:, 1], next_states[:, 0])` over the valid steps.
- Produces, in `render.py`: `render_frames(env_wrapper, trajectory) -> list[np.ndarray]`.
  - Pendulum: a PIL drawing of the pole from `theta`.
  - MuJoCo (Brax and Playground): `mujoco.Renderer` on the environment's `mj_model`, with `qpos` taken from the
    trajectory's environment states.
  - Unsupported envs raise `NotImplementedError` with the env id.
- `eval_returns.evaluate` and `run_policy.run_policy` use `load_policy` and `run_episode` and keep their CLIs.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:

```python
import numpy as np

from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts import eval_balance, eval_returns


def test_eval_uses_frozen_obs_rms_and_seeded_starts(trpo_conf, tmp_path):
    run_trpo({**trpo_conf, "obs_norm": True})
    (run_dir,) = tmp_path.glob("Pendulum-v1/trpo/*")
    env, policy, state = eval_balance.load_policy(str(run_dir))
    stored = np.load(run_dir / "model/obs_rms.npz")
    np.testing.assert_allclose(state.get("obs_rms").mean, stored["mean"], rtol=1e-6)
    r1, t1 = eval_balance.run_episode(env, policy, state, 10_000)
    r2, t2 = eval_balance.run_episode(env, policy, state, 10_000)
    assert r1 == r2 and t1.shape == (200,)  # greedy and seeded: identical episodes, stats never change
    name, returns = eval_returns.evaluate(str(run_dir), episodes=2)
    assert name == "Pendulum-v1" and returns.shape == (2,)


def test_pendulum_gif_frames(trpo_conf, tmp_path):
    run_trpo(trpo_conf)
    (run_dir,) = tmp_path.glob("Pendulum-v1/trpo/*")
    gif = tmp_path / "x.gif"
    eval_balance.save_gif(str(run_dir), str(gif))
    assert gif.exists() and gif.stat().st_size > 0
```

Re-enable `test_eval_balance_runs_on_trained_policy` and `test_best_policy_is_stored_and_evaluated`. In the latter,
compare `best.json` with `returns[0]`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/test_eval.py`
Expected: FAIL, because `load_policy` returns two values and uses `torch.load`.

- [ ] **Step 3: Implement**

Port the three scripts line by line, keeping their CLIs, docstrings and output formats:
- `torch.load` / `load_state_dict` becomes `eqx.tree_deserialise_leaves` into the template;
- the `torch.no_grad()` loops become `run_episode`;
- `frozen_eval_env` rendering becomes `render.render_frames`.

`render.py`:
- **Pendulum:** draw a 250×250 RGB frame per step with PIL, a line from the centre at angle theta, length 100.
- **MuJoCo:** the Brax adapter exposes `env._env.sys.mj_model`, and the Playground adapter exposes `env._env.mj_model`.
  Add a public `mj_model` property to both adapters in `jax_envs.py` and keep each step's `qpos`. To do so, add the
  annotation `qpos` to the rollout only when `env.mj_model` exists, as a static Python check in `execute_policy`.

  Render with:

  ```python
  renderer = mujoco.Renderer(model, 240, 240)
  data = mujoco.MjData(model)
  for q in qpos:
      data.qpos[:] = q
      mujoco.mj_forward(model, data)
      renderer.update_scene(data)
      frames.append(renderer.render())
  ```

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest -o addopts="" -q tests/test_eval.py tests/test_training_smoke.py`
Expected: all pass, and nothing is skipped in `test_training_smoke.py`. Then run the full suite.

```bash
git add src/abstract_rl/scripts/eval_balance.py src/abstract_rl/scripts/eval_returns.py src/abstract_rl/run/run_policy.py \
    src/abstract_rl/env/render.py src/abstract_rl/env/jax_envs.py src/abstract_rl/env/mcmc_env_wrapper.py tests/test_eval.py tests/test_training_smoke.py
git commit -m "Evaluate, replay and render stored policies on the JAX environments.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Remove torch and Gymnasium from the library; README

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `src/abstract_rl/__init__.py`, `src/abstract_rl/misc/mixed.py`,
  `src/abstract_rl/scripts/run_baselines.py`, `tests/conftest.py`, `tests/test_hydra_config.py`,
  `tests/test_acrobot_swingup.py`, `README.md`
- Delete: `tests/fixtures/record_torch_golden.py`. It stays reproducible from commit 4f7828e, and its docstring already
  says so.
- Create: `tests/test_no_torch.py`

**Interfaces:**
- Produces:
  - core dependencies without `torch` and `gymnasium`;
  - a `baselines` extra holding `stable-baselines3`, `sb3-contrib`, `gymnasium[classic-control,mujoco]` and
    `dm-control`;
  - a `dev` group holding `gymnasium[classic-control]` for the Hydra and baseline tests;
  - the Gymnasium acrobot adapter (`env/acrobot_swingup.py`) is kept and registered lazily by `run_baselines.py`, which
    keeps the SB3 acrobot baseline working.

- [ ] **Step 1: Write the failing test**

`tests/test_no_torch.py`:

```python
import importlib
import pkgutil
import sys

import abstract_rl

# the optional SB3 baseline runner imports torch (through SB3) only inside its run function
OPTIONAL = {"abstract_rl.scripts.run_baselines", "abstract_rl.env.acrobot_swingup"}


def test_library_does_not_import_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now raises ImportError
    for m in pkgutil.walk_packages(abstract_rl.__path__, "abstract_rl."):
        if m.name not in OPTIONAL:
            importlib.import_module(m.name)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -o addopts="" -q tests/test_no_torch.py`
Expected: FAIL. `misc/mixed.py` or another module still imports torch.

- [ ] **Step 3: Implement**

- `pyproject.toml`: drop `torch>=2.5` and `gymnasium[classic-control]>=1.0` from `dependencies`, drop the `mujoco` extra
  (MJX is core), and move `dm-control` into `baselines`. Add `"gymnasium[classic-control]>=1.0"` to the dev group, and
  remove `torch` from the dev group if present. Keep the pytorch-cpu index, which the baselines extra uses through SB3.
  Then run `uv sync --extra mps --extra baselines`.
- `src/abstract_rl/__init__.py`: remove the acrobot auto-registration.
- `scripts/run_baselines.py`: import `gymnasium` and call `register_acrobot_swingup()` lazily inside the function that
  builds environments.
- `misc/mixed.py`: no torch import remains.
- `tests/conftest.py`: remove the Gymnasium `PointMass` class and its `gym.register`, keeping the JAX `PointMassEnv`
  registration.
- `tests/test_hydra_config.py`: check `make_env(conf["env_name"])` succeeds instead of `gym.spec`.
- `tests/test_acrobot_swingup.py`: keep the budget test. Move the Gymnasium/dm_control half behind
  `pytest.importorskip("dm_control")` / `pytest.importorskip("gymnasium")`, and keep the JAX wrapper half.
- Delete `tests/fixtures/record_torch_golden.py`, after checking its docstring names commit 4f7828e.
- `README.md`:
  - rewrite the intro and install section for JAX: `uv sync`, `uv sync --extra mps`, `uv sync --extra baselines`;
  - make the "JAX: many seeds at once" section cover MPO, using the Task 5 benchmark numbers;
  - keep the historical torch benchmark tables, labelled "torch implementation (before the JAX port)";
  - list deviations D1–D7 with one line each, linking the spec.

- [ ] **Step 4: Run the checks and commit**

Run:
- `uv run pytest -o addopts="" -q 2>&1 | tail -3`, expecting no failures and no skips except optional-dependency skips;
- `uv run ruff check src tests`;
- `uv run mypy 2>&1 | tail -5`, expecting no errors in `src`.

Also run `uv run python -m abstract_rl.scripts.run env=pendulum alg=mpo seeds=[1,2] alg.num_epochs=2 data_root=/private/tmp/x`,
expecting it to finish.

```bash
git add pyproject.toml uv.lock src tests README.md
git rm tests/fixtures/record_torch_golden.py
git commit -m "Remove torch and Gymnasium from the library; the JAX port is complete.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(When staging `src` and `tests`, check `git status` first, so that the untracked
`src/abstract_rl/env/.jax_envs.py.swp` is not added. Stage the changed files by explicit path.)
