# JAX Port, Plan 1: Spike and Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First, verify that the JAX, Metal, MJX and env stack works (spike S1). Then record torch reference outputs
and port the self-contained numerical core to Equinox and jaxtyping with exact parity. The core is CG, flat params, the
temperature dual, `Batch`, `ModelState`, the MLP, the univariate, diagonal and full-covariance policies, the V and Q
networks, and `VanillaGradient`.

**Architecture:** Every module keeps its file, class names, method names and bodies. It is converted line by line
from torch to `jnp`. Models become `eqx.Module` pytrees, with static configuration in `eqx.field(static=True)`.
Methods that read models through `self.mc.get(...)` take an explicit `state: ModelState` with the same accessor.
Stochastic methods take a leading `key`.

**Tech Stack:** Python 3.12, `jax>=0.11,<0.12` with `jax-mps` 0.11 (Metal), Equinox, Optax, jaxtyping with beartype,
pytest. Torch stays installed until Plan 3, to record the fixtures and to run the parts not yet ported.

**Spec:** `docs/superpowers/specs/2026-10-22-jax-port-design.md`. Read sections 1a, 5, 5a, 7 and 10 before starting.

## Global Constraints

- **Minimal diff (spec 1a).** No renames of files, classes, methods, attributes, config keys or locals. Bodies are
  converted line by line: `torch.` becomes `jnp.`, an in-place update becomes a returned value, and a Python loop on
  traced values becomes `lax`. Comments and docstrings stay, edited only where they describe torch behaviour.
- **Allowed signature changes.** Only explicit inputs and outputs may be added:
  - a leading `key: PRNGKeyArray` on stochastic methods;
  - `state: ModelState` where the torch code called `self.mc.get(...)`;
  - a keyword-only `key` on constructors;
  - a returned `(state, metrics)` where the torch method mutated state and returned `None`.
- **Versions.** `requires-python = ">=3.12,<3.14"`, `.python-version` = `3.12`, `jax>=0.11,<0.12`, `jax-mps>=0.11,<0.12`.
  Never pin `jaxlib` directly.
- **Dtype.** `DTYPE = jnp.float32` everywhere. `jax_enable_x64` is never enabled. The temperature dual also runs in
  float32. This is deviation D4, as revised in Task 3.
- **Non-trainable arrays.** Action bounds, the fixed covariance and similar constants are static Python tuples or
  floats, never JAX arrays. Every inexact array leaf of a model is therefore a trainable parameter, exactly like torch
  `parameters()`.
- **Activations.** Module-level functions only, never lambdas, so that two separately built modules have equal static
  structure.
- **Masked reductions.** Use `jnp.where(mask, x, 0)`, never `x * mask`, so NaN in padded slots can't leak into values
  or gradients.
- **Commit messages.** Sentence case, imperative, ending with a period. Each ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- **Commands.** Run everything with `uv run ...` from the repo root `/Users/kosmitive/git/abstract_rl`.
- **Appending to test files.** When a step appends code to an existing test file, move the appended imports to the top
  of that file, because ruff's E402 is enabled.

## Review Focus

1. **NaN in padded rows.** A padded slot may hold NaN or inf (an unwritten buffer, `atanh(±1)`). Masked means, stds and
   their gradients must stay finite. Covered by Task 6, `test_masked_reductions_ignore_nan_padding`.
2. **float64 inputs.** Numpy float64 inputs, for example Pendulum observations, must give float32 outputs and keep
   parameters float32. Covered by Task 8, `test_float64_states_give_float32_stats`.
3. **Equal static structure between separately built modules.** `ModelState.sync` tree-maps online against target, so
   static fields must compare equal. Covered by Task 8, `test_two_policies_have_equal_static_structure`.
4. **Hydra-style nested lists in `act_fn`.** For example `["tanh", ["none", "softplus"]]` must hash under `jax.jit`.
   Covered by Task 7, `test_mlp_with_per_output_activations_jits`.
5. **`sync_steps > 1`.** Sync happens when `overall_step % s == 0`, checked before the increment, exactly like
   `ModelConfiguration.sync`. Covered by Task 6, `test_sync_respects_sync_steps`.

---

### Task 1: Spike S1, stack compatibility and Metal op coverage (throwaway)

**Files:**
- Create (scratch, not committed): `$SCRATCH/jax-spike/probe.py`. `$SCRATCH` is the executor's scratchpad directory.
- Create (committed): `docs/superpowers/specs/2026-10-22-jax-port-spike.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the resolved version set and the go or no-go table that Plans 2 and 3 pin against.

- [ ] **Step 1: Create the venv and resolve the stack**

```bash
SP="$SCRATCH/jax-spike"; mkdir -p "$SP"
uv venv "$SP/.venv" --python 3.12
uv pip install --python "$SP/.venv" "jax>=0.11,<0.12" "jax-mps>=0.11,<0.12" equinox optax jaxtyping beartype \
    gymnax brax mujoco mujoco-mjx playground
uv pip freeze --python "$SP/.venv" > "$SP/freeze.txt"
```

If resolution fails, remove packages one at a time in this order: `playground`, then `brax`, then `mujoco-mjx`. For
each removed package, find the newest release whose metadata accepts jax 0.11. Inspect it with
`uv pip install --python "$SP/.venv" --dry-run "<pkg>==<ver>"`. Record every pin you had to choose.

- [ ] **Step 2: Write the probe**

```python
"""Spike S1: run with JAX_PLATFORMS=cpu and JAX_PLATFORMS=mps."""
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def timed(name, fn, *args):
    try:
        out = jax.block_until_ready(fn(*args))  # compile
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        print(f"OK   {name:40s} {time.perf_counter() - t:8.4f}s")
        return out
    except Exception as exc:  # noqa: BLE001 - a spike reports every failure
        print(f"FAIL {name:40s} {type(exc).__name__}: {str(exc)[:160]}")


print("platform", os.environ.get("JAX_PLATFORMS"), jax.devices(), jax.__version__)
k = jax.random.key(0)
a = jax.random.normal(k, (8, 8))
spd = a @ a.T + 8 * jnp.eye(8)
L = jnp.linalg.cholesky(spd)
timed("while_loop", jax.jit(lambda x: jax.lax.while_loop(lambda c: c[0] < 10, lambda c: (c[0] + 1, c[1] * 1.01), (0, x))[1]), a)
timed("scan reverse", jax.jit(lambda x: jax.lax.scan(lambda c, y: (c * 0.9 + y, c), 0.0, x, reverse=True)[1]), a[0])
timed("scatter .at[].set", jax.jit(lambda x: x.at[:, jnp.array([0, 2])].set(1.0)), a)
timed("cholesky", jax.jit(jnp.linalg.cholesky), spd)
timed("solve_triangular batched", jax.jit(lambda l, b: jax.scipy.linalg.solve_triangular(l, b, lower=True)),
      jnp.broadcast_to(L, (4, 3, 8, 8)), jnp.ones((4, 3, 8, 1)))
timed("logsumexp", jax.jit(lambda x: jax.scipy.special.logsumexp(x, axis=1)), a)
timed("random.normal", jax.jit(lambda kk: jax.random.normal(kk, (1024, 20))), k)
timed("jvp(grad) hvp", jax.jit(lambda x, v: jax.jvp(jax.grad(lambda y: jnp.sum(jnp.tanh(y) ** 2)), (x,), (v,))[1]), a[0], a[1])
timed("argsort+cumsum", jax.jit(lambda x: jnp.cumsum(jnp.sort(x[0])[::-1])), a)

# 16 seeds x small MLP Adam step (vmapped training)
def init(kk):
    return eqx.nn.MLP(3, 1, 64, 2, key=kk)
models = eqx.filter_vmap(init)(jax.random.split(k, 16))
opt = optax.adam(1e-3)
opt_state = eqx.filter_vmap(lambda m: opt.init(eqx.filter(m, eqx.is_inexact_array)))(models)
x = jax.random.normal(k, (256, 3))
@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None))
def step(m, s, xx):
    g = eqx.filter_grad(lambda mm: jnp.mean(jax.vmap(mm)(xx) ** 2))(m)
    u, s = opt.update(g, s)
    return eqx.apply_updates(m, u), s
timed("16-seed MLP adam step", step, models, opt_state, x)

# gymnax: 20 envs x 200 steps
import gymnax
for env_id in ("Pendulum-v1", "MountainCarContinuous-v0"):
    env, params = gymnax.make(env_id)
    def roll(kk, env=env, params=params):
        obs, st = jax.vmap(env.reset, (0, None))(jax.random.split(kk, 20), params)
        def body(c, kk2):
            st = c
            a = jnp.zeros((20,) + env.action_space(params).shape)
            o, st, r, d, _ = jax.vmap(env.step_env, (0, 0, 0, None))(jax.random.split(kk2, 20), st, a, params)
            return st, (r, d)
        return jax.lax.scan(body, st, jax.random.split(kk, 200))[1]
    timed(f"gymnax {env_id} 20x200", jax.jit(roll), k)

# brax mjx: 16 envs x 1000 steps
from brax import envs as brax_envs
for env_id in ("inverted_pendulum", "inverted_double_pendulum", "halfcheetah"):
    try:
        env = brax_envs.get_environment(env_id, backend="mjx")
        print(env_id, "obs", env.observation_size, "act", env.action_size)
        def roll(kk, env=env):
            st = jax.vmap(env.reset)(jax.random.split(kk, 16))
            def body(c, _):
                c = jax.vmap(env.step)(c, jnp.zeros((16, env.action_size)))
                return c, (c.reward, c.done)
            return jax.lax.scan(body, st, None, length=1000)[1]
        timed(f"brax-mjx {env_id} 16x1000", jax.jit(roll), k)
    except Exception as exc:  # noqa: BLE001
        print("FAIL", env_id, exc)

# playground acrobot: 16 envs x 1000 steps
try:
    from mujoco_playground import registry
    env = registry.load("AcrobotSwingup")
    print("AcrobotSwingup obs", env.observation_size, "act", env.action_size)
    def roll(kk):
        st = jax.vmap(env.reset)(jax.random.split(kk, 16))
        def body(c, _):
            c = jax.vmap(env.step)(c, jnp.zeros((16, env.action_size)))
            return c, (c.reward, c.done)
        return jax.lax.scan(body, st, None, length=1000)[1]
    timed("playground AcrobotSwingup 16x1000", jax.jit(roll), k)
except Exception as exc:  # noqa: BLE001
    print("FAIL playground", exc)
```

- [ ] **Step 3: Run the probe on both backends**

```bash
JAX_PLATFORMS=cpu "$SP/.venv/bin/python" -I "$SP/probe.py" | tee "$SP/cpu.txt"
JAX_PLATFORMS=mps "$SP/.venv/bin/python" -I "$SP/probe.py" | tee "$SP/mps.txt"
```

Expected: every line on CPU reads `OK`. MPS lines may read `FAIL`, and each failure is a finding to record.

- [ ] **Step 4: Write the results document**

Create `docs/superpowers/specs/2026-10-22-jax-port-spike.md` with these sections:
1. "Resolved versions": the relevant lines of `freeze.txt` (jax, jaxlib, jax-mps, equinox, optax, jaxtyping, gymnax,
   brax, mujoco, mujoco-mjx, playground) and any manual pins from Step 1.
2. "Op coverage": a table with columns probe, cpu, mps, and the error text.
3. "Throughput": one table with the CPU and MPS seconds for each rollout probe.
4. "Decisions":
   - Is MPS usable for (a) networks only and (b) MJX envs, each as yes or no?
   - Which env ids are available?
   - Which pins go into `pyproject.toml` in Task 3?

  If MPS fails any op used by the policies (cholesky, solve_triangular, scatter), Task 3 still adds the `mps` extra, but
  the document marks `jax_platform: mps` as unsupported for those features.

- [ ] **Step 5: Commit the results only**

```bash
git add docs/superpowers/specs/2026-10-22-jax-port-spike.md
git commit -m "Record the JAX stack spike results.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**Gate:** stop and report to the human partner if the CPU column has any `FAIL`, or if no brax version resolves with
jax 0.11. Plans 2 and 3 depend on these answers.

---

### Task 2: Record the torch golden fixtures

This runs on the current torch code, before anything is ported.

**Files:**
- Create: `tests/fixtures/record_torch_golden.py`
- Create: `tests/fixtures/torch_golden.npz` (generated)

**Interfaces:**
- Consumes: the current torch modules, `tests/conftest.py` (for `TestPointMass-v0`),
  `abstract_rl.scripts.{trpo,mpo}_pendulum.run_config`.
- Produces: `torch_golden.npz` with these key families (array names after the slash):
  - `mlp_<case>/{x,y,params/<name>}`
  - `<policy_case>/{states,stats,actions,lp,single,lp1,mode,lls,a,kl_<mode>_<est>,lr,entropy?,params/<n>,target/<n>,grads/<n>}`,
    plus `<policy_case>_lr/grads/<n>`
  - `v_net/{states,tv,forward,obj,params/<n>,grads/<n>,adam3/params/<n>}`
  - `q_net/{states,actions,add_actions,tq,q_val,q_val_multi,obj,params/<n>,grads/<n>}`
  - `adv_<truncated|terminated>/{all_states,rewards,dones,tv,td,a}`, `adv_v/params/<n>`
  - `retrace_<case>/{all_states,actions,rewards,dones,log_likelihoods,q,tq,add_act,add_q,add_pi}`,
    `retrace_policy/params/<n>`, `retrace_q/params/<n>`
  - `cg/{A,b,x}`
  - `trpo_<uni|diag>/{states,actions,lls,a,before/params/<n>,after/params/<n>}`
  - `dual_<cold|warm>/{q,eta0,eta}`
  - `mpo_<uni|split|diag>/{states,add_act,grad_ranking,step<i>/params/<n>,step<i>/lagrange,target/params/<n>}`
  - `filter/{xs,means,stds,outs}`, `obsnorm/{xs,means,vars,outs}`, `meta/commit`

  Parameter names are torch `named_parameters()` names: `cov`, `network.layer_<i>.weight`, `net.layer_<i>.bias`,
  `fc_net.layer_<i>.weight`, `layer_<i>.weight`.

- [ ] **Step 1: Write the recorder**

```python
"""Record reference outputs of the PyTorch implementation for the JAX parity tests (tests/test_parity_*.py).

Run on the last torch commit. After the port removes torch, re-record from a worktree of that commit:
    git worktree add ../abstract_rl_torch 8d3dc51
    cd ../abstract_rl_torch && uv run python tests/fixtures/record_torch_golden.py <repo>/tests/fixtures/torch_golden.npz
"""

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conftest  # noqa: E402,F401  (registers TestPointMass-v0)

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (  # noqa: E402
    MPOAlgorithm,
)
from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm  # noqa: E402
from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient  # noqa: E402
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature  # noqa: E402
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration  # noqa: E402
from abstract_rl.data_structures.temporal_difference_data.batch import Batch  # noqa: E402
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder  # noqa: E402
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper  # noqa: E402
from abstract_rl.misc.data_logger import DataLogger  # noqa: E402
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter  # noqa: E402
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron  # noqa: E402
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator  # noqa: E402
from abstract_rl.operator.resample_actions_operator import ResampleOperator  # noqa: E402
from abstract_rl.operator.retrace_operator import RetraceOperator  # noqa: E402
from abstract_rl.operator.td_operator import TemporalDifferenceOperator  # noqa: E402
from abstract_rl.policy.continuous.select_policy import select_policy  # noqa: E402
from abstract_rl.scripts.mpo_pendulum import run_config as mpo_config  # noqa: E402
from abstract_rl.scripts.trpo_pendulum import run_config as trpo_config  # noqa: E402
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet  # noqa: E402
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet  # noqa: E402

OUT: dict[str, np.ndarray] = {}


def put(prefix: str, **arrays) -> None:
    for k, v in arrays.items():
        OUT[f"{prefix}/{k}"] = v.detach().cpu().numpy().copy() if torch.is_tensor(v) else np.asarray(v)


def put_params(prefix: str, module: torch.nn.Module, grads: bool = False, sub: str = "params") -> None:
    for n, p in module.named_parameters():
        OUT[f"{prefix}/{sub}/{n}"] = p.detach().numpy().copy()
        if grads:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            OUT[f"{prefix}/grads/{n}"] = g.detach().numpy().copy()


def make_mc(root: str, base: dict, **overrides) -> ModelConfiguration:
    conf = {**copy.deepcopy(base), "data_root": root, **overrides}
    mc = ModelConfiguration(conf)
    mc.add_main("log", DataLogger(mc.data_dir))
    with mc["conf"].ns("env"):
        mc.add_environment(MCMCEnvWrapper(mc))
    return mc


def perturb(module: torch.nn.Module, scale: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))


def make_traj(env, length: int, done_last: int, rng: np.random.Generator):
    high = np.asarray(env.action_space.high, dtype=np.float64)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    builder = TrajectoryBuilder(0, env, rng.normal(size=env.observation_dim))
    for t in range(length):
        act = rng.uniform(low * 0.9, high * 0.9)
        done = int(done_last and t == length - 1)
        builder.observe(act, float(rng.normal()), rng.normal(size=env.observation_dim), float(rng.normal() - 1), done)
    return builder.finalize()


def record_mlps(root: str) -> None:
    mc = make_mc(root, trpo_config, alg="trpo")
    cases = {
        "tanh_tanh_none": ([3, 64, 64, 2], ["tanh", "tanh", "none"], False, "xavier_gaussian"),
        "ln_elu": ([3, 32, 32, 1], ["elu", "elu", "none"], True, "xavier_gaussian"),
        "split_head": ([3, 16, 2], ["tanh", ["none", "softplus"]], False, "xavier_gaussian"),
        "single_act": ([3, 8, 8, 1], ["relu"], False, "orthogonal"),
        "prelu_lrelu": ([3, 8, 8, 1], ["prelu", "lrelu", "sigmoid"], False, "gaussian"),
    }
    for name, (structure, act_fn, layer_norm, init) in cases.items():
        torch.manual_seed(0)
        net = MultilayerPerceptron(mc, np.array(structure), act_fn, layer_norm, init)
        x = torch.randn(16, structure[0])
        put(f"mlp_{name}", x=x, y=net.forward(x))
        put_params(f"mlp_{name}", net)


POLICY_CASES = {
    "uni_single_squash": ("Pendulum-v1", {"policy_cov_type": "single", "policy_squash": True}),
    "uni_state_nosquash": ("Pendulum-v1", {"policy_cov_type": "state", "policy_squash": False}),
    "uni_fixed_squash": ("Pendulum-v1", {"policy_cov_type": "fixed", "policy_squash": True, "policy_sigma": 0.5}),
    "diag_single_squash": ("TestPointMass-v0", {"policy_cov_type": "single", "policy_squash": True}),
    "diag_state_squash": ("TestPointMass-v0", {"policy_cov_type": "state", "policy_squash": True}),
    "diag_full_squash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": True}),
    "diag_full_nosquash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": False}),
}


def record_policy(root: str, name: str, env: str, overrides: dict) -> None:
    torch.manual_seed(1)
    mc = make_mc(root, trpo_config, alg="trpo", env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    target = mc.get("policy", True)
    perturb(target, 0.05)
    n = 32
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        stats = policy.forward(states)
        actions = policy.sample_actions(suff_stats=stats, num_actions=5)
        single = policy.sample_actions(suff_stats=stats, num_actions=1).reshape(n, -1)
        lp = policy.log_prob(actions, suff_stats=stats)
        lp1 = policy.log_prob(single, states=states)
        mode = policy.mode(states=states)
    lls = lp1.reshape(n, 1) + 0.1 * torch.randn(n, 1)
    a = torch.randn(n)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), single.numpy(), lls.numpy(), zeros, states.numpy(), zeros, {"a": a.numpy()})
    kls = {
        f"kl_{m.split('-')[0]}_{e}": policy.kl_divergence(batch, mode=m, est=e).detach()
        for m in ("i-projection", "m-projection")
        for e in ("mean", "sum")
    }
    put(name, states=states, stats=stats, actions=actions, lp=lp, single=single, lp1=lp1, mode=mode, lls=lls, a=a, **kls)
    if not policy.squash:
        put(name, entropy=policy.entropy(batch).detach())
    policy.zero_grad()
    policy.kl_divergence(batch, mode="m-projection", est="mean").backward()
    put_params(name, policy, grads=True)
    put_params(name, target, sub="target")
    policy.zero_grad()
    lr = policy.likelihood_ratio(batch)
    lr.backward()
    put(name, lr=lr.detach())
    put_params(f"{name}_lr", policy, grads=True)


def record_values(root: str) -> None:
    torch.manual_seed(2)
    mc = make_mc(root, trpo_config, alg="trpo")
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc)
    states, tv = torch.randn(40, 3), torch.randn(40)
    zeros = np.zeros((40, 1))
    batch = Batch(states.numpy(), zeros, zeros, zeros, states.numpy(), zeros, {"tv": tv.numpy()})
    obj = v.tr_obj(batch)
    obj.backward()
    put("v_net", states=states, tv=tv, forward=v.forward(states).detach(), obj=obj.detach())
    put_params("v_net", v, grads=True)
    for s in range(3):
        v.perform_adam_step(s, batch)
    put_params("v_net", v, sub="adam3/params")

    torch.manual_seed(2)
    mc = make_mc(root, mpo_config, alg="mpo")
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc)
    states, actions, add_actions = torch.randn(40, 3), torch.rand(40, 1) * 4 - 2, torch.rand(40, 4) * 4 - 2
    tq = torch.randn(40, 1)
    batch = Batch(states.numpy(), actions.numpy(), zeros, zeros, states.numpy(), zeros, {"tq": tq.numpy()})
    obj = q.tr_obj(batch)
    obj.backward()
    put(
        "q_net",
        states=states,
        actions=actions,
        add_actions=add_actions,
        tq=tq,
        q_val=q.q_val(states, actions).detach(),
        q_val_multi=q.q_val_multi(states, add_actions).detach(),
        obj=obj.detach(),
    )
    put_params("q_net", q, grads=True)


def record_advantages(root: str) -> None:
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    mc = make_mc(root, trpo_config, alg="trpo")
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc)
        mc.add_syncable("v_network", v, 1)
    with mc["conf"].ns("tdop"):
        td = TemporalDifferenceOperator(mc)
    with mc["conf"].ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
    for case, (length, done_last) in {"truncated": (12, 0), "terminated": (7, 1)}.items():
        traj = make_traj(mc.env, length, done_last, rng)
        td.transform(traj)
        gae.transform(traj)
        put(f"adv_{case}", all_states=traj.all_states, rewards=traj.rewards, dones=traj.dones,
            tv=traj["tv"], td=traj["td"], a=traj["a"])
    put_params("adv_v", v)


def _deterministic_rsample(self, sample_shape=torch.Size()):  # noqa: B008
    return self.loc.expand(self._extended_shape(sample_shape)).clone()


def record_retrace(root: str) -> None:
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    Normal.rsample, MultivariateNormal.rsample = _deterministic_rsample, _deterministic_rsample
    mc = make_mc(root, mpo_config, alg="mpo")
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc)
        mc.add_syncable("q_network", q, 1)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    with mc["conf"].ns("res"):
        res = ResampleOperator(mc)
    for case, (length, done_last) in {"truncated": (12, 0), "terminated": (7, 1)}.items():
        traj = make_traj(mc.env, length, done_last, rng)
        res.transform(traj)
        ret.transform(traj)
        put(f"retrace_{case}", all_states=traj.all_states, actions=traj.actions, rewards=traj.rewards,
            dones=traj.dones, log_likelihoods=traj.log_likelihoods, q=traj["q"], tq=traj["tq"],
            add_act=traj["add_act"], add_q=traj["add_q"], add_pi=traj["add_pi"])
    put_params("retrace_policy", policy)
    put_params("retrace_q", q)


def record_cg() -> None:
    torch.manual_seed(5)
    n = 8
    m = torch.randn(n, n)
    a = m @ m.T + n * torch.eye(n)
    b = torch.randn(n)
    put("cg", A=a, b=b, x=conjugate_gradient(lambda v: a @ v, b, 10, 1e-10))


def record_trpo(root: str, name: str, env: str) -> None:
    torch.manual_seed(6)
    mc = make_mc(root, trpo_config, alg="trpo", env_name=env)
    with mc["conf"].ns("val"):
        mc.add_syncable("v_network", FullyVNeuralNet(mc), 1)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with mc["conf"].ns("trpo"):
        alg = TRPOAlgorithm(mc)
    n = 64
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        actions = policy.sample_actions(states=states).reshape(n, -1)
        lls = policy.log_prob(actions, states=states).reshape(n, 1)
    a = torch.randn(n)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), actions.numpy(), lls.numpy(), zeros, states.numpy(), zeros, {"a": a.numpy()})
    put(name, states=states, actions=actions, lls=lls, a=a)
    put_params(name, policy, sub="before/params")
    alg.perform_tr_sgd(0, batch, batch)
    put_params(name, policy, sub="after/params")


def record_dual() -> None:
    torch.manual_seed(7)
    q = torch.randn(256, 20) * 10
    for case, eta0 in {"cold": 1.0, "warm": 3.0}.items():
        put(f"dual_{case}", q=q, eta0=np.float64(eta0), eta=find_optimal_temperature(q, 0.1, eta0, 0.05))


def record_mpo(root: str, name: str, overrides: dict) -> None:
    torch.manual_seed(8)
    mc = make_mc(root, mpo_config, alg="mpo", **overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    target = mc.get("policy", True)
    perturb(target, 0.05)
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    n = 64
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        add_act = target.sample_actions(states=states, num_actions=20)
    grad_ranking = torch.log_softmax(torch.randn(n, 20), dim=1)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), zeros, zeros, zeros, states.numpy(), zeros,
                  {"add_act": add_act.numpy(), "grad_ranking": grad_ranking.numpy(), "add_pi": np.zeros((n, 20))})
    put(name, states=states, add_act=add_act, grad_ranking=grad_ranking)
    put_params(name, target, sub="target/params")
    put_params(name, policy, sub="step0/params")
    put(f"{name}/step0", lagrange=torch.stack([lm.detach() for lm in alg.lagrange_mults]))
    for s in range(3):
        alg.perform_step(s, batch)
        put_params(name, policy, sub=f"step{s + 1}/params")
        put(f"{name}/step{s + 1}", lagrange=torch.stack([lm.detach() for lm in alg.lagrange_mults]))


def record_filters() -> None:
    rng = np.random.default_rng(9)
    xs = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([0.0, 2.0, -1.0])
    f = RunningCenteredFilter("states", 3)
    means, stds, outs = [], [], []
    for x in xs:
        outs.append(f(x))
        means.append(f.mean.copy())
        stds.append(f.std.copy())
    put("filter", xs=xs, means=np.stack(means), stds=np.stack(stds), outs=np.stack(outs))

    from gymnasium.wrappers.utils import RunningMeanStd

    rms = RunningMeanStd(shape=(3,))
    means, variances, outs = [], [], []
    for x in xs:
        rms.update(np.array([x]))
        outs.append((x - rms.mean) / np.sqrt(rms.var + 1e-8))
        means.append(rms.mean.copy())
        variances.append(rms.var.copy())
    put("obsnorm", xs=xs, means=np.stack(means), vars=np.stack(variances), outs=np.stack(outs))


def main(out_path: str) -> None:
    with tempfile.TemporaryDirectory() as root:
        record_mlps(root)
        for name, (env, overrides) in POLICY_CASES.items():
            record_policy(root, name, env, overrides)
        record_values(root)
        record_advantages(root)
        record_cg()
        record_trpo(root, "trpo_uni", "Pendulum-v1")
        record_trpo(root, "trpo_diag", "TestPointMass-v0")
        record_dual()
        record_mpo(root, "mpo_uni", {})
        record_mpo(root, "mpo_split", {"mpo_coordinate_ascent_objective": True,
                                       "mpo_coordinate_ascent_constraint": True,
                                       "mpo_trust_regions": [0.01, 0.02]})
        record_mpo(root, "mpo_diag", {"env_name": "TestPointMass-v0"})
        record_filters()
        record_retrace(root)  # last: it patches torch sampling to be deterministic
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    OUT["meta/commit"] = np.array(commit)
    np.savez_compressed(out_path, **OUT)
    print(f"wrote {len(OUT)} arrays to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run it**

Run: `uv run python tests/fixtures/record_torch_golden.py tests/fixtures/torch_golden.npz`

Expected: `wrote <N> arrays to tests/fixtures/torch_golden.npz`, with N > 300 and no exception. If a conf key is
missing, add it to the matching `make_mc(...)` override in the recorder. Never change library code in this task.

- [ ] **Step 3: Sanity-check the file**

Run: `uv run python -c "import numpy as np; d=np.load('tests/fixtures/torch_golden.npz'); print(sorted({k.split('/')[0] for k in d.files}))"`

Expected: the list contains all the families from **Produces**.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/record_torch_golden.py tests/fixtures/torch_golden.npz
git commit -m "Record torch reference outputs for the JAX parity tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Dependencies, Python pin, typing aliases (P3), jaxtyping hook, spec revision

**Files:**
- Modify: `pyproject.toml`, `.python-version`
- Create: `src/abstract_rl/misc/typing.py`
- Create: `tests/golden.py`
- Modify: `tests/conftest.py` (top of the file only)
- Modify: `docs/superpowers/specs/2026-10-22-jax-port-design.md` (D4, section 7, R5 wording)
- Test: `tests/test_typing.py`

**Interfaces:**
- Produces:
  - `abstract_rl.misc.typing`: `DTYPE`, `Key`, and the aliases below.
  - `tests/golden.py`: `golden(case) -> dict[str, np.ndarray]`, `sub(d, prefix) -> dict`,
    `load_named(module, params) -> module`, `named_parameters(module) -> dict[str, np.ndarray]`.
  - `tests/conftest.py`: `JAXTYPED_MODULES: list[str]`, which later tasks extend.

- [ ] **Step 1: Update `pyproject.toml`**

Change these values:
- `requires-python = ">=3.12,<3.14"`;
- add the classifier `"Programming Language :: Python :: 3.12"`;
- `[tool.ruff] target-version = "py312"`;
- `[tool.mypy] python_version = "3.12"`.

Add to `dependencies`, keeping `torch>=2.5` until Plan 3, with the versions from the spike document:

```toml
    "jax>=0.11,<0.12",
    "equinox>=0.13",
    "optax>=0.2",
    "jaxtyping>=0.3",
    "beartype>=0.19",
```

Add the optional extra:

```toml
mps = [
    "jax-mps>=0.11,<0.12; sys_platform == 'darwin' and platform_machine == 'arm64'",
]
```

Write `.python-version` as `3.12`.

- [ ] **Step 2: Sync and confirm the baseline is still green on 3.12**

Run: `uv sync --extra mps --extra mujoco --extra baselines --extra dm-control && uv run pytest`

Expected: the same pass count as before the change, and the venv on Python 3.12
(`uv run python -V` prints `Python 3.12.x`). If dm-control or labmaze fail to build on 3.12, drop `--extra dm-control`
from this command and note it in the commit message.

- [ ] **Step 3: Write the typing module**

`src/abstract_rl/misc/typing.py`:

```python
"""Shared jaxtyping aliases and the float dtype of the JAX port.

Dimension names: D states in a batch, Ds state dim, A action dim, K actions per state, S suff stats, T time, E episodes.
"""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

DTYPE = jnp.float32

Key = PRNGKeyArray
Scalar = Float[Array, ""]
States = Float[Array, "D Ds"]
SuffStats = Float[Array, "D S"]
Actions = Float[Array, "D A"]
MultiActions = Float[Array, "D K A"]
UniActions = Float[Array, "D K"]
LogProbs = Float[Array, "D K"]
Mask = Bool[Array, "D"]
Step = Int[Array, ""]

__all__ = ["DTYPE", "Actions", "Array", "Bool", "Float", "Int", "Key", "LogProbs", "Mask", "MultiActions",
           "Scalar", "States", "Step", "SuffStats", "UniActions"]
```

- [ ] **Step 4: Install the jaxtyping hook in the conftest**

Insert at the very top of `tests/conftest.py`, before every other import:

```python
from jaxtyping import install_import_hook

# modules ported to JAX are checked against their jaxtyping annotations at runtime (beartype)
JAXTYPED_MODULES = ["abstract_rl.misc.typing"]
install_import_hook(JAXTYPED_MODULES, "beartype.beartype")
```

- [ ] **Step 5: Write the golden helper**

`tests/golden.py`:

```python
"""Access to the torch reference outputs (tests/fixtures/torch_golden.npz) and torch <-> Equinox parameter naming."""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

_GOLDEN = np.load(Path(__file__).parent / "fixtures" / "torch_golden.npz")
MLP_ATTRS = ("network", "net", "fc_net")


def golden(case: str) -> dict[str, np.ndarray]:
    """All arrays of one case, keyed by the name after ``<case>/``."""
    p = case + "/"
    return {k[len(p) :]: _GOLDEN[k] for k in _GOLDEN.files if k.startswith(p)}


def sub(d: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    p = prefix + "/"
    return {k[len(p) :]: v for k, v in d.items() if k.startswith(p)}


def named_parameters(module: Any) -> dict[str, np.ndarray]:
    """torch ``named_parameters()`` names of an Equinox model (or of its gradient pytree)."""
    out: dict[str, Any] = {}
    if getattr(module, "cov", None) is not None:
        out["cov"] = module.cov
    mlps = [(f"{a}.", getattr(module, a)) for a in MLP_ATTRS if getattr(module, a, None) is not None]
    if hasattr(module, "layers"):
        mlps = [("", module)]
    for prefix, mlp in mlps:
        for i, layer in enumerate(mlp.layers):
            out[f"{prefix}layer_{i}.weight"] = layer.weight
            out[f"{prefix}layer_{i}.bias"] = layer.bias
    return {k: np.asarray(v) for k, v in out.items()}


def load_named(module: Any, params: dict[str, np.ndarray]) -> Any:
    """Copy torch-named parameters into an Equinox model."""
    for name, value in params.items():
        parts = name.split(".")
        if parts == ["cov"]:
            module = eqx.tree_at(lambda m: m.cov, module, jnp.asarray(value))
            continue
        attr = None if parts[0].startswith("layer_") else parts[0]
        layer_name, field = parts[-2], parts[-1]
        i = int(layer_name.split("_")[1])

        def where(m, attr=attr, i=i, field=field):
            mlp = m if attr is None else getattr(m, attr)
            return getattr(mlp.layers[i], field)

        module = eqx.tree_at(where, module, jnp.asarray(value))
    return module
```

- [ ] **Step 6: Write the failing typing test**

`tests/test_typing.py`:

```python
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import jaxtyped
from beartype import beartype

from abstract_rl.misc.typing import DTYPE, States, SuffStats


def test_dtype_is_float32():
    assert DTYPE == jnp.float32


def test_aliases_check_shapes():
    @jaxtyped(typechecker=beartype)
    def f(s: States) -> SuffStats:
        return s[:, :2]

    assert f(jnp.zeros((4, 3))).shape == (4, 2)
    with pytest.raises(BeartypeCallHintParamViolation):
        f(jnp.zeros((4,)))
```

- [ ] **Step 7: Run it**

Run: `uv run pytest tests/test_typing.py -v`
Expected: 2 passed.

- [ ] **Step 8: Revise the spec**

Edit `docs/superpowers/specs/2026-10-22-jax-port-design.md`:
- In section 7, replace the "The temperature dual is float64 on CPU" bullet and its sub-bullets with:
  "**The temperature dual runs in float32 on every backend.** `jax_enable_x64` is never enabled, because x64 would
  turn MJX, Brax and the networks into float64 on CPU."
- Replace the D4 row of section 10 with:
  "D4 | The temperature dual runs in float32 instead of float64. Parity is checked at rtol 1e-3. | x64 globally would
  make MJX and all networks float64, and Metal has no float64."
- In R5, replace "This is the one base-class edit, and every operator subclass is otherwise untouched." with:
  "The operator subclasses keep their bodies. Their model lookups move from `__init__` into `transform`, as
  `state.get(...)`."

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .python-version uv.lock src/abstract_rl/misc/typing.py tests/golden.py tests/conftest.py \
    tests/test_typing.py docs/superpowers/specs/2026-10-22-jax-port-design.md
git commit -m "Add the JAX dependencies, the Python 3.12 pin and the shared jaxtyping aliases.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Conjugate gradient and flat parameters

**Files:**
- Modify: `src/abstract_rl/algorithms/mixed/conjugate_gradient.py` (whole file)
- Modify: `src/abstract_rl/misc/flat_params.py` (whole file)
- Modify: `tests/test_conjugate_gradient.py` (whole file)
- Modify: `tests/test_training_smoke.py` (add a module skip)
- Modify: `tests/conftest.py` (extend `JAXTYPED_MODULES`)
- Create: `tests/test_parity_core.py`

**Interfaces:**
- Produces:
  - `conjugate_gradient(A: Callable[[Float[Array, "n"]], Float[Array, "n"]], b: Float[Array, "n"], max_k: int = 10, tol: float = 1e-10) -> Float[Array, "n"]`.
    It is jittable and uses `lax.while_loop`.
  - `get_flat_params_from(model) -> Float[Array, "n"]`.
  - `set_flat_params_to(model, flat_params) -> model`. It **returns** the new model, where torch copied in place.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_conjugate_gradient.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient


def _solve(a, b, max_k=20, tol=1e-10):
    x = jax.jit(lambda bb: conjugate_gradient(lambda v: a @ v, bb, max_k, tol))(b)
    return x, jnp.linalg.norm(a @ x - b)


def test_identity():
    n = 3
    _, residual = _solve(jnp.eye(n), jnp.ones(n))
    assert residual < 1e-3


def test_random_spd_matrix():
    m = jax.random.normal(jax.random.key(0), (10, 10))
    a = m @ m.T + 10 * jnp.eye(10)
    b = jax.random.normal(jax.random.key(1), (10,))
    x, residual = _solve(a, b, max_k=50)
    assert residual < 1e-3
    np.testing.assert_allclose(x, jnp.linalg.solve(a, b), rtol=1e-3, atol=1e-4)


def test_stops_at_tolerance():
    # the exact solution of a diagonal system is found in at most n iterations, extra iterations must not change it
    a = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    b = jnp.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(_solve(a, b, max_k=3, tol=1e-12)[0], _solve(a, b, max_k=50, tol=1e-12)[0], atol=1e-6)
```

Create `tests/test_parity_core.py`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from golden import golden

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to


def test_cg_matches_torch():
    g = golden("cg")
    a = jnp.asarray(g["A"])
    x = conjugate_gradient(lambda v: a @ v, jnp.asarray(g["b"]), 10, 1e-10)
    np.testing.assert_allclose(x, g["x"], rtol=1e-4, atol=1e-5)


def test_flat_params_round_trip():
    mlp = eqx.nn.MLP(3, 2, 4, 1, key=jax.random.key(0))
    flat = get_flat_params_from(mlp)
    assert flat.ndim == 1 and flat.size == sum(x.size for x in jax.tree.leaves(eqx.filter(mlp, eqx.is_inexact_array)))
    back = set_flat_params_to(mlp, flat + 1.0)
    np.testing.assert_allclose(get_flat_params_from(back), flat + 1.0)
    assert mlp.layers[0].weight is not back.layers[0].weight  # the input is not mutated
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_conjugate_gradient.py tests/test_parity_core.py -v`
Expected: FAIL. The torch `conjugate_gradient` can't consume `jnp` arrays (`TypeError`), and `set_flat_params_to`
returns `None`.

- [ ] **Step 3: Convert `conjugate_gradient.py`**

```python
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def conjugate_gradient(
    A: Callable[[Float[Array, "n"]], Float[Array, "n"]], b: Float[Array, "n"], max_k: int = 10, tol: float = 1e-10
) -> Float[Array, "n"]:
    """
    Implementation of conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py.
    :param A: A function f : R[n] -> R[n], lambda to evaluate matrix vector product.
    :param b: The result of the vector multiplication.
    :param max_k: The maximum number of iterations.
    :param damping: Damping for vector products.
    :return: The found solution for the equation,
    """

    # start with residual
    x = jnp.zeros_like(b)
    r = b
    p = b

    k = 0
    rdotr = jnp.dot(r, r)

    Carry = tuple[Int[Array, ""], Float[Array, "n"], Float[Array, "n"], Float[Array, "n"], Float[Array, ""], Bool[Array, ""]]

    def cond(carry: Carry) -> Bool[Array, ""]:
        k, _, _, _, _, converged = carry
        return (k < max_k) & ~converged

    def body(carry: Carry) -> Carry:
        k, x, r, p, rdotr, _ = carry

        # remaining fields
        _Avp = A(p)
        alpha = rdotr / jnp.dot(p, _Avp)

        # update x and r
        x = x + alpha * p
        r = r - alpha * _Avp

        new_rdotr = jnp.dot(r, r)
        beta = new_rdotr / rdotr
        p = beta * p + r
        rdotr = new_rdotr

        # calculate next conjugate gradient, the torch loop breaks once rdotr <= tol
        return k + 1, x, r, p, rdotr, rdotr <= tol

    init: Carry = (jnp.asarray(k), x, r, p, rdotr, jnp.asarray(False))
    _, x, _, _, _, _ = jax.lax.while_loop(cond, body, init)
    return x
```

- [ ] **Step 4: Convert `flat_params.py`**

```python
from typing import Any

import equinox as eqx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float


def get_flat_params_from(model: Any) -> Float[Array, "n"]:
    """Extract flat parameters form an equinox model (every inexact array leaf is a parameter)."""

    flat_params, _ = ravel_pytree(eqx.filter(model, eqx.is_inexact_array))
    return flat_params


def set_flat_params_to(model: Any, flat_params: Float[Array, "n"]) -> Any:
    """Set flat parameters to an equinox model. Models are immutable, so the updated model is returned."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    _, unravel = ravel_pytree(params)
    return eqx.combine(unravel(flat_params), static)
```

- [ ] **Step 5: Skip the torch training smoke tests until Plan 2 re-enables them**

Add below the imports in `tests/test_training_smoke.py`:

```python
import pytest

# the torch training loops call the ported (jnp) conjugate gradient; re-enabled when Plan 2 ports the run loops
pytestmark = pytest.mark.skip(reason="porting to JAX: re-enabled in plan 2")
```

Extend `JAXTYPED_MODULES` in `tests/conftest.py` with `"abstract_rl.algorithms.mixed.conjugate_gradient"` and
`"abstract_rl.misc.flat_params"`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_conjugate_gradient.py tests/test_parity_core.py -v`
Expected: 5 passed.

Run: `uv run pytest`
Expected: everything else passes or is skipped, with no errors.

- [ ] **Step 7: Commit**

```bash
git add src/abstract_rl/algorithms/mixed/conjugate_gradient.py src/abstract_rl/misc/flat_params.py \
    tests/test_conjugate_gradient.py tests/test_parity_core.py tests/test_training_smoke.py tests/conftest.py
git commit -m "Port the conjugate gradient and the flat parameter helpers to JAX.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Temperature dual

**Files:**
- Modify: `src/abstract_rl/algorithms/mixed/non_parametric_trust_region_optimizatioon.py` (whole file)
- Modify: `tests/test_operators.py` (whole file)
- Modify: `tests/test_parity_core.py` (append)
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `temperature_dual(q_tensor: Float[Array, "N K"], eta: Float[Array, "1"], eps: float, mask: Bool[Array, "N"] | None = None) -> Float[Array, "1"]`.
  - `find_optimal_temperature(q_tensor: Float[Array, "N K"], eps: float, eta0: float | Float[Array, ""] = 1.0, lr: float = 0.05, max_steps: int = 200, mask: Bool[Array, "N"] | None = None) -> Float[Array, "1"]`.
  - Both are jittable. `mask` marks the valid rows of a padded memory (Plan 3).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_operators.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature


def test_temperature_satisfies_kl_bound():
    q = jax.random.normal(jax.random.key(0), (256, 20)) * 10
    eps = 0.1
    eta = float(jax.jit(lambda qq: find_optimal_temperature(qq, eps))(q)[0])

    # the KL between the reweighted and the uniform (sampling) distribution matches the bound
    w = jax.nn.softmax(q / eta, axis=1)
    kl = float(jnp.mean(jnp.sum(w * jnp.log(w * q.shape[1] + 1e-12), axis=1)))
    assert eta > 0
    assert np.isclose(kl, eps, rtol=0.1, atol=0.01)


def test_masked_rows_are_ignored():
    q = jax.random.normal(jax.random.key(1), (64, 20)) * 5
    padded = jnp.concatenate([q, jnp.full((16, 20), 1e6)])
    mask = jnp.arange(80) < 64
    np.testing.assert_allclose(
        find_optimal_temperature(padded, 0.1, mask=mask), find_optimal_temperature(q, 0.1), rtol=1e-5
    )
```

Append to `tests/test_parity_core.py`:

```python
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature


def test_temperature_dual_matches_torch():
    for case in ("cold", "warm"):
        g = golden(f"dual_{case}")
        eta = find_optimal_temperature(jnp.asarray(g["q"]), 0.1, float(g["eta0"]), 0.05)
        # D4: float32 here, float64 in torch
        np.testing.assert_allclose(eta, g["eta"], rtol=1e-3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_operators.py tests/test_parity_core.py -v`
Expected: FAIL. The torch implementation calls `.double()` on a jax array (`AttributeError`).

- [ ] **Step 3: Convert the module**

```python
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float

from abstract_rl.misc.typing import DTYPE


def _mean(x: Float[Array, "N"], mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    return jnp.mean(x) if mask is None else jnp.sum(jnp.where(mask, x, 0.0)) / jnp.sum(mask)


def temperature_dual(
    q_tensor: Float[Array, "N K"], eta: Float[Array, "1"], eps: float, mask: Bool[Array, "N"] | None = None
) -> Float[Array, "1"]:
    """
    Dual function of the non parametric E-step for a fixed temperature, see https://arxiv.org/abs/1806.06920.

        g(eta) = eta * eps + eta * mean_s log mean_a exp(Q(s, a) / eta)

    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eta: The temperature.
    :param eps: The KL bound.
    :param mask: The valid rows of a padded memory, all rows if None.
    :return: The value of the dual function.
    """
    max_q = jnp.max(q_tensor, axis=1, keepdims=True)
    log_mean_exp = jax.scipy.special.logsumexp((q_tensor - max_q) / eta, axis=1) - np.log(q_tensor.shape[1])
    return eta * eps + _mean(max_q[:, 0], mask) + eta * _mean(log_mean_exp, mask)


def find_optimal_temperature(
    q_tensor: Float[Array, "N K"],
    eps: float,
    eta0: float | Float[Array, ""] = 1.0,
    lr: float = 0.05,
    max_steps: int = 200,
    mask: Bool[Array, "N"] | None = None,
) -> Float[Array, "1"]:
    """
    Uses a batch of data to find the optimal temperature eta for reweighting by minimizing the convex dual with Adam.
    The statistics of the problem, eta* and the dual value are printed on the host from the returned metrics
    (NonParametricVariationalPolicyBuilder.calc_eta), not inside the traced solve.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: Adam learning rate on the log temperature. Defaults to 0.05.
    :param max_steps: Maximum number of Adam steps.
    :param mask: The valid rows of a padded memory, all rows if None.
    :return: The found optimal temperature as a tensor of shape [1].
    """
    # D4: the dual runs in float32 (the torch version used float64)
    q_tensor = jnp.where(mask[:, None], q_tensor, 0.0) if mask is not None else q_tensor
    q_tensor = q_tensor.astype(DTYPE)
    eta0_f = jnp.clip(jnp.asarray(eta0, DTYPE), 1e-6, None)
    log_eta = jnp.log(eta0_f).reshape(1)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_eta)

    def step(_: int, carry: tuple[Float[Array, "1"], optax.OptState]) -> tuple[Float[Array, "1"], optax.OptState]:
        log_eta, opt_state = carry
        grad = jax.grad(lambda le: temperature_dual(q_tensor, jnp.exp(le), eps, mask)[0])(log_eta)
        updates, opt_state = optimizer.update(grad, opt_state)
        return optax.apply_updates(log_eta, updates), opt_state

    log_eta, _ = jax.lax.fori_loop(0, max_steps, step, (log_eta, opt_state))
    return jnp.exp(log_eta).astype(DTYPE)
```

Padded rows are zeroed before the max, the subtraction and the logsumexp, so a garbage value in them can't overflow.
Their contribution is then dropped by the masked means.

- [ ] **Step 4: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon"`.

Run: `uv run pytest tests/test_operators.py tests/test_parity_core.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/abstract_rl/algorithms/mixed/non_parametric_trust_region_optimizatioon.py tests/test_operators.py \
    tests/test_parity_core.py tests/conftest.py
git commit -m "Port the temperature dual to JAX with masked rows.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `Batch`, masked reductions and `ModelState`

**Files:**
- Modify: `src/abstract_rl/data_structures/temporal_difference_data/batch.py` (whole file)
- Create: `src/abstract_rl/data_structures/abstract_conf/model_state.py`
- Modify: `src/abstract_rl/data_structures/abstract_conf/model_configuration.py` (add `state()` and `load_state()`)
- Create: `tests/test_model_state.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Produces:
  - `Batch(states, actions, log_likelihood, rewards, next_states, ds, annotations=None, mask=None)`, an `eqx.Module`.
    It has the properties `states`, `actions`, `log_likelihoods`, `rewards`, `next_states`, `dones`, `mask`, plus
    `__getitem__(key)` and `__len__`. `__setitem__` is removed, because pytrees are immutable and nothing in `src`
    writes to a batch.
  - `batch_mask(batch) -> mask | None`. It returns `getattr(batch, "mask", None)`, so stubs without a mask count as
    fully valid.
  - `masked_mean(x, mask) -> Float[Array, ""]`, `masked_std(x, mask) -> Float[Array, ""]` (unbiased), and
    `masked_reduce(x, mask, est: str) -> Float[Array, ""]` with `est` in {"sum", "mean"}. Any other `est` raises
    `NotImplementedError(est)`.
  - `ModelState`, an `eqx.Module` with the fields `data: dict[str, Any]`, `syncables: dict[str, tuple[Any, Any]]`,
    `sync_steps: tuple[tuple[str, int], ...]` (static) and `overall_step: Int[Array, ""]`. Its methods:
    - `get(key, target=False)` and `__getitem__(key)`;
    - `set(key, value, target=False) -> ModelState`;
    - `sync() -> ModelState`;
    - the classmethod `create(data, syncables, sync_steps=None) -> ModelState`.
  - `ModelConfiguration.state() -> ModelState` and `ModelConfiguration.load_state(state) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model_state.py`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, masked_mean, masked_reduce, masked_std


class _Lin(eqx.Module):
    w: jax.Array

    def sync(self, v):
        return self


def test_masked_reductions_match_unpadded():
    x = jnp.arange(5.0)
    padded = jnp.concatenate([x, jnp.array([100.0, -7.0])])
    mask = jnp.arange(7) < 5
    assert masked_mean(padded, mask) == pytest.approx(float(jnp.mean(x)))
    assert masked_std(padded, mask) == pytest.approx(float(jnp.std(x, ddof=1)), rel=1e-6)
    assert masked_reduce(padded, mask, "sum") == pytest.approx(10.0)
    assert masked_mean(x, None) == pytest.approx(2.0)
    with pytest.raises(NotImplementedError):
        masked_reduce(x, None, "max")


def test_masked_reductions_ignore_nan_padding():
    padded = jnp.array([1.0, 2.0, 3.0, jnp.nan, jnp.inf])
    mask = jnp.array([True, True, True, False, False])
    assert jnp.isfinite(masked_mean(padded, mask)) and jnp.isfinite(masked_std(padded, mask))
    grad = jax.grad(lambda p: masked_mean(p * 2.0, mask))(padded)
    assert jnp.all(jnp.isfinite(grad[:3])) and jnp.all(grad[3:] == 0)


def test_batch_keeps_the_torch_accessors():
    s = jnp.ones((4, 3))
    b = Batch(s, jnp.zeros((4, 1)), jnp.zeros((4, 1)), jnp.zeros((4, 1)), s, jnp.zeros((4, 1)), {"a": jnp.arange(4.0)})
    assert len(b) == 4 and b.states.shape == (4, 3) and b["a"].shape == (4,) and b.mask is None
    assert b.log_likelihoods.shape == (4, 1) and b.dones.shape == (4, 1) and b.next_states.shape == (4, 3)
    jax.jit(lambda bb: bb["a"].sum())(b)  # a batch is a pytree


def test_get_set_and_target_access():
    st = ModelState.create({"opt": jnp.zeros(2)}, {"policy": (_Lin(jnp.ones(2)), _Lin(jnp.zeros(2)))})
    assert float(st.get("policy").w[0]) == 1.0 and float(st.get("policy", True).w[0]) == 0.0
    assert st["opt"].shape == (2,)
    st2 = st.set("policy", _Lin(jnp.full(2, 5.0))).set("opt", jnp.ones(2))
    assert float(st2.get("policy").w[0]) == 5.0 and float(st2.get("policy", True).w[0]) == 0.0
    assert float(st2["opt"][0]) == 1.0 and float(st.get("policy").w[0]) == 1.0  # immutable


def test_sync_respects_sync_steps():
    online, target = _Lin(jnp.ones(1)), _Lin(jnp.zeros(1))
    st = ModelState.create({}, {"a": (online, target), "b": (online, target)}, {"a": 1, "b": 2})
    seen = []
    for k in range(4):
        st = st.sync()
        seen.append((float(st.get("a", True).w[0]), float(st.get("b", True).w[0])))
        st = st.set("a", _Lin(jnp.full(1, k + 2.0))).set("b", _Lin(jnp.full(1, k + 2.0)))
    # ModelConfiguration.sync: copy when overall_step % s == 0, then increment overall_step
    assert seen == [(1.0, 1.0), (2.0, 1.0), (3.0, 3.0), (4.0, 3.0)]
    assert int(st.overall_step) == 4


def test_sync_is_jittable_and_vmappable():
    st = ModelState.create({}, {"a": (_Lin(jnp.ones((3, 2))), _Lin(jnp.zeros((3, 2))))})
    st = eqx.tree_at(lambda s: s.overall_step, st, jnp.zeros(3, jnp.int32))
    out = jax.jit(jax.vmap(lambda s: s.sync()))(st)
    np.testing.assert_allclose(out.get("a", True).w, 1.0)


def test_model_configuration_exports_state(make_pendulum_mc):
    mc = make_pendulum_mc()
    # add_syncable needs a Syncable model (Task 7+), a main element exercises the export
    mc.add_main("opt", jnp.zeros(3))
    st = mc.state()
    assert "opt" in st.data and "conf" not in st.data and "log" not in st.data
    mc.load_state(st.set("opt", jnp.ones(3)))
    assert float(mc["opt"][0]) == 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_model_state.py -v`
Expected: FAIL with `ModuleNotFoundError: abstract_rl.data_structures.abstract_conf.model_state`.

- [ ] **Step 3: Convert `batch.py`**

```python
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def _broadcast_mask(x: Array, mask: Bool[Array, "N"]) -> Array:
    return jnp.broadcast_to(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), x.shape)


def masked_mean(x: Array, mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    """Mean over all entries whose leading index is valid. Padded entries never reach the value or the gradient."""
    if mask is None:
        return jnp.mean(x)
    m = _broadcast_mask(x, mask)
    return jnp.sum(jnp.where(m, x, 0.0)) / jnp.sum(m)


def masked_std(x: Array, mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    """Unbiased standard deviation (torch ``Tensor.std()``) over the valid entries."""
    if mask is None:
        return jnp.std(x, ddof=1)
    m = _broadcast_mask(x, mask)
    mean = jnp.sum(jnp.where(m, x, 0.0)) / jnp.sum(m)
    return jnp.sqrt(jnp.sum(jnp.where(m, (x - mean) ** 2, 0.0)) / (jnp.sum(m) - 1))


def batch_mask(batch: Any) -> Any:
    """The valid rows of a batch; objects without a mask (light stubs with only ``states``) count as fully valid."""
    return getattr(batch, "mask", None)


def masked_reduce(x: Array, mask: Bool[Array, "N"] | None, est: str) -> Float[Array, ""]:
    """``est`` = "sum" or "mean" over the valid entries."""
    if est == "sum":
        return jnp.sum(x) if mask is None else jnp.sum(jnp.where(_broadcast_mask(x, mask), x, 0.0))
    elif est == "mean":
        return masked_mean(x, mask)
    raise NotImplementedError(est)


class Batch(eqx.Module):
    """
    Represents a batch of state, action, reward data tuples. ``mask`` marks the valid rows of a padded batch, None if
    every row is valid.
    """

    _states: Any
    _actions: Any
    _log_likelihoods: Any
    _rewards: Any
    _next_states: Any
    _dones: Any
    _annotations: dict[str, Any]
    _mask: Any

    def __init__(
        self,
        states: Any,
        actions: Any,
        log_likelihood: Any,
        rewards: Any,
        next_states: Any,
        ds: Any,
        annotations: dict[str, Any] | None = None,
        mask: Any = None,
    ) -> None:
        """
        Creates a new batch built upon the passed values.
        :param states: The states this batch contains.
        :param actions: The actions this batch contains.
        :param log_likelihood: The log likelihoods this batch contains.
        :param rewards: The rewards this batch contains.
        :param next_states: The next states this batch contains.
        :param annotations: The annotations this batch contains.
        :param mask: The valid rows, None if all rows are valid.
        """
        self._annotations = dict(annotations) if annotations is not None else {}
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._next_states = next_states
        self._dones = ds
        self._log_likelihoods = log_likelihood
        self._mask = mask

    # --- some syntactic sugar

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key: str) -> Any:
        return self._annotations[key]

    @property
    def mask(self) -> Any:
        """
        Valid rows of the batch.
        :return: A boolean vector [N] or None if every row is valid.
        """
        return self._mask

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
    def next_states(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._next_states

    @property
    def dones(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._dones
```

- [ ] **Step 4: Write `model_state.py`**

```python
"""Traced counterpart of ModelConfiguration: the models, optimiser states and buffers of one seed."""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


class ModelState(eqx.Module):
    """
    A pytree with the accessors of ModelConfiguration for everything jitted code needs. ``data`` holds the main elements
    (optimiser states, buffers), ``syncables`` the (online, target) pair of every sync able. ``get`` and ``set`` take the
    same keys and the same ``target`` flag as ``ModelConfiguration.get``, so ``self.mc.get(...)`` becomes
    ``state.get(...)``.
    """

    data: dict[str, Any]
    syncables: dict[str, tuple[Any, Any]]
    sync_steps: tuple[tuple[str, int], ...] = eqx.field(static=True)
    # a scalar per seed, "*seeds" so a host copy batched over seeds is a valid state too
    overall_step: Int[Array, "*seeds"]

    @classmethod
    def create(
        cls, data: dict[str, Any], syncables: dict[str, tuple[Any, Any]], sync_steps: dict[str, int] | None = None
    ) -> "ModelState":
        steps = sync_steps if sync_steps is not None else {k: 1 for k in syncables}
        return cls(dict(data), dict(syncables), tuple((k, int(steps[k])) for k in syncables), jnp.asarray(0, jnp.int32))

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def get(self, key: str, target: bool = False) -> Any:
        """
        Access a specific key, the target instance of a sync able if ``target`` is True.
        """
        ti = 0 if not target else 1
        return self.data[key] if key in self.data else self.syncables[key][ti]

    def set(self, key: str, value: Any, target: bool = False) -> "ModelState":
        """A new state where ``key`` (its target if ``target`` is True) is replaced by ``value``."""
        if key in self.data:
            return dataclasses.replace(self, data={**self.data, key: value})
        online, t_value = self.syncables[key]
        pair = (online, value) if target else (value, t_value)
        return dataclasses.replace(self, syncables={**self.syncables, key: pair})

    def sync(self) -> "ModelState":
        """
        ModelConfiguration.sync: the target of every sync able is replaced by the online instance if
        ``overall_step % sync_steps == 0``, afterwards ``overall_step`` is incremented.
        """
        syncables = dict(self.syncables)
        for k, s in self.sync_steps:
            o, to = syncables[k]
            synced = o.sync(to)
            do_sync = self.overall_step % s == 0
            syncables[k] = (o, jax.tree.map(lambda a, b, do=do_sync: jnp.where(do, a, b), synced, to))
        return dataclasses.replace(self, syncables=syncables, overall_step=self.overall_step + 1)
```

- [ ] **Step 5: Add `state()` and `load_state()` to `ModelConfiguration`**

Add the import `from abstract_rl.data_structures.abstract_conf.model_state import ModelState`, plus `import equinox as eqx`,
`import jax` and `import jax.numpy as jnp`. Add these methods after `sync`:

```python
    def state(self) -> ModelState:
        """
        The traced part of this configuration: every sync able and every main element which is a pytree of arrays
        (optimiser states, buffers). Host objects like the conf, the logger or the operators stay here.
        """

        def is_traced(obj: Any) -> bool:
            leaves = jax.tree.leaves(obj)
            return len(leaves) > 0 and all(eqx.is_array(leaf) for leaf in leaves)

        data = {k: v for k, v in self.data.items() if is_traced(v)}
        syncables = {k: (o, to) for k, (_, o, to) in self.syncables.items()}
        steps = {k: s for k, (s, _, _) in self.syncables.items()}
        return eqx.tree_at(
            lambda st: st.overall_step,
            ModelState.create(data, syncables, steps),
            jnp.asarray(self.overall_step, jnp.int32),
        )

    def load_state(self, state: ModelState) -> None:
        """Write a (host copy of a) traced state back, e.g. before ``store``."""
        for k, v in state.data.items():
            self.data[k] = v
        for k, (o, to) in state.syncables.items():
            self.syncables[k] = (self.syncables[k][0], o, to)
        self.overall_step = int(jnp.max(state.overall_step))
```

- [ ] **Step 6: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.data_structures.temporal_difference_data.batch"` and
`"abstract_rl.data_structures.abstract_conf.model_state"`.

Run: `uv run pytest tests/test_model_state.py -v`
Expected: 8 passed.

Run: `uv run pytest`
Expected: everything else passes or is skipped.

- [ ] **Step 7: Commit**

```bash
git add src/abstract_rl/data_structures/temporal_difference_data/batch.py \
    src/abstract_rl/data_structures/abstract_conf/model_state.py \
    src/abstract_rl/data_structures/abstract_conf/model_configuration.py tests/test_model_state.py tests/conftest.py
git commit -m "Add the traced ModelState and port Batch to a masked pytree.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `MultilayerPerceptron` and the activations

**Files:**
- Modify: `src/abstract_rl/neural_modules/neural_network.py` (whole file)
- Modify: `src/abstract_rl/misc/mixed.py` (`select_activation` only, plus the module-level activation functions)
- Create: `tests/test_parity_networks.py`
- Create: `tests/test_neural_network.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: `DTYPE`, `Syncable` (unchanged plain mixin).
- Produces:
  - `MultilayerPerceptron(mc, hidden_neurons, act_fn="tanh", layer_norm=False, init="orthogonal", *, key)`.
  - Its fields: `structure: tuple[int, ...]`, `layer_norm`, `init`, `act_fn` (all static), and
    `layers: list[eqx.nn.Linear]`, where each weight is `[out, in]` and float32.
  - Its methods: `forward(x: Float[Array, "*D in"]) -> Float[Array, "*D out"]`, `num_params() -> int`,
    `sync(v) -> self` and
    `initialize_layers(layers, init, keys) -> list[eqx.nn.Linear]` (it returns the layers; torch mutated them).
  - `select_activation(name) -> Callable[[Array], Array]`, a module-level function, never a lambda.

- [ ] **Step 1: Write the failing tests**

`tests/test_neural_network.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


@pytest.mark.parametrize("init", ["xavier_gaussian", "xavier_uniform", "orthogonal", "gaussian", "uniform", "high", "none"])
def test_init_zeroes_biases_and_keeps_float32(make_pendulum_mc, init):
    net = MultilayerPerceptron(make_pendulum_mc(), np.array([3, 16, 2]), ["tanh", "none"], False, init, key=jax.random.key(0))
    assert all(float(jnp.abs(l.bias).sum()) == 0.0 for l in net.layers)
    assert all(l.weight.dtype == jnp.float32 and l.weight.shape == (o, i) for l, (i, o) in zip(net.layers, [(3, 16), (16, 2)]))


def test_xavier_gaussian_has_the_torch_variance(make_pendulum_mc):
    net = MultilayerPerceptron(make_pendulum_mc(), np.array([200, 300]), ["none"], False, "xavier_gaussian", key=jax.random.key(1))
    assert float(jnp.std(net.layers[0].weight)) == pytest.approx(np.sqrt(2 / 500), rel=0.02)


def test_unknown_init_raises(make_pendulum_mc):
    with pytest.raises(NotImplementedError):
        MultilayerPerceptron(make_pendulum_mc(), np.array([3, 2]), ["none"], False, "foo", key=jax.random.key(0))


def test_mlp_with_per_output_activations_jits(make_pendulum_mc):
    # Hydra hands nested lists, the static structure must stay hashable
    net = MultilayerPerceptron(make_pendulum_mc(), np.array([3, 8, 2]), ["tanh", ["none", "softplus"]], key=jax.random.key(0))
    out = jax.jit(lambda n, x: n.forward(x))(net, jnp.full((4, 3), -50.0))
    assert out.shape == (4, 2) and jnp.all(out[:, 1] >= 0)


def test_two_builds_have_equal_static_structure(make_pendulum_mc):
    mc = make_pendulum_mc()
    a = MultilayerPerceptron(mc, np.array([3, 8, 1]), ["tanh", "tanh"], key=jax.random.key(0))
    b = MultilayerPerceptron(mc, np.array([3, 8, 1]), ["tanh", "tanh"], key=jax.random.key(1))
    assert jax.tree.structure(a) == jax.tree.structure(b)
```

`tests/test_parity_networks.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import golden, load_named, sub

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron

MLP_CASES = {
    "tanh_tanh_none": ([3, 64, 64, 2], ["tanh", "tanh", "none"], False, "xavier_gaussian"),
    "ln_elu": ([3, 32, 32, 1], ["elu", "elu", "none"], True, "xavier_gaussian"),
    "split_head": ([3, 16, 2], ["tanh", ["none", "softplus"]], False, "xavier_gaussian"),
    "single_act": ([3, 8, 8, 1], ["relu"], False, "orthogonal"),
    "prelu_lrelu": ([3, 8, 8, 1], ["prelu", "lrelu", "sigmoid"], False, "gaussian"),
}


@pytest.mark.parametrize("name", MLP_CASES)
def test_mlp_forward_matches_torch(make_pendulum_mc, name):
    structure, act_fn, layer_norm, init = MLP_CASES[name]
    g = golden(f"mlp_{name}")
    net = MultilayerPerceptron(make_pendulum_mc(), np.array(structure), act_fn, layer_norm, init, key=jax.random.key(0))
    net = load_named(net, sub(g, "params"))
    np.testing.assert_allclose(net.forward(jnp.asarray(g["x"])), g["y"], rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_neural_network.py tests/test_parity_networks.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'key'`.

- [ ] **Step 3: Convert `select_activation` in `misc/mixed.py`**

Keep `default_settings` and its torch import unchanged; Plan 3 removes them. Replace `select_activation` and add the
activation functions above it:

```python
import jax
import jax.numpy as jnp


def _identity(x: jax.Array) -> jax.Array:
    return x


def _prelu(x: jax.Array) -> jax.Array:
    # torch nn.PReLU at its initial slope 0.25: the module was never registered on the network, so it was never trained
    return jnp.where(x >= 0, x, 0.25 * x)


def _lrelu(x: jax.Array) -> jax.Array:
    return jax.nn.leaky_relu(x, 0.01)


def select_activation(name: str | None) -> Any:
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: The activation function (a module level function, so two networks compare equal).
    """
    if name == "none" or name is None:
        return _identity

    elif name.lower() == "softplus":
        return jax.nn.softplus

    elif name.lower() == "elu":
        return jax.nn.elu

    elif name.lower() == "prelu":
        return _prelu

    elif name.lower() == "relu":
        return jax.nn.relu

    elif name.lower() == "lrelu":
        return _lrelu

    elif name.lower() == "tanh":
        return jnp.tanh

    elif name.lower() == "sigmoid":
        return jax.nn.sigmoid

    raise NotImplementedError(f"unknown activation {name}")
```

- [ ] **Step 4: Convert `neural_network.py`**

```python
import math as m
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from numpy import ndarray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.misc.mixed import select_activation
from abstract_rl.misc.typing import DTYPE


class MultilayerPerceptron(eqx.Module, Syncable):
    """
    Fully connected neural network
    """

    structure: tuple[int, ...] = eqx.field(static=True)
    layer_norm: bool = eqx.field(static=True)
    init: str = eqx.field(static=True)
    act_fn: tuple[Any, ...] = eqx.field(static=True)
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        mc: ModelConfiguration,
        hidden_neurons: ndarray,
        act_fn: str | Sequence[str | list[str]] = "tanh",
        layer_norm: bool = False,
        init: str = "orthogonal",
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Create a new neural network, whereas the structure is passed as a list of numbers, representing the size of
        each fully connected layer including the input as well as the output layer.

        :param hidden_neurons: List of integers containing the size of the hidden layers
        :param act_fn: A string encoding the activation function.
        :param layer_norm: Activate layer normalization in between.
        :param init: A string representing the initialization method.
        :param key: The random key for the initialization.

        """
        assert len(act_fn) == len(hidden_neurons) - 1 or len(act_fn) == 1

        # save settings
        self.structure = tuple(int(n) for n in hidden_neurons)
        self.layer_norm = layer_norm
        self.init = init

        # select the correct activation functions (tuples, the static structure has to be hashable).
        self.act_fn = tuple(
            tuple(select_activation(ae) for ae in a) if isinstance(a, list | tuple) else select_activation(a)
            for a in act_fn
        )

        # create the layers and initialize
        num_layers = len(self.structure) - 1
        keys = jax.random.split(key, 2 * num_layers)
        layers = [
            eqx.nn.Linear(self.structure[i], self.structure[i + 1], key=keys[i], dtype=DTYPE)
            for i in range(num_layers)
        ]
        self.layers = self.initialize_layers(layers, init, keys[num_layers:])

    @staticmethod
    def initialize_layers(layers: list[eqx.nn.Linear], init: str, keys: Any) -> list[eqx.nn.Linear]:
        """
        This method can be used to initialize a layer
        """

        layers = [eqx.tree_at(lambda la: la.bias, layer, jnp.zeros_like(layer.bias)) for layer in layers]
        normal, uniform = jax.random.normal, jax.random.uniform
        if init == "xavier_gaussian":
            w_init = jax.nn.initializers.variance_scaling(1.0, "fan_avg", "normal", in_axis=-1, out_axis=-2)

        elif init == "xavier_uniform":
            w_init = jax.nn.initializers.variance_scaling(1.0, "fan_avg", "uniform", in_axis=-1, out_axis=-2)

        elif init == "orthogonal":
            w_init = jax.nn.initializers.orthogonal(m.sqrt(2))

        elif init == "gaussian":
            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return 0.1 * normal(k, shape, dtype)

        elif init == "uniform":
            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return uniform(k, shape, dtype, -0.1, 0.1)

        elif init == "high":
            def w_init(k: Any, shape: Any, dtype: Any) -> Any:
                return uniform(k, shape, dtype, 0.05, 0.1)

        elif init == "none":
            return layers
        else:
            raise NotImplementedError("initialization method not defined.")

        return [
            eqx.tree_at(lambda la: la.weight, layer, w_init(k, layer.weight.shape, DTYPE))
            for layer, k in zip(layers, keys)
        ]

    def forward(self, x: Float[Array, "*D n_in"]) -> Float[Array, "*D n_out"]:
        """
        Apply represented function to x.
        :param x: The input to the network.
        :return: The outputted values of the network.
        """

        # apply layer norm if wanted (torch nn.LayerNorm created per call: unit weight, zero bias, eps 1e-5)
        out = x
        out = out @ self.layers[0].weight.T + self.layers[0].bias
        if self.layer_norm:
            mean = jnp.mean(out, axis=-1, keepdims=True)
            var = jnp.var(out, axis=-1, keepdims=True)
            out = (out - mean) / jnp.sqrt(var + 1e-5)

        out = self.act_fn[0](out)

        # iteratively complete the graph in the middle
        num_layers = len(self.layers)
        for layer_idx in range(1, num_layers):
            # assemble linear part, layer norm and activation for each layer
            linear_f = out @ self.layers[layer_idx].weight.T + self.layers[layer_idx].bias
            act = self.act_fn[layer_idx] if len(self.act_fn) > 1 else self.act_fn[0]
            if isinstance(act, tuple):
                splitted_ts = [linear_f[..., i : i + 1] for i in range(linear_f.shape[-1])]
                for i in range(len(splitted_ts)):
                    splitted_ts[i] = act[i](splitted_ts[i])
                out = jnp.concatenate(splitted_ts, axis=-1)

            else:
                out = act(linear_f)

        return out

    def num_params(self) -> int:
        """
        Count parameters of module.
        """

        return sum(p.size for p in jax.tree.leaves(eqx.filter(self, eqx.is_inexact_array)))

    def sync(self, v: "MultilayerPerceptron") -> "MultilayerPerceptron":
        """
        Copies the content of the current instance to the supplied parameter instance. Modules are immutable, so the
        synced instance is the current one.
        :param v: The other instance, where the current values should be copied to.
        """
        return self
```

`eqx.nn.Linear` draws its default weights from `U(±1/sqrt(in))`, the same distribution as torch's default. So
`init="none"` keeps torch's default weight distribution, with zeroed biases as before.

- [ ] **Step 5: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.neural_modules.neural_network"`.

Run: `uv run pytest tests/test_neural_network.py tests/test_parity_networks.py -v`
Expected: 16 passed.

- [ ] **Step 6: Commit**

```bash
git add src/abstract_rl/neural_modules/neural_network.py src/abstract_rl/misc/mixed.py tests/test_neural_network.py \
    tests/test_parity_networks.py tests/conftest.py
git commit -m "Port the multilayer perceptron and its activations to Equinox.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Policy bases and `UnivariateGaussianPolicy`

**Files:**
- Modify: `src/abstract_rl/policy/policy.py`, `src/abstract_rl/policy/continuous/continuous_policy.py` (whole files)
- Modify: `src/abstract_rl/policy/continuous/univariate_gaussian_policy.py` (whole file)
- Modify: `tests/test_policies.py` (whole file)
- Modify: `tests/test_parity_networks.py` (append)
- Modify: `tests/test_env_wrapper.py`, `tests/test_obs_norm.py` (add a module skip)
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: `ModelState` and `Batch` (with `mask`), `masked_mean`, `masked_std`, `masked_reduce`,
  `MultilayerPerceptron(..., key=)`, and `DTYPE`.
- Produces, in `univariate_gaussian_policy.py`:
  - the constants `SQUASH_U_MAX`, `SQUASH_BOUND`, `SIGMA_MIN`, unchanged;
  - `normal_log_prob(value, loc, scale)`, `normal_kl(p_loc, p_scale, q_loc, q_scale)`, `normal_entropy(scale)`. These
    are the exact torch `Normal` formulas, and the diagonal policy reuses them.
- Produces, `UnivariateGaussianPolicy(mc, key)` with these methods:
  - `forward(states) -> [D, 2]`;
  - `log_prob(actions [D, K] | [D, 1], states=None, suff_stats=None) -> [D, K]`;
  - `mode(states=None, suff_stats=None) -> [D, 1]`;
  - `sample_actions(key, states=None, suff_stats=None, num_actions=1) -> [D, K]`;
  - `entropy(state, batch)` and `kl_divergence(state, batch, suff_stats=None, mode="i-projection", est="sum")`;
  - `get_suff_stats_dim() -> 2`, `clone() -> self`, `sync(v) -> self`;
  - the property `cov_variable`.
- Produces, `ContinuousPolicy`:
  - `likelihood_ratio(state, batch) -> Float[Array, ""]`, which uses masked mean and std;
  - the generic `kl_divergence(state, batch, suff_stats=None, mode="i-projection", est="sum")`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_policies.py`:

```python
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

KEY = jax.random.key(0)


def _policy(make_pendulum_mc, cls, **overrides):
    mc = make_pendulum_mc(**overrides)
    with mc["conf"].ns("policy"):
        return cls(mc, KEY)


def _force_sigma_preactivation(policy, value):
    """Make the last layer output ``[0, value]`` independent of the state."""
    last = policy.network.layers[-1]
    return eqx.tree_at(
        lambda p: (p.network.layers[-1].weight, p.network.layers[-1].bias),
        policy,
        (jnp.zeros_like(last.weight), jnp.array([0.0, value], jnp.float32)),
    )


def test_state_dependent_sigma_is_positive(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    states = jax.random.normal(jax.random.key(1), (512, 3)) * 10
    sigma = policy.forward(states)[:, 1]
    assert jnp.all(sigma > 0)
    assert jnp.all(jnp.isfinite(policy.log_prob(policy.sample_actions(KEY, states=states), states=states)))


def test_state_dependent_sigma_can_be_small_with_default_act_fn(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy, policy_cov_type="state")
    policy = _force_sigma_preactivation(policy, -20.0)
    sigma = policy.forward(jnp.zeros((4, 3)))[:, 1]
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)


def test_state_dependent_sigma_is_not_softplused_twice(make_pendulum_mc):
    # the original configurations put a softplus on the sigma head through the activation function
    policy = _policy(
        make_pendulum_mc,
        UnivariateGaussianPolicy,
        policy_cov_type="state",
        policy_act_fn=["tanh", "tanh", ["none", "softplus"]],
    )
    states = jnp.zeros((4, 3))
    sigma = _force_sigma_preactivation(policy, -20.0).forward(states)[:, 1]
    # a second softplus would floor sigma at ln 2 = 0.693
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    # a strongly negative input underflows the softplus to zero, sigma must stay strictly positive
    p = _force_sigma_preactivation(policy, -1000.0)
    assert jnp.all(p.forward(states)[:, 1] > 0)
    assert jnp.all(jnp.isfinite(p.log_prob(p.sample_actions(KEY, states=states), states=states)))
    # positive pre-activations are passed through a single softplus: softplus(3) = log(1 + e^3)
    p = _force_sigma_preactivation(policy, 3.0)
    np.testing.assert_allclose(p.forward(states)[:, 1], np.full(4, math.log1p(math.exp(3.0))), atol=1e-5)


def test_entropy_refuses_squashed_policy(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    assert policy.squash
    with pytest.raises(NotImplementedError):
        policy.entropy(None, None)


def test_float64_states_give_float32_stats(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    stats = policy.forward(np.random.default_rng(0).normal(size=(5, 3)))  # numpy float64, e.g. Pendulum observations
    assert stats.dtype == jnp.float32


def test_two_policies_have_equal_static_structure(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("policy"):
        a, b = UnivariateGaussianPolicy(mc, KEY), UnivariateGaussianPolicy(mc, jax.random.key(9))
    assert jax.tree.structure(a) == jax.tree.structure(b)


def test_policy_methods_jit(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, UnivariateGaussianPolicy)
    s = jnp.zeros((6, 3))
    out = jax.jit(lambda p, k: p.log_prob(p.sample_actions(k, states=s, num_actions=4), states=s))(policy, KEY)
    assert out.shape == (6, 4)
```

Append to `tests/test_parity_networks.py`:

```python
import equinox as eqx

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.policy.continuous.select_policy import select_policy
from golden import named_parameters

POLICY_CASES = {
    "uni_single_squash": ("Pendulum-v1", {"policy_cov_type": "single", "policy_squash": True}),
    "uni_state_nosquash": ("Pendulum-v1", {"policy_cov_type": "state", "policy_squash": False}),
    "uni_fixed_squash": ("Pendulum-v1", {"policy_cov_type": "fixed", "policy_squash": True, "policy_sigma": 0.5}),
    "diag_single_squash": ("TestPointMass-v0", {"policy_cov_type": "single", "policy_squash": True}),
    "diag_state_squash": ("TestPointMass-v0", {"policy_cov_type": "state", "policy_squash": True}),
    "diag_full_squash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": True}),
    "diag_full_nosquash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": False}),
}


def _golden_policy(make_pendulum_mc, name):
    env, overrides = POLICY_CASES[name]
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        fresh = select_policy(mc, jax.random.key(0))
    g = golden(name)
    policy, target = load_named(fresh, sub(g, "params")), load_named(fresh, sub(g, "target"))
    return policy, ModelState.create({}, {"policy": (policy, target)}), g


def _batch(g):
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    return Batch(jnp.asarray(g["states"]), jnp.asarray(g["single"]), jnp.asarray(g["lls"]), z, jnp.asarray(g["states"]), z, {"a": jnp.asarray(g["a"])})


def _close_grads(grads, expected):
    got = named_parameters(grads)
    assert set(got) == set(expected)
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], rtol=1e-4, atol=1e-5, err_msg=k)


@pytest.mark.parametrize("name", [n for n in POLICY_CASES if n.startswith("uni")])
def test_univariate_policy_matches_torch(make_pendulum_mc, name):
    policy, state, g = _golden_policy(make_pendulum_mc, name)
    states, stats = jnp.asarray(g["states"]), policy.forward(jnp.asarray(g["states"]))
    np.testing.assert_allclose(stats, g["stats"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["actions"]), suff_stats=stats), g["lp"], atol=1e-4)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["single"]), states=states), g["lp1"], atol=1e-4)
    np.testing.assert_allclose(policy.mode(states=states), g["mode"], atol=1e-6)
    batch = _batch(g)
    for mode in ("i", "m"):
        for est in ("mean", "sum"):
            kl = policy.kl_divergence(state, batch, mode=f"{mode}-projection", est=est)
            np.testing.assert_allclose(kl, g[f"kl_{mode}_{est}"], rtol=1e-4, atol=1e-6)
    if "entropy" in g:
        np.testing.assert_allclose(policy.entropy(state, batch), g["entropy"], rtol=1e-5)
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    _close_grads(grads, sub(g, "grads"))
    lr = policy.likelihood_ratio(state, batch)
    np.testing.assert_allclose(lr, g["lr"], rtol=1e-4, atol=1e-6)
    lr_grads = eqx.filter_grad(lambda p: p.likelihood_ratio(state.set("policy", p), batch))(policy)
    _close_grads(lr_grads, sub(golden(f"{name}_lr"), "grads"))
```

Skip the two env-wrapper tests until Plan 2 ports the rollout. Add this to both `tests/test_env_wrapper.py` and
`tests/test_obs_norm.py` below their imports:

```python
import pytest

# the torch env wrapper drives the ported (JAX) policy; re-enabled when Plan 2 ports the rollout
pytestmark = pytest.mark.skip(reason="porting to JAX: re-enabled in plan 2")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_policies.py tests/test_parity_networks.py -k "univariate or policies" -v`
Expected: FAIL. `UnivariateGaussianPolicy(mc, KEY)` raises `TypeError` (an unexpected positional argument).

- [ ] **Step 3: Convert `policy/policy.py`**

```python
from typing import Any

import equinox as eqx


class Policy(eqx.Module):
    """
    Represents a policy, e.g a gaussian policy or a eps greedy policy
    """

    def get_suff_stats_dim(self) -> int:
        """
        Obtain the dimension of sufficient statistics vector for the policy.
        :return: give dimension of suff stats vector.
        """
        raise NotImplementedError

    def forward(self, states: Any) -> Any:
        """
        Simply produces sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, get_suff_stats_dim()]
        """
        raise NotImplementedError

    def sample_actions(self, key: Any, states: Any = None, suff_stats: Any = None, num_actions: int = 1) -> Any:
        """
        Can be used to obtain action samples for all states.
        :param key: The random key.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :param num_actions: The number of actions or K.
        :return: A matrix [D, K] containing action samples generated by the suff stats of the specific state.
        """
        raise NotImplementedError
```

The torch `Policy.__init__(mc)` only stored `self.mc`. Models no longer hold the host configuration; the target policy
comes from `state`.

- [ ] **Step 4: Convert `policy/continuous/continuous_policy.py`**

```python
from typing import Any

import jax
import jax.numpy as jnp

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import (
    Batch,
    batch_mask,
    masked_mean,
    masked_reduce,
    masked_std,
)
from abstract_rl.misc.typing import DTYPE
from abstract_rl.policy.policy import Policy


class ContinuousPolicy(Policy, Syncable):
    """
    Represents a continuous policy, e.g a gaussian policy
    """

    def log_prob(self, actions: Any, states: Any = None, suff_stats: Any = None) -> Any:
        """
        Can be used to obtain log probability of actions relative to states.
        :param actions: A matrix of size [D, K] containing actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :return: A matrix [D, K] containing the log prob for each action in each row of
            actions matched to the row and the the suff stats of the specific state.
        """
        raise NotImplementedError

    def clone(self) -> "ContinuousPolicy":
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """

        raise NotImplementedError

    def sync(self, v: Any) -> Any:
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        raise NotImplementedError

    def mode(self, states: Any = None, suff_stats: Any = None) -> Any:
        """
        The most likely action of each state.
        :return: A matrix [D, A].
        """
        raise NotImplementedError

    def kl_divergence(
        self,
        state: ModelState,
        batch: Batch,
        suff_stats: Any = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> Any:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param state: The traced model state (gives the target policy).
        :param batch:
        :return:
        """

        tp = state.get("policy", True)

        # calc sufficient statistics
        states = jnp.asarray(batch.states, DTYPE)
        if suff_stats is None:
            suff_stats = self.forward(states)

        actions = jnp.asarray(batch.actions, DTYPE)
        log_likelihoods = jax.lax.stop_gradient(tp.log_prob(actions, states))
        new_lls = self.log_prob(actions, suff_stats=suff_stats)

        # use I projection
        if mode == "m-projection":
            p = log_likelihoods
            q = new_lls

        # use M Projection
        elif mode == "i-projection":
            p = new_lls
            q = log_likelihoods

        else:
            raise NotImplementedError

        # build kl terms adn sum up
        kl = jnp.exp(p) * (p - q)
        return masked_reduce(kl, batch_mask(batch), est)

    def likelihood_ratio(self, state: ModelState, batch: Batch) -> Any:
        """
        Calculate the objective of the underlying objective using the passed batch.
        :param state: The traced model state (gives the current policy).
        :param batch: Use this batch to estimate objective.
        :return: A un detached version of the objective
        """

        policy = state.get("policy", False)

        # calculate the objective
        states = jnp.asarray(batch.states, DTYPE)
        actions = jnp.asarray(batch.actions, DTYPE)
        suff_stats = policy.forward(states)
        ll = policy.log_prob(actions, suff_stats=suff_stats).reshape(-1)
        old_ll = jnp.asarray(batch.log_likelihoods, DTYPE).reshape(-1)

        # calculate the ratio term
        a_vals = jnp.asarray(batch["a"], DTYPE)
        mean_a = masked_mean(a_vals, batch_mask(batch))
        std_a = masked_std(a_vals, batch_mask(batch))
        advantages = (a_vals - mean_a) / (std_a + 1e-8)
        r = jnp.exp(ll - old_ll)
        obj = masked_mean(r * advantages, batch_mask(batch))
        return -obj
```

- [ ] **Step 5: Convert `policy/continuous/univariate_gaussian_policy.py`**

```python
import math
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean, masked_reduce
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy

# pre-squash samples are limited to |u| <= 5, so tanh stays invertible in float32
SQUASH_U_MAX = 5.0
SQUASH_BOUND = math.tanh(SQUASH_U_MAX)
SIGMA_MIN = 1e-6


def normal_log_prob(value: Array, loc: Array, scale: Array) -> Array:
    """``torch.distributions.Normal(loc, scale).log_prob(value)``."""
    return -((value - loc) ** 2) / (2 * scale**2) - jnp.log(scale) - math.log(math.sqrt(2 * math.pi))


def normal_kl(p_loc: Array, p_scale: Array, q_loc: Array, q_scale: Array) -> Array:
    """``torch.distributions.kl.kl_divergence(Normal(p_loc, p_scale), Normal(q_loc, q_scale))``."""
    var_ratio = (p_scale / q_scale) ** 2
    t1 = ((p_loc - q_loc) / q_scale) ** 2
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))


def normal_entropy(scale: Array) -> Array:
    """``torch.distributions.Normal(loc, scale).entropy()``."""
    return 0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(scale)


def _sigma_head_is_softplus(act_fn: list[str | list[str]]) -> bool:
    """Whether the configured activation of the last network layer already applies a softplus to the sigma output.

    ``act_fn`` is the per layer list of the network configuration, an entry of the last layer is either one activation
    name for all outputs or a list with one name per output, e.g. ``["tanh", "tanh", ["none", "softplus"]]`` (outputs
    are ``[mu, sigma]``).
    """
    last = act_fn[-1] if len(act_fn) > 1 else act_fn[0]
    if isinstance(last, (list, tuple)):
        last = last[1] if len(last) > 1 else last[0]
    return isinstance(last, str) and last.lower() == "softplus"


class UnivariateGaussianPolicy(ContinuousPolicy):
    """
    Represents a univariate gaussian policy. This only supports one action dimension.
    """

    cov_type: str = eqx.field(static=True)
    num_output_net: int = eqx.field(static=True)
    squash: bool = eqx.field(static=True)
    action_scale: float = eqx.field(static=True)
    _sigma_is_softplus: bool = eqx.field(static=True)
    network: MultilayerPerceptron
    # log sigma of cov_type fixed: a constant, so it is static (torch buffer) and never a parameter
    _fixed_cov: float | None = eqx.field(static=True, default=None)
    # log sigma of cov_type single: the only parameter besides the network
    cov: Float[Array, "1 1"] | None = None

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray) -> None:
        """
        Initializes a new uni variate gaussian policy.
        :param mc: The model configuration to use.
        :param key: The random key of the network initialization.
        """
        assert isinstance(mc, ModelConfiguration)

        run_conf = mc.get("conf")
        self.cov_type = run_conf["cov_type"]

        if self.cov_type == "state":
            self.num_output_net = 2
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            init_log_sigma = np.log(run_conf["sigma"])
            self._fixed_cov = float(init_log_sigma)
            self.num_output_net = 1
        elif self.cov_type == "single":
            # single state independent standard deviation, stored in log space
            init_log_sigma = np.log(run_conf.get("sigma", 1.0))
            self.cov = jnp.full([1, 1], init_log_sigma, DTYPE)
            self.num_output_net = 1
        else:
            raise NotImplementedError(f"unknown covariance type {self.cov_type}")

        # optionally squash the gaussian with tanh into the (symmetric) action bounds of the environment, so the
        # policy and the q function only ever see actions which the environment actually executes
        self.squash = bool(run_conf.get("squash", False))
        self.action_scale = float(np.max(np.abs(mc.env.action_space.high)))

        # create policy network
        sd = mc.env.observation_dim
        policy_net_structure: np.ndarray = np.concatenate(
            cast(Any, ([sd], run_conf["structure"], [self.num_output_net]))
        )

        af = run_conf["act_fn"]
        self._sigma_is_softplus = _sigma_head_is_softplus(af)
        network = MultilayerPerceptron(
            mc,
            hidden_neurons=policy_net_structure,
            act_fn=af,
            layer_norm=run_conf["layer_norm"],
            init=run_conf["init"],
            key=key,
        )
        if self.cov_type == "state":
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            bias = network.layers[-1].bias
            network = eqx.tree_at(lambda n: n.layers[-1].bias, network, bias.at[1].set(raw))
        self.network = network

    @property
    def cov_variable(self) -> Float[Array, "1 1"]:
        """The log standard deviation of the single and fixed covariance types."""
        return self.cov if self.cov_type == "single" else jnp.full([1, 1], self._fixed_cov, DTYPE)

    def get_suff_stats_dim(self) -> int:
        """
        Obtain the dimension of sufficient statistics vector for a univariate gaussian.
        :return: depending on the configuration 1 for mean-only and 2 for mean and covariance.
        """

        return 2

    def forward(self, states: Float[Array, "D Ds"]) -> Float[Array, "D 2"]:
        """
        Simply produces sufficient statistics of the univariate gaussian policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2]
        """
        states = jnp.asarray(states, DTYPE)

        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, 0:1], out[:, 1:2]
            # if the activation function of the network already is a softplus on the sigma head, only guard against an
            # underflow to 0, otherwise the unconstrained output is made strictly positive with a softplus
            sigma = jnp.maximum(sigma, SIGMA_MIN) if self._sigma_is_softplus else jax.nn.softplus(sigma) + SIGMA_MIN
            return jnp.concatenate([mu, sigma], 1)

        else:
            mu = self.network.forward(states)
            fw = jnp.exp(self.cov_variable)
            sigma = jnp.broadcast_to(fw, mu.shape)
            return jnp.concatenate([mu, sigma], 1)

    def _suff_stats(self, states: Any, suff_stats: Any) -> Float[Array, "D 2"]:
        # check if it full fills a xor relation
        assert states is not None or suff_stats is not None
        assert states is None or suff_stats is None

        # get suff statistics
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        return suff_stats

    def log_prob(self, actions: Float[Array, "D K"], states: Any = None, suff_stats: Any = None) -> Float[Array, "D K"]:
        """
        Can be used to obtain log probability of actions relative to states.
        :param actions: A matrix of size [D, K] containing actions for each state.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :return: A matrix [D, K] containing the log prob for each action in each row of
            actions matched to the row and the the suff stats of the specific state.
        """
        suff_stats = self._suff_stats(states, suff_stats)
        actions = jnp.asarray(actions, DTYPE)

        # extract mu and sigma depending on settings
        mu, sigma = suff_stats[:, 0:1], suff_stats[:, 1:2]
        if not self.squash:
            return normal_log_prob(actions, mu, sigma)

        # change of variables a = c * tanh(u): log p(a) = log N(u) - log(c * (1 - tanh(u)^2))
        y = jnp.clip(actions / self.action_scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        ll = normal_log_prob(u, mu, sigma) - jnp.log(self.action_scale * (1 - y**2))
        return ll

    def mode(self, states: Any = None, suff_stats: Any = None) -> Float[Array, "D 1"]:
        suff_stats = self._suff_stats(states, suff_stats)
        mu = suff_stats[:, 0:1]
        return self.action_scale * jnp.tanh(mu) if self.squash else mu

    def sample_actions(
        self, key: PRNGKeyArray, states: Any = None, suff_stats: Any = None, num_actions: int = 1
    ) -> Float[Array, "D K"]:
        """
        Can be used to obtain action samples for all states.
        :param key: The random key.
        :param states: A matrix of size [D, Ds]. If this None => suff_states not None
        :param suff_stats: A matrix of size [D, get_suff_stats_dim()] containing suff stats.
            If this None => states is not None
        :param num_actions: The number of actions or K.
        :return: A matrix [D, K] containing action samples generated by the suff stats of the specific state.
        """
        suff_stats = self._suff_stats(states, suff_stats)

        # extract mu and sigma depending on settings, reparametrised sample mu + sigma * eps (torch rsample)
        mu, sigma = suff_stats[:, 0:1], suff_stats[:, 1:2]
        eps = jax.random.normal(key, (mu.shape[0], num_actions), DTYPE)
        samples = mu + eps * sigma
        if self.squash:
            samples = self.action_scale * jnp.tanh(jnp.clip(samples, -SQUASH_U_MAX, SQUASH_U_MAX))
        return samples

    def clone(self) -> "UnivariateGaussianPolicy":
        """
        Clones the current instance. Modules are immutable, so the clone is the instance itself.
        :return: A new identical clones instance of the current.
        """
        return self

    def sync(self, v: "UnivariateGaussianPolicy") -> "UnivariateGaussianPolicy":
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        return self

    def entropy(self, state: ModelState, batch: Batch) -> Float[Array, ""]:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        policy = state.get("policy", False)
        t_states = jnp.asarray(batch.states, DTYPE)
        suff_stats = policy.forward(t_states)

        sigma = suff_stats[:, 1]
        ent = masked_mean(normal_entropy(sigma), batch_mask(batch))
        return ent

    def kl_divergence(
        self, state: ModelState, batch: Batch, suff_stats: Any = None, mode: str = "i-projection", est: str = "sum"
    ) -> Float[Array, ""]:
        """
        Calculates the kl divergence between the current policy and the saved likelihood values.
        :param state: The traced model state (gives the target policy).
        :param batch:
        :return:
        """

        # calc sufficient statistics
        states = jnp.asarray(batch.states, DTYPE)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)

        # create this distribution
        mu, sigma = suff_stats[:, 0], suff_stats[:, 1]

        # get the old sufficient statistics
        t_policy = state.get("policy", True)
        t_suff_stats = jax.lax.stop_gradient(t_policy.forward(states))
        t_mu, t_sigma = t_suff_stats[:, 0], t_suff_stats[:, 1]

        # use I projection
        if mode == "i-projection":
            kl = normal_kl(mu, sigma, t_mu, t_sigma)

        # use M Projection
        elif mode == "m-projection":
            kl = normal_kl(t_mu, t_sigma, mu, sigma)

        else:
            raise NotImplementedError

        return masked_reduce(kl, batch_mask(batch), est)
```

- [ ] **Step 5b: Give `select_policy` its key**

The univariate parity test builds policies through `select_policy`. Apply the full conversion from Task 9, Step 4 now.
`DiagonalGaussianPolicy(mc, key)` only works after Task 9, but no Task 8 test reaches that branch.

- [ ] **Step 6: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.policy.policy"`, `"abstract_rl.policy.continuous.continuous_policy"` and
`"abstract_rl.policy.continuous.univariate_gaussian_policy"`.

Run: `uv run pytest tests/test_policies.py tests/test_parity_networks.py -k "univariate or policies or mlp" -v`
Expected: all passed, including 3 univariate parity cases.

- [ ] **Step 7: Commit**

```bash
git add src/abstract_rl/policy/policy.py src/abstract_rl/policy/continuous/continuous_policy.py \
    src/abstract_rl/policy/continuous/univariate_gaussian_policy.py src/abstract_rl/policy/continuous/select_policy.py \
    tests/test_policies.py tests/test_parity_networks.py tests/test_env_wrapper.py tests/test_obs_norm.py tests/conftest.py
git commit -m "Port the policy bases and the univariate Gaussian policy to Equinox.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `DiagonalGaussianPolicy` (diagonal and full covariance) and `select_policy`

**Files:**
- Modify: `src/abstract_rl/policy/continuous/diagonal_gaussian_policy.py` (whole file)
- Modify: `src/abstract_rl/policy/continuous/select_policy.py` (whole file)
- Modify: `tests/test_diagonal_policy.py` (whole file)
- Modify: `tests/test_parity_networks.py` (append)
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: `normal_log_prob`, `normal_kl`, `normal_entropy`, `SIGMA_MIN`, `SQUASH_BOUND`, `SQUASH_U_MAX` from Task 8.
- Produces: `DiagonalGaussianPolicy(mc, key)` with the same method set as Task 8. The shapes:
  - samples are `[D, K, A]`;
  - `log_prob` takes `[D, K, A]` or `[D, A]` and returns `[D, K]`;
  - `mode` returns `[D, A]`;
  - suff stats are `[D, 2A]`, or `[D, A + A(A+1)/2]` for `full`.
- Produces: `select_policy(mc, key)`. Also `head_activations` and `sigma_is_softplus`, unchanged.
- Produces the private helpers `_mvn_log_prob(value, loc, scale_tril)` and `_mvn_kl(p_loc, p_tril, q_loc, q_tril)`.
  These are the exact `torch.distributions.MultivariateNormal` formulas.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_diagonal_policy.py`:

```python
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.policy.continuous.diagonal_gaussian_policy import (
    DiagonalGaussianPolicy,
    head_activations,
    sigma_is_softplus,
)
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy

LOW, HIGH = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
KEY = jax.random.key(0)


class _Batch:
    def __init__(self, states):
        self.states = states
        self.mask = None


def _policy(make_pendulum_mc, env="TestPointMass-v0", cls=DiagonalGaussianPolicy, **overrides):
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        return cls(mc, KEY)


def _force_head(policy, mu_pre, sigma_pre):
    """Make the last layer output ``[mu_pre, sigma_pre]`` independent of the state."""
    last = policy.network.layers[-1]
    return eqx.tree_at(
        lambda p: (p.network.layers[-1].weight, p.network.layers[-1].bias),
        policy,
        (jnp.zeros_like(last.weight), jnp.asarray([*mu_pre, *sigma_pre], jnp.float32)),
    )


def _perturb(module, key, scale):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    leaves = [leaf + scale * jax.random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _unpack_tril(packed, action_dim):
    rows, cols = np.tril_indices(action_dim)
    return jnp.zeros((packed.shape[0], action_dim, action_dim)).at[:, rows, cols].set(packed)


def _kl_normal(m0, s0, m1, s1):
    return 0.5 * (s0**2 / s1**2 + (m1 - m0) ** 2 / s1**2 - 1 + 2 * np.log(s1 / s0))


def _kl_mvn(m0, c0, m1, c1):
    k = m0.shape[-1]
    inv = np.linalg.inv(c1)
    d = m1 - m0
    return 0.5 * (np.trace(inv @ c0) + d @ inv @ d - k + np.log(np.linalg.det(c1) / np.linalg.det(c0)))


def test_select_policy_uses_the_diagonal_policy_for_multi_dimensional_actions(make_pendulum_mc):
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, "Pendulum-v1", select_policy), UnivariateGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_type="diagonal"), DiagonalGaussianPolicy)
    assert isinstance(_policy(make_pendulum_mc, cls=select_policy, policy_cov_type="full"), DiagonalGaussianPolicy)
    assert isinstance(
        _policy(make_pendulum_mc, "Pendulum-v1", select_policy, policy_cov_type="full"), DiagonalGaussianPolicy
    )


def test_shapes(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state")
    states = jax.random.normal(KEY, (7, 4))
    stats = policy.forward(states)
    assert stats.shape == (7, 4) == (7, policy.get_suff_stats_dim())
    actions = policy.sample_actions(KEY, suff_stats=stats, num_actions=5)
    assert actions.shape == (7, 5, 2)
    assert policy.log_prob(actions, suff_stats=stats).shape == (7, 5)
    assert policy.mode(suff_stats=stats).shape == (7, 2)
    # one action per state, e.g. the ones stored in a batch
    assert policy.log_prob(policy.mode(suff_stats=stats), suff_stats=stats).shape == (7, 1)


def test_head_activations_are_expanded_per_output():
    act, sigma_acts = head_activations(["tanh", "tanh", ["none", "softplus"]], 3)
    assert act[-1] == ["none"] * 3 + ["softplus"] * 3
    assert sigma_acts == ["softplus"] * 3
    assert sigma_is_softplus(sigma_acts)
    # one dimension keeps the original univariate configuration, a single name applies to all outputs
    assert head_activations(["tanh", "tanh", ["none", "softplus"]], 1)[0][-1] == ["none", "softplus"]
    assert not sigma_is_softplus(head_activations(["tanh", "tanh", "none"], 2)[1])
    with pytest.raises(ValueError):
        head_activations(["tanh", "tanh", ["none", "softplus", "none"]], 2)
    with pytest.raises(ValueError):
        sigma_is_softplus(["softplus", "none"])


def test_state_dependent_sigma_default_activation_applies_one_softplus(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", "none"])
    states = jnp.zeros((3, 4))
    # a strongly negative raw output may give a tiny, but strictly positive sigma (a second softplus would floor it)
    sigma = _force_head(policy, [0.0, 0.0], [-20.0, -20.0]).forward(states)[:, 2:]
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(_force_head(policy, [0.0, 0.0], [3.0, -3.0]).forward(states)[0, 2:], expected + 1e-6, atol=1e-5)


def test_state_dependent_sigma_with_softplus_activation_is_not_softplused_twice(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_cov_type="state", policy_act_fn=["tanh", "tanh", ["none", "softplus"]])
    states = jnp.zeros((3, 4))
    p = _force_head(policy, [0.0, 0.0], [-20.0, -1000.0])
    sigma = p.forward(states)[:, 2:]
    # the underflowing softplus is clamped, never zero and never floored at ln 2
    assert jnp.all(sigma > 0) and jnp.all(sigma < 1e-3)
    assert jnp.all(jnp.isfinite(p.log_prob(p.sample_actions(KEY, states=states), states=states)))
    expected = np.array([math.log1p(math.exp(3.0)), math.log1p(math.exp(-3.0))])
    np.testing.assert_allclose(_force_head(policy, [0.0, 0.0], [3.0, -3.0]).forward(states)[0, 2:], expected, atol=1e-5)


def test_mean_is_not_activated_twice(make_pendulum_mc):
    # the mean output is passed through unchanged by forward, the tanh squash is applied exactly once in mode / samples
    policy = _policy(
        make_pendulum_mc, policy_cov_type="single", policy_squash=True, policy_act_fn=["tanh", "tanh", "none"]
    )
    mu_pre = [0.5, -1.5]
    policy = _force_head(policy, mu_pre, [])
    states = jnp.zeros((2, 4))
    np.testing.assert_allclose(policy.forward(states)[0, :2], mu_pre, atol=1e-7)
    center, scale = (HIGH + LOW) / 2, (HIGH - LOW) / 2
    np.testing.assert_allclose(policy.mode(states=states)[0], center + scale * np.tanh(mu_pre), atol=1e-6)


def test_squashed_samples_lie_in_the_per_dimension_bounds(make_pendulum_mc):
    policy = _force_head(_policy(make_pendulum_mc, policy_squash=True, policy_sigma=5.0), [3.0, -3.0], [])
    a = policy.sample_actions(KEY, states=jnp.zeros((4, 4)), num_actions=2000)
    assert jnp.all(a[..., 0] <= 1) and jnp.all(a[..., 0] >= -1)
    assert jnp.all(a[..., 1] <= 2) and jnp.all(a[..., 1] >= -2)
    # extreme mean and large sigma saturate the bounds but keep the log probability finite
    assert jnp.isfinite(policy.log_prob(a, states=jnp.zeros((4, 4)))).all()
    assert a[..., 0].mean() > 0.3 and a[..., 1].mean() < -0.6


@pytest.mark.parametrize("squash", [False, True])
def test_log_prob_is_a_density(make_pendulum_mc, squash):
    policy = _policy(make_pendulum_mc, policy_squash=squash, policy_cov_type="single", policy_sigma=0.5)
    stats = jnp.array([[0.3, -0.2, 0.5, 0.5]])
    if squash:
        lo, hi, n = LOW + 1e-4, HIGH - 1e-4, 801
    else:
        lo, hi, n = np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 801
    g0, g1 = jnp.linspace(lo[0], hi[0], n), jnp.linspace(lo[1], hi[1], n)
    grid = jnp.stack(jnp.meshgrid(g0, g1, indexing="ij"), axis=-1).reshape(1, -1, 2)
    density = jnp.exp(policy.log_prob(grid, suff_stats=stats)).reshape(n, n)
    integral = float(jnp.trapezoid(jnp.trapezoid(density, g1, axis=1), g0))
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_squashed_log_prob_matches_the_change_of_variables(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=True)
    mu, sigma = np.array([[0.3, -0.7]]), np.array([[0.8, 1.1]])
    u = np.array([[[0.2, -0.4]]])
    scale = (HIGH - LOW) / 2
    a = (HIGH + LOW) / 2 + scale * np.tanh(u)
    expected = norm.logpdf(u[:, 0], mu, sigma).sum() - (np.log(scale) + np.log1p(-np.tanh(u) ** 2)).sum()
    got = policy.log_prob(jnp.asarray(a), suff_stats=jnp.asarray(np.concatenate([mu, sigma], 1))).sum()
    assert float(got) == pytest.approx(float(expected), abs=1e-4)


def test_kl_divergence_matches_the_per_dimension_sum_and_monte_carlo(make_pendulum_mc):
    target = _policy(make_pendulum_mc, policy_cov_type="state", policy_squash=True)
    # move the policy away from its (synced) target
    policy = _perturb(target, jax.random.key(1), 0.3)
    state = ModelState.create({}, {"policy": (policy, target)})
    states = jax.random.normal(jax.random.key(2), (3, 4))
    batch = _Batch(states)
    stats, t_stats = np.asarray(policy.forward(states)), np.asarray(target.forward(states))
    mu, sigma, t_mu, t_sigma = stats[:, :2], stats[:, 2:], t_stats[:, :2], t_stats[:, 2:]

    kl = policy.kl_divergence(state, batch, mode="i-projection", est="mean")
    expected = _kl_normal(mu, sigma, t_mu, t_sigma).sum(axis=1).mean()
    assert float(kl) == pytest.approx(expected, abs=1e-5)
    assert float(policy.kl_divergence(state, batch, est="sum")) == pytest.approx(expected * 3, abs=1e-4)
    reverse = policy.kl_divergence(state, batch, mode="m-projection", est="mean")
    assert float(reverse) == pytest.approx(_kl_normal(t_mu, t_sigma, mu, sigma).sum(axis=1).mean(), abs=1e-5)

    # Monte Carlo estimate of KL(current || target) from the squashed log probabilities (the squashing is a bijection)
    a = policy.sample_actions(jax.random.key(3), suff_stats=jnp.asarray(stats), num_actions=100_000)
    mc_kl = float(jnp.mean(policy.log_prob(a, suff_stats=jnp.asarray(stats)) - policy.log_prob(a, suff_stats=jnp.asarray(t_stats))))
    assert mc_kl == pytest.approx(float(expected), rel=0.1, abs=0.02)
    # the gradient reaches the policy parameters
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="i-projection", est="mean"))(policy)
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))


def test_one_dimensional_diagonal_policy_equals_the_univariate_policy(make_pendulum_mc):
    mc = make_pendulum_mc(policy_cov_type="state", policy_squash=True)
    with mc["conf"].ns("policy"):
        diag, uni = DiagonalGaussianPolicy(mc, KEY), UnivariateGaussianPolicy(mc, jax.random.key(1))
    uni = eqx.tree_at(lambda u: u.network, uni, diag.network)
    states = jax.random.normal(KEY, (6, 3))
    sd, su = diag.forward(states), uni.forward(states)
    np.testing.assert_allclose(sd, su, atol=1e-7)
    actions = uni.sample_actions(KEY, suff_stats=su, num_actions=4)
    np.testing.assert_allclose(diag.log_prob(actions[..., None], suff_stats=sd), uni.log_prob(actions, suff_stats=su), atol=1e-5)
    np.testing.assert_allclose(diag.mode(suff_stats=sd), uni.mode(suff_stats=su), atol=1e-7)


def test_full_covariance_log_prob_matches_multivariate_normal(make_pendulum_mc):
    # packed row-major triangle for two actions: L00, L10, L11
    policy = _force_head(_policy(make_pendulum_mc, policy_cov_type="full", policy_squash=False), [0.2, -0.4], [0.0, 0.5, -0.3])
    states = jnp.zeros((3, 4))
    stats = policy.forward(states)
    diagonal = jax.nn.softplus(jnp.array([0.0, -0.3])) + 1e-6
    np.testing.assert_allclose(stats[0, :2], [0.2, -0.4], atol=1e-7)
    np.testing.assert_allclose(stats[0, 2:], [diagonal[0], 0.5, diagonal[1]], atol=1e-6)
    actions = jnp.array([[[0.1, -0.2], [0.0, 0.3]], [[-0.4, 0.2], [0.2, -0.1]], [[0.3, 0.1], [-0.2, 0.0]]])
    factor = _unpack_tril(stats[:, 2:], 2)
    cov = factor @ jnp.swapaxes(factor, -1, -2)
    expected = jnp.stack(
        [jax.vmap(multivariate_normal.logpdf)(actions[:, k], stats[:, :2], cov) for k in range(actions.shape[1])], axis=1
    )
    np.testing.assert_allclose(policy.log_prob(actions, suff_stats=stats), expected, atol=1e-5)
    # one action per state, the shape stored on a batch
    np.testing.assert_allclose(policy.log_prob(actions[:, 0], suff_stats=stats)[:, 0], expected[:, 0], atol=1e-5)


def test_full_covariance_kl_of_a_policy_with_itself_is_zero(make_pendulum_mc):
    target = _policy(make_pendulum_mc, policy_cov_type="full", policy_squash=True)
    batch = _Batch(jnp.asarray(np.random.default_rng(0).normal(size=(6, 4)), jnp.float32))
    same = ModelState.create({}, {"policy": (target, target)})
    # TRPO uses the m-projection mean, MPO the same total KL
    assert float(target.kl_divergence(same, batch, mode="i-projection", est="mean")) == pytest.approx(0.0, abs=1e-6)
    assert float(target.kl_divergence(same, batch, mode="m-projection", est="sum")) == pytest.approx(0.0, abs=1e-5)

    policy = _perturb(target, jax.random.key(4), 0.2)
    state = ModelState.create({}, {"policy": (policy, target)})
    stats, t_stats = np.asarray(policy.forward(batch.states)), np.asarray(target.forward(batch.states))
    f, tf = np.asarray(_unpack_tril(jnp.asarray(stats[:, 2:]), 2)), np.asarray(_unpack_tril(jnp.asarray(t_stats[:, 2:]), 2))
    expected = np.mean([_kl_mvn(t_stats[i, :2], tf[i] @ tf[i].T, stats[i, :2], f[i] @ f[i].T) for i in range(6)])
    kl = policy.kl_divergence(state, batch, mode="m-projection", est="mean")
    assert float(kl) == pytest.approx(expected, abs=1e-4)
    assert float(kl) > 0
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))


def test_full_covariance_is_isotropic_at_init(make_pendulum_mc):
    sigma = 0.4
    policy = _policy(make_pendulum_mc, policy_cov_type="full", policy_sigma=sigma)
    states = jax.random.normal(KEY, (16, 4))
    factor = _unpack_tril(policy.forward(states)[:, policy.action_dim :], policy.action_dim)
    cov = factor @ jnp.swapaxes(factor, -1, -2)
    np.testing.assert_allclose(cov, np.broadcast_to((sigma**2) * np.eye(2), cov.shape), atol=1e-5)


def test_state_and_full_covariance_start_at_policy_sigma(make_pendulum_mc):
    sigma = 0.4
    state = _policy(make_pendulum_mc, policy_cov_type="state", policy_sigma=sigma)
    np.testing.assert_allclose(state.forward(jnp.zeros((1, 4)))[:, 2:], np.full((1, 2), sigma), atol=1e-5)
    full = _policy(make_pendulum_mc, policy_cov_type="full", policy_sigma=sigma)
    np.testing.assert_allclose(full.forward(jnp.zeros((1, 4)))[0, jnp.array([2, 4])], np.full(2, sigma), atol=1e-5)
    uni = _policy(
        make_pendulum_mc, "Pendulum-v1", UnivariateGaussianPolicy, policy_cov_type="state", policy_sigma=sigma
    )
    np.testing.assert_allclose(uni.forward(jnp.zeros((1, 3)))[:, 1], [sigma], atol=1e-5)


def test_entropy_of_the_unsquashed_policy_and_refusal_for_squashed(make_pendulum_mc):
    policy = _policy(make_pendulum_mc, policy_squash=False, policy_cov_type="single", policy_sigma=0.7)
    expected = 2 * 0.5 * math.log(2 * math.pi * math.e * 0.7**2)
    assert float(policy.entropy(None, _Batch(jnp.zeros((3, 4))))) == pytest.approx(expected, abs=1e-5)
    with pytest.raises(NotImplementedError):
        _policy(make_pendulum_mc, policy_squash=True).entropy(None, _Batch(jnp.zeros((3, 4))))
```

Append to `tests/test_parity_networks.py`:

```python
@pytest.mark.parametrize("name", [n for n in POLICY_CASES if n.startswith("diag")])
def test_diagonal_policy_matches_torch(make_pendulum_mc, name):
    policy, state, g = _golden_policy(make_pendulum_mc, name)
    states, stats = jnp.asarray(g["states"]), policy.forward(jnp.asarray(g["states"]))
    np.testing.assert_allclose(stats, g["stats"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["actions"]), suff_stats=stats), g["lp"], atol=1e-4)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["single"]), states=states), g["lp1"], atol=1e-4)
    np.testing.assert_allclose(policy.mode(states=states), g["mode"], atol=1e-6)
    batch = _batch(g)
    for mode in ("i", "m"):
        for est in ("mean", "sum"):
            kl = policy.kl_divergence(state, batch, mode=f"{mode}-projection", est=est)
            np.testing.assert_allclose(kl, g[f"kl_{mode}_{est}"], rtol=1e-4, atol=1e-6)
    if "entropy" in g:
        np.testing.assert_allclose(policy.entropy(state, batch), g["entropy"], rtol=1e-5)
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    _close_grads(grads, sub(g, "grads"))
    np.testing.assert_allclose(policy.likelihood_ratio(state, batch), g["lr"], rtol=1e-4, atol=1e-6)
    lr_grads = eqx.filter_grad(lambda p: p.likelihood_ratio(state.set("policy", p), batch))(policy)
    _close_grads(lr_grads, sub(golden(f"{name}_lr"), "grads"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_diagonal_policy.py tests/test_parity_networks.py -k diag -v`
Expected: FAIL. `DiagonalGaussianPolicy(mc, KEY)` raises a `TypeError`.

- [ ] **Step 3: Convert `diagonal_gaussian_policy.py`**

Keep `head_activations` and `sigma_is_softplus` byte-for-byte; they are pure Python. Replace the imports and the class:

```python
import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean, masked_reduce
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import (
    SIGMA_MIN,
    SQUASH_BOUND,
    SQUASH_U_MAX,
    normal_entropy,
    normal_kl,
    normal_log_prob,
)


def _mvn_log_prob(value: Array, loc: Array, scale_tril: Array) -> Array:
    """``torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(value)``."""
    diff = value - loc
    factor = jnp.broadcast_to(scale_tril, diff.shape + diff.shape[-1:])
    z = jax.scipy.linalg.solve_triangular(factor, diff[..., None], lower=True)[..., 0]
    half_log_det = jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    return -0.5 * (value.shape[-1] * math.log(2 * math.pi) + jnp.sum(z**2, axis=-1)) - half_log_det


def _mvn_kl(p_loc: Array, p_tril: Array, q_loc: Array, q_tril: Array) -> Array:
    """``torch.distributions.kl.kl_divergence(MultivariateNormal(p...), MultivariateNormal(q...))``."""
    n = p_loc.shape[-1]
    half_term1 = jnp.log(jnp.diagonal(q_tril, axis1=-2, axis2=-1)).sum(-1) - jnp.log(
        jnp.diagonal(p_tril, axis1=-2, axis2=-1)
    ).sum(-1)
    term2 = jnp.sum(jax.scipy.linalg.solve_triangular(q_tril, p_tril, lower=True) ** 2, axis=(-2, -1))
    term3 = jnp.sum(jax.scipy.linalg.solve_triangular(q_tril, (q_loc - p_loc)[..., None], lower=True)[..., 0] ** 2, -1)
    return half_term1 + 0.5 * (term2 + term3 - n)


# head_activations and sigma_is_softplus: unchanged


class DiagonalGaussianPolicy(ContinuousPolicy):
    """
    (class docstring unchanged)
    """

    cov_type: str = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    num_output_net: int = eqx.field(static=True)
    squash: bool = eqx.field(static=True)
    action_center: tuple[float, ...] = eqx.field(static=True)
    action_scale: tuple[float, ...] = eqx.field(static=True)
    _sigma_is_softplus: bool = eqx.field(static=True)
    network: MultilayerPerceptron
    # log sigma of cov_type fixed: a constant, so it is static (torch buffer) and never a parameter
    _fixed_cov: tuple[float, ...] | None = eqx.field(static=True, default=None)
    # log sigma of cov_type single
    cov: Float[Array, "1 A"] | None = None

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray) -> None:
        """
        Initializes a new diagonal gaussian policy.
        :param mc: The model configuration to use.
        :param key: The random key of the network initialization.
        """
        assert isinstance(mc, ModelConfiguration)

        run_conf = mc.get("conf")
        self.cov_type = run_conf["cov_type"]
        self.action_dim = mc.env.action_dim
        a_dim = self.action_dim

        if self.cov_type == "state":
            self.num_output_net = 2 * a_dim
        elif self.cov_type == "fixed":
            # constant standard deviation, stored in log space and not trained
            self._fixed_cov = (float(np.log(run_conf["sigma"])),) * a_dim
            self.num_output_net = a_dim
        elif self.cov_type == "single":
            # one state independent standard deviation per action dimension, stored in log space
            self.cov = jnp.full([1, a_dim], np.log(run_conf.get("sigma", 1.0)), DTYPE)
            self.num_output_net = a_dim
        elif self.cov_type == "full":
            # mean and the packed lower-triangular Cholesky factor L(s)
            self.num_output_net = a_dim + a_dim * (a_dim + 1) // 2
        else:
            raise NotImplementedError(f"unknown covariance type {self.cov_type}")

        # optionally squash the gaussian with tanh into the action bounds of the environment
        self.squash = bool(run_conf.get("squash", False))
        low = np.asarray(mc.env.action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(mc.env.action_space.high, dtype=np.float64).reshape(-1)
        if self.squash and not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("squashing needs a bounded action space")
        center, scale = np.zeros(a_dim), np.ones(a_dim)
        if self.squash:
            center, scale = (high + low) / 2, (high - low) / 2
        # float32 like the torch buffers, kept as static tuples so they are constants and never parameters
        self.action_center = tuple(float(c) for c in center.astype(np.float32))
        self.action_scale = tuple(float(s) for s in scale.astype(np.float32))

        # create policy network, the activations of the mu / sigma outputs are applied inside of the network.
        # a full covariance keeps one activation name per layer and softplus-es only diag L
        sd = mc.env.observation_dim
        structure = np.concatenate(([sd], run_conf["structure"], [self.num_output_net]))
        if self.cov_type == "full":
            af = list(run_conf["act_fn"])
            last = af[-1]
            if isinstance(last, list | tuple):
                raise ValueError(
                    "the full-covariance head uses one activation name; softplus is applied to the Cholesky diagonal"
                )
            sigma_acts: list[Any] = [last]
        else:
            af, sigma_acts = head_activations(run_conf["act_fn"], a_dim)
        self._sigma_is_softplus = sigma_is_softplus(sigma_acts)
        network = MultilayerPerceptron(
            mc, hidden_neurons=structure, act_fn=af, layer_norm=run_conf["layer_norm"], init=run_conf["init"], key=key
        )
        if self.cov_type in ("state", "full"):
            # softplus(bias) + SIGMA_MIN = policy_sigma, matching the std single and fixed start from
            raw = math.log(math.expm1(float(run_conf.get("sigma", 1.0)) - SIGMA_MIN))
            layer = network.layers[-1]
            # covariance outputs are the bias alone, so sigma and L(s) = policy_sigma * I at every state
            weight = layer.weight.at[a_dim:].set(0.0)
            if self.cov_type == "state":
                bias = layer.bias.at[a_dim:].set(raw)
            else:
                diag = np.array([a_dim + i for i in self._diag_index()])
                bias = layer.bias.at[a_dim:].set(0.0).at[diag].set(raw)
            network = eqx.tree_at(lambda n: (n.layers[-1].weight, n.layers[-1].bias), network, (weight, bias))
        self.network = network

    @property
    def _center(self) -> Float[Array, "A"]:
        return jnp.asarray(self.action_center, DTYPE)

    @property
    def _scale(self) -> Float[Array, "A"]:
        return jnp.asarray(self.action_scale, DTYPE)

    def get_suff_stats_dim(self) -> int:
        """
        :return: 2 * A, the means followed by the standard deviations, or A + A(A+1)/2 for a full Cholesky factor.
        """
        a_dim = self.action_dim
        if self.cov_type == "full":
            return a_dim + a_dim * (a_dim + 1) // 2
        return 2 * a_dim

    def forward(self, states: Float[Array, "D Ds"]) -> Float[Array, "D S"]:
        """
        Produces the sufficient statistics of the policy.
        :param states: A matrix of shape [D, Ds] where each row corresponds to one state.
        :return: A matrix of shape [D, 2A], ``[mu, sigma]`` with sigma strictly positive, or ``[mu, packed L]`` when
            ``cov_type`` is ``full``.
        """
        states = jnp.asarray(states, DTYPE)
        a_dim = self.action_dim
        if self.cov_type == "full":
            out = self.network.forward(states)
            mu, raw = out[:, :a_dim], out[:, a_dim:]
            return jnp.concatenate([mu, self._with_positive_diagonal(raw)], axis=1)
        if self.cov_type == "state":
            out = self.network.forward(states)
            mu, sigma = out[:, :a_dim], out[:, a_dim:]
            # the configured activation of the sigma outputs already is a softplus: only guard against underflow,
            # otherwise the unconstrained output is made positive here (never both, a second softplus floors sigma)
            sigma = jnp.maximum(sigma, SIGMA_MIN) if self._sigma_is_softplus else jax.nn.softplus(sigma) + SIGMA_MIN
        else:
            mu = self.network.forward(states)
            log_sigma = self.cov if self.cov_type == "single" else jnp.asarray(self._fixed_cov, DTYPE)[None]
            sigma = jnp.broadcast_to(jnp.exp(log_sigma), mu.shape)
        return jnp.concatenate([mu, sigma], axis=1)

    def _diag_index(self) -> list[int]:
        """Row-major index of each diagonal entry inside a packed lower triangle."""
        return [i * (i + 3) // 2 for i in range(self.action_dim)]

    def _with_positive_diagonal(self, packed: Float[Array, "D L"]) -> Float[Array, "D L"]:
        """``softplus(raw) + SIGMA_MIN`` on the diagonal of a packed Cholesky factor. Off-diagonals stay raw."""
        idx = np.array(self._diag_index())
        diagonal = packed[:, idx]
        diagonal = jnp.maximum(diagonal, SIGMA_MIN) if self._sigma_is_softplus else jax.nn.softplus(diagonal) + SIGMA_MIN
        return packed.at[:, idx].set(diagonal)

    def _mu_factor(self, suff_stats: Float[Array, "D S"]) -> tuple[Float[Array, "D A"], Float[Array, "D A A"]]:
        """Mean and the lower-triangular Cholesky factor ``L`` of a full-covariance statistic row."""
        a_dim = self.action_dim
        mu, packed = suff_stats[:, :a_dim], suff_stats[:, a_dim:]
        rows, cols = np.tril_indices(a_dim)
        factor = jnp.zeros((packed.shape[0], a_dim, a_dim), packed.dtype).at[:, rows, cols].set(packed)
        return mu, factor

    def _split(self, suff_stats: Float[Array, "D S"]) -> tuple[Float[Array, "D A"], Float[Array, "D A"]]:
        return suff_stats[:, : self.action_dim], suff_stats[:, self.action_dim :]

    def _suff_stats(self, states: Any, suff_stats: Any) -> Float[Array, "D S"]:
        assert (states is None) != (suff_stats is None), "pass either states or suff_stats"
        if suff_stats is None:
            assert states is not None
            return self.forward(states)
        return suff_stats

    def log_prob(self, actions: Any, states: Any = None, suff_stats: Any = None) -> Float[Array, "D K"]:
        """
        Log probability of actions relative to states.
        :param actions: A tensor of size [D, K, A], or [D, A] for one action per state.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :return: A matrix [D, K] containing the log probability of each action.
        """
        stats = self._suff_stats(states, suff_stats)
        actions = jnp.asarray(actions, DTYPE)
        if self.cov_type == "full":
            return self._full_log_prob(actions, stats)
        mu, sigma = self._split(stats)
        if actions.ndim == 2:
            actions = actions[:, None]
        mu, sigma = mu[:, None], sigma[:, None]
        if not self.squash:
            return normal_log_prob(actions, mu, sigma).sum(axis=-1)

        # change of variables a = center + scale * tanh(u): log p(a) = log N(u) - sum log(scale * (1 - tanh(u)^2))
        y = jnp.clip((actions - self._center) / self._scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        log_jac = jnp.log(self._scale * (1 - y**2))
        return (normal_log_prob(u, mu, sigma) - log_jac).sum(axis=-1)

    def _full_log_prob(self, actions: Any, suff_stats: Float[Array, "D S"]) -> Float[Array, "D K"]:
        """Log density of ``N(mu, L L^T)``, with the same squash as the diagonal policy."""
        mu, factor = self._mu_factor(suff_stats)
        if actions.ndim == 2:
            actions = actions[:, None]
        mu, factor = mu[:, None], factor[:, None]
        if not self.squash:
            return _mvn_log_prob(actions, mu, factor)
        y = jnp.clip((actions - self._center) / self._scale, -SQUASH_BOUND, SQUASH_BOUND)
        u = jnp.arctanh(y)
        log_jac = jnp.log(self._scale * (1 - y**2)).sum(axis=-1)
        return _mvn_log_prob(u, mu, factor) - log_jac

    def mode(self, states: Any = None, suff_stats: Any = None) -> Float[Array, "D A"]:
        """
        :return: A matrix [D, A] with the most likely (mean) action of each state.
        """
        mu, _ = self._split(self._suff_stats(states, suff_stats))
        return self._center + self._scale * jnp.tanh(mu) if self.squash else mu

    def sample_actions(
        self, key: PRNGKeyArray, states: Any = None, suff_stats: Any = None, num_actions: int = 1
    ) -> Float[Array, "D K A"]:
        """
        Samples K actions for each state.
        :param key: The random key.
        :param states: A matrix of size [D, Ds]. If this None => suff_stats not None
        :param suff_stats: A matrix of size [D, 2A]. If this None => states is not None
        :param num_actions: The number of actions K.
        :return: A tensor [D, K, A] of (reparametrised) action samples.
        """
        stats = self._suff_stats(states, suff_stats)
        eps = jax.random.normal(key, (stats.shape[0], num_actions, self.action_dim), DTYPE)
        if self.cov_type == "full":
            mu, factor = self._mu_factor(stats)
            u = mu[:, None, :] + jnp.einsum("dij,dkj->dki", factor, eps)
        else:
            mu, sigma = self._split(stats)
            u = mu[:, None, :] + eps * sigma[:, None, :]
        if self.squash:
            u = self._center + self._scale * jnp.tanh(jnp.clip(u, -SQUASH_U_MAX, SQUASH_U_MAX))
        return u

    def clone(self) -> "DiagonalGaussianPolicy":
        """
        Clones the current instance. Modules are immutable, so the clone is the instance itself.
        :return: A new identical clones instance of the current.
        """
        return self

    def sync(self, v: "DiagonalGaussianPolicy") -> "DiagonalGaussianPolicy":
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        return self

    def entropy(self, state: ModelState, batch: Batch) -> Float[Array, ""]:
        if self.squash:
            # tanh squashed gaussians have no closed form entropy, the gaussian entropy would ignore the jacobian
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        stats = self.forward(jnp.asarray(batch.states, DTYPE))
        mask = batch_mask(batch)
        if self.cov_type == "full":
            # 1/2 * A * log(2 pi e) + sum log diag(L), mean over the batch
            _, factor = self._mu_factor(stats)
            log_diag = jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=-1)
            return masked_mean(0.5 * self.action_dim * math.log(2 * math.pi * math.e) + log_diag, mask)
        _, sigma = self._split(stats)
        return masked_mean(normal_entropy(sigma).sum(axis=1), mask)

    def kl_divergence(
        self,
        state: ModelState,
        batch: Batch,
        suff_stats: Any = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> Float[Array, ""]:
        """
        (docstring unchanged, plus)
        :param state: The traced model state (gives the target policy).
        """
        states = jnp.asarray(batch.states, DTYPE)
        mask = batch_mask(batch)
        if suff_stats is None:
            assert states is not None
            suff_stats = self.forward(states)
        t_stats = jax.lax.stop_gradient(state.get("policy", True).forward(states))
        if self.cov_type == "full":
            mu, factor = self._mu_factor(suff_stats)
            t_mu, t_factor = self._mu_factor(t_stats)
            if mode == "i-projection":
                kl_full = _mvn_kl(mu, factor, t_mu, t_factor)
            elif mode == "m-projection":
                kl_full = _mvn_kl(t_mu, t_factor, mu, factor)
            else:
                raise NotImplementedError(mode)
            return masked_reduce(kl_full, mask, est)
        mu, sigma = self._split(suff_stats)
        t_mu, t_sigma = self._split(t_stats)

        if mode == "i-projection":
            kl = normal_kl(mu, sigma, t_mu, t_sigma)
        elif mode == "m-projection":
            kl = normal_kl(t_mu, t_sigma, mu, sigma)
        else:
            raise NotImplementedError(mode)
        kl = kl.sum(axis=1)
        return masked_reduce(kl, mask, est)
```

- [ ] **Step 4: Convert `select_policy.py` (already applied in Task 8, Step 5b; verify it matches)**

```python
from jaxtyping import PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def select_policy(mc: ModelConfiguration, key: PRNGKeyArray) -> UnivariateGaussianPolicy | DiagonalGaussianPolicy:
    """
    (docstring unchanged, plus)
    :param key: The random key of the network initialization.
    """
    if mc["conf"].get("cov_type", "single") == "full":
        return DiagonalGaussianPolicy(mc, key)
    kind = mc["conf"].get("type", "auto")
    if kind == "auto":
        kind = "univariate" if mc.env.action_dim == 1 else "diagonal"
    if kind == "univariate":
        return UnivariateGaussianPolicy(mc, key)
    if kind == "diagonal":
        return DiagonalGaussianPolicy(mc, key)
    raise NotImplementedError(f"unknown policy type {kind}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.policy.continuous.diagonal_gaussian_policy"` and
`"abstract_rl.policy.continuous.select_policy"`.

Run: `uv run pytest tests/test_diagonal_policy.py tests/test_policies.py tests/test_parity_networks.py -v`
Expected: all passed, including 4 diagonal parity cases.

- [ ] **Step 6: Commit**

```bash
git add src/abstract_rl/policy/continuous/diagonal_gaussian_policy.py src/abstract_rl/policy/continuous/select_policy.py \
    src/abstract_rl/policy/continuous/univariate_gaussian_policy.py src/abstract_rl/policy/continuous/continuous_policy.py \
    tests/test_diagonal_policy.py tests/test_parity_networks.py tests/conftest.py
git commit -m "Port the diagonal and full-covariance Gaussian policy to Equinox.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `VanillaGradient`, `FullyVNeuralNet` and `FullyQNeuralNet`

**Files:**
- Modify: `src/abstract_rl/algorithms/mixed/vanilla_gradient.py` (whole file)
- Modify: `src/abstract_rl/value_functions/fc_v_networks.py`, `src/abstract_rl/value_functions/fc_q_networks.py` (whole files)
- Modify: `tests/test_parity_networks.py` (append)
- Create: `tests/test_value_functions.py`
- Modify: `tests/conftest.py` (`JAXTYPED_MODULES`)

**Interfaces:**
- Consumes: `ModelState`, `Batch`, `masked_mean`, `get_flat_params_from`, `MultilayerPerceptron(..., key=)`, `DTYPE`.
- Produces, `VanillaGradient`, an `eqx.Module` mixin with the static fields `lr` and `mc_key` and the class constants
  `fmtstr` and `titlestr`:
  - `init_opt() -> optax.OptState`;
  - `perform_mult_sgd(state, key, num_steps, tc, batch_size, fields) -> tuple[ModelState, dict[str, Array]]`. It reads
    the opt state from `state.get(f"{self.mc_key}_opt")` and writes the model under `mc_key` and the opt state back.
    The metrics are `{"k_pre_post": Float[Array, "num_steps 2"]}`, and the post value is NaN unless `k % 10 == 0`.
  - `perform_adam_step(opt, opt_state, curst, batch) -> tuple[model, opt_state, pre_obj, post_obj]`.
  - `tc` must provide `sample(key, batch_size, copy_fields) -> Batch`. Plan 2 implements it on `TrajectoryCollection`.
- Produces, `FullyVNeuralNet(mc, key, mc_key="v_network")`:
  - `forward(states) -> [D, 1]` and `v_val(states) -> [D, 1]`;
  - `tr_obj(batch) -> scalar` (the MSE on `batch["tv"]` plus `l2 * ||w||`);
  - `clone()` and `sync()`.
- Produces, `FullyQNeuralNet(mc, key, mc_key="q_network")`:
  - `forward(states, actions)` and `q_val(states, actions) -> [D, 1]`;
  - `q_val_multi(states, actions [D, K] | [D, K, A]) -> [D, K]`, keeping the `repeat_interleave` row order;
  - `v_val(state, key, states, exp_actions=10, policy=None) -> [D, 1]`, defaulting to `state.get("policy", True)`;
  - `tr_obj(batch)` (the MSE on `batch["tq"]`).

- [ ] **Step 1: Write the failing tests**

`tests/test_value_functions.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


class _StubCollection:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, key, batch_size, copy_fields):
        return self.batch


def _v_batch(n=32):
    s = jax.random.normal(jax.random.key(0), (n, 3))
    z = jnp.zeros((n, 1))
    return Batch(s, z, z, z, s, z, {"tv": jnp.sin(s[:, 0]) * 3})


def test_perform_mult_sgd_fits_and_threads_the_state(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc, jax.random.key(0))
    state = ModelState.create({"v_network_opt": v.init_opt()}, {"v_network": (v, v)})
    batch = _v_batch()
    before = float(v.tr_obj(batch))
    step = jax.jit(lambda st, k: st.get("v_network").perform_mult_sgd(st, k, 50, _StubCollection(batch), 32, ["tv"]))
    state, metrics = step(state, jax.random.key(1))
    assert float(state.get("v_network").tr_obj(batch)) < before
    assert metrics["k_pre_post"].shape == (50, 2)
    assert jnp.all(jnp.isnan(metrics["k_pre_post"][1:10, 1])) and jnp.isfinite(metrics["k_pre_post"][10, 1])
    # the target is untouched by the fit
    np.testing.assert_allclose(state.get("v_network", True).net.layers[0].weight, v.net.layers[0].weight)


def test_q_val_multi_keeps_the_repeat_interleave_order(make_pendulum_mc):
    mc = make_pendulum_mc(alg="mpo", val_struct=[256, 256], val_act_fn=["elu", "elu", "none"])
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc, jax.random.key(0))
    states = jax.random.normal(jax.random.key(1), (5, 3))
    actions = jax.random.uniform(jax.random.key(2), (5, 4), minval=-2, maxval=2)
    multi = q.q_val_multi(states, actions)
    single = jnp.stack([q.q_val(states, actions[:, k : k + 1])[:, 0] for k in range(4)], axis=1)
    np.testing.assert_allclose(multi, single, atol=1e-6)
```

Append to `tests/test_parity_networks.py`:

```python
import optax

from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


def test_v_net_matches_torch(make_pendulum_mc):
    g = golden("v_net")
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = load_named(FullyVNeuralNet(mc, jax.random.key(0)), sub(g, "params"))
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(jnp.asarray(g["states"]), z, z, z, jnp.asarray(g["states"]), z, {"tv": jnp.asarray(g["tv"])})
    np.testing.assert_allclose(v.forward(jnp.asarray(g["states"])), g["forward"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v.tr_obj(batch), g["obj"], rtol=1e-5)
    _close_grads(eqx.filter_grad(lambda m: m.tr_obj(batch))(v), sub(g, "grads"))
    opt = optax.adam(v.lr)
    opt_state = v.init_opt()
    for s in range(3):
        v, opt_state, _, _ = v.perform_adam_step(opt, opt_state, s, batch)
    got = named_parameters(v)
    for k, expected in sub(g, "adam3/params").items():
        np.testing.assert_allclose(got[k], expected, rtol=1e-4, atol=1e-6, err_msg=k)


def test_q_net_matches_torch(make_pendulum_mc):
    g = golden("q_net")
    # the golden Q net was built from the MPO conf (val_struct [256, 256], elu), the fixture conf is the TRPO one
    mc = make_pendulum_mc(alg="mpo", val_struct=[256, 256], val_act_fn=["elu", "elu", "none"])
    with mc["conf"].ns("val"):
        q = load_named(FullyQNeuralNet(mc, jax.random.key(0)), sub(g, "params"))
    s, a = jnp.asarray(g["states"]), jnp.asarray(g["actions"])
    z = jnp.zeros((len(s), 1))
    batch = Batch(s, a, z, z, s, z, {"tq": jnp.asarray(g["tq"])})
    np.testing.assert_allclose(q.q_val(s, a), g["q_val"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q.q_val_multi(s, jnp.asarray(g["add_actions"])), g["q_val_multi"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q.tr_obj(batch), g["obj"], rtol=1e-5)
    _close_grads(eqx.filter_grad(lambda m: m.tr_obj(batch))(q), sub(g, "grads"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_value_functions.py tests/test_parity_networks.py -k "v_net or q_net or sgd or interleave" -v`
Expected: FAIL. `FullyVNeuralNet(mc, key)` raises a `TypeError`.

- [ ] **Step 3: Convert `vanilla_gradient.py`**

```python
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch


class VanillaGradient(eqx.Module):
    """Simple vanilla gradient. Simply inherit from this class to add support.

    Value networks are fit with Adam on minibatches. The Adam state lives in the model state under
    ``<mc_key>_opt``; the printed table of the torch version is returned as metrics.
    """

    fmtstr: ClassVar[str] = "%10i | %14.10f | %14.10f"
    titlestr: ClassVar[str] = "%10s | %14s | %14s"
    lr: float = eqx.field(static=True)
    mc_key: str = eqx.field(static=True)

    def init_opt(self) -> optax.OptState:
        """The initial Adam state of this model."""
        return optax.adam(self.lr).init(eqx.filter(self, eqx.is_inexact_array))

    def perform_mult_sgd(
        self, state: ModelState, key: PRNGKeyArray, num_steps: int, tc: Any, batch_size: int, fields: list[str]
    ) -> tuple[ModelState, dict[str, Array]]:
        """
        Performs multiple stochastic gradient descent steps.

        :param state: The traced model state, holds this model under ``mc_key`` and its Adam state.
        :param key: The random key of the minibatch sampling.
        :param num_steps: The number of steps.
        :param tc: The trajectory collection to sample from.
        :param batch_size: The batch size to use for each batch.
        :param fields: The additional fields, which need to be sampled.
        :return: The new state and the pre / post objectives of every step (``titlestr`` columns).
        """
        opt = optax.adam(self.lr)

        def step(carry: tuple[Any, Any, PRNGKeyArray], s: Int[Array, ""]) -> tuple[tuple[Any, Any, PRNGKeyArray], Array]:
            model, opt_state, key = carry
            key, sub = jax.random.split(key)
            batch = tc.sample(sub, batch_size, fields)
            model, opt_state, obj, new_obj = model.perform_adam_step(opt, opt_state, s, batch)
            return (model, opt_state, key), jnp.stack([obj, new_obj])

        init = (self, state.get(f"{self.mc_key}_opt"), key)
        (model, opt_state, _), objs = jax.lax.scan(step, init, jnp.arange(num_steps))
        state = state.set(self.mc_key, model).set(f"{self.mc_key}_opt", opt_state)
        return state, {"k_pre_post": objs}

    def perform_adam_step(
        self, opt: optax.GradientTransformation, opt_state: optax.OptState, curst: Int[Array, ""] | int, batch: Batch
    ) -> tuple[Any, optax.OptState, Float[Array, ""], Float[Array, ""]]:
        """Performs a single adam step on the passed batch.

        :param opt: The Adam transformation.
        :param opt_state: Its state.
        :param curst: The current step.
        :param batch: The batch to use.
        :return: The updated model and Adam state, the objective before and (every 10th step) after the step.
        """
        obj, grads = eqx.filter_value_and_grad(lambda m: m.tr_obj(batch))(self)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(self, eqx.is_inexact_array))
        model = eqx.apply_updates(self, updates)
        new_obj = jax.lax.cond(
            curst % 10 == 0, lambda: model.tr_obj(batch), lambda: jnp.asarray(jnp.nan, obj.dtype)
        )
        return model, opt_state, obj, new_obj

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """The training objective."""
        raise NotImplementedError
```

- [ ] **Step 4: Convert `fc_v_networks.py`**

```python
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean
from abstract_rl.misc.flat_params import get_flat_params_from
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


class FullyVNeuralNet(VanillaGradient, Syncable):
    """Fully connected neural network."""

    l2: float = eqx.field(static=True)
    net: MultilayerPerceptron

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray, mc_key: str = "v_network") -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""

        conf = mc.get("conf")
        struct = conf["struct"]

        # shortcut for state and action dim
        sd = mc.env.observation_dim

        # neural nets
        layer_norm = conf.get("layer_norm", False)
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]
        self.l2 = conf["l2_reg"]

        # create network
        self.net = MultilayerPerceptron(mc, np.concatenate(([sd], struct, [1])), def_fn, layer_norm, init, key=key)

        # set up the adam fit
        self.lr = conf.get("lr", 1e-3)
        self.mc_key = mc_key

    def clone(self) -> "FullyVNeuralNet":
        """Clones the Neural Network, modules are immutable so the clone is the instance itself."""
        return self

    def sync(self, v: "FullyVNeuralNet") -> "FullyVNeuralNet":
        """Clones the Neural Network"""
        return self

    def forward(self, states: Float[Array, "D Ds"]) -> Float[Array, "D 1"]:
        """Propagate the x in a forward fashion through the network."""
        return self.v_val(states)

    def v_val(self, states: Float[Array, "D Ds"]) -> Float[Array, "D 1"]:
        v_vals = self.net.forward(jnp.asarray(states, DTYPE))
        return v_vals

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:

        # build up mse loss for v function
        states = jnp.asarray(batch.states, DTYPE)
        v_vals = self.v_val(states).reshape(-1)
        v_targets = jnp.asarray(batch["tv"], DTYPE).reshape(-1)
        w = get_flat_params_from(self)
        obj = masked_mean((v_vals - v_targets) ** 2, batch_mask(batch)) + self.l2 * jnp.linalg.norm(w)
        return obj
```

Equinox needs the static fields of `VanillaGradient` (`lr`, `mc_key`) assigned in `__init__`. The torch code passed them
through `VanillaGradient.__init__(self, self.net, lr=...)`. In Equinox they are assigned directly, because an
`eqx.Module` subclass with its own `__init__` assigns all fields itself.

- [ ] **Step 5: Convert `fc_q_networks.py`**

Keep the license header. Then:

```python
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, batch_mask, masked_mean
from abstract_rl.misc.typing import DTYPE
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


class FullyQNeuralNet(VanillaGradient, Syncable):
    """Fully connected neural network."""

    batch_size: int = eqx.field(static=True)
    fc_net: MultilayerPerceptron

    def __init__(self, mc: ModelConfiguration, key: PRNGKeyArray, mc_key: str = "q_network") -> None:
        """Create a new neural network, whereas the structure is passed
        as a list of numbers, representing the size of each fully connected
        layer including the input as well as the output layer."""

        conf = mc.get("conf")

        # required settings
        struct = conf["struct"]
        self.batch_size = conf["batch_size"]

        # neural nets
        layer_norm = conf.get("layer_norm", False)
        init = conf.get("init", "orthogonal")
        def_fn = conf["act_fn"]

        # shortcut for state and action dim
        sd = mc.env.observation_dim
        ad = mc.env.action_dim

        # neural nets
        self.fc_net = MultilayerPerceptron(mc, np.concatenate(([sd + ad], struct, [1])), def_fn, layer_norm, init, key=key)

        # set up the adam fit
        self.lr = conf.get("lr", 1e-3)
        self.mc_key = mc_key

    def clone(self) -> "FullyQNeuralNet":
        """Clones the Neural Network, modules are immutable so the clone is the instance itself."""
        return self

    def sync(self, v: "FullyQNeuralNet") -> "FullyQNeuralNet":
        """Clones the Neural Network"""
        return self

    def forward(self, states: Float[Array, "D Ds"], actions: Float[Array, "D A"]) -> Float[Array, "D 1"]:
        """Propagate the x in a forward fashion through the network."""

        return self.q_val(states, actions)

    def v_val(
        self,
        state: ModelState,
        key: PRNGKeyArray,
        states: Float[Array, "D Ds"],
        exp_actions: int = 10,
        policy: Any = None,
    ) -> Float[Array, "D 1"]:
        """Estimate v values by averaging q values of actions sampled from the target policy.

        :param state: The traced model state (gives the target policy).
        :param key: The random key of the action samples.
        :param states: A matrix [D, Ds] of states.
        :param exp_actions: The number of sampled actions K per state.
        :param policy: The policy to evaluate, defaults to the target policy.
        :return: A matrix [D, 1] of v values.
        """

        policy = policy if policy is not None else state.get("policy", True)
        samp_actions = policy.sample_actions(key, states, num_actions=exp_actions)
        q_vals = self.q_val_multi(states, samp_actions)
        return jnp.mean(q_vals, axis=1, keepdims=True)

    def q_val_multi(self, states: Float[Array, "D Ds"], actions: Any) -> Float[Array, "D K"]:
        """Evaluate K actions per state.

        :param states: A matrix [D, Ds] of states.
        :param actions: A matrix [D, K] of (univariate) actions or a tensor [D, K, A] for A action dimensions.
        :return: A matrix [D, K] of q values.
        """
        d, k = actions.shape[:2]
        rep_states = jnp.repeat(jnp.asarray(states, DTYPE), k, axis=0)
        q_vals = self.q_val(rep_states, actions.reshape([d * k, -1]))
        return q_vals.reshape([d, k])

    def q_val(self, states: Float[Array, "D Ds"], actions: Float[Array, "D A"]) -> Float[Array, "D 1"]:

        comb = jnp.concatenate([jnp.asarray(states, DTYPE), jnp.asarray(actions, DTYPE)], axis=1)
        q_vals = self.fc_net.forward(comb)
        return q_vals

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:

        # build up mse loss for q function
        t_states = jnp.asarray(batch.states, DTYPE)
        t_actions = jnp.asarray(batch.actions, DTYPE)
        q_vals = self.q_val(t_states, t_actions).reshape(-1)
        q_targets = jnp.asarray(batch["tq"], DTYPE).reshape(-1)
        return masked_mean((q_vals - q_targets) ** 2, batch_mask(batch))
```

- [ ] **Step 6: Run the tests to verify they pass**

Extend `JAXTYPED_MODULES` with `"abstract_rl.algorithms.mixed.vanilla_gradient"`,
`"abstract_rl.value_functions.fc_v_networks"` and `"abstract_rl.value_functions.fc_q_networks"`.

Run: `uv run pytest tests/test_value_functions.py tests/test_parity_networks.py -v`
Expected: all passed.

Run: `uv run pytest`
Expected: everything passes or is skipped. The skips are `test_training_smoke`, `test_env_wrapper` and
`test_obs_norm`, which Plan 2 re-enables. `test_imports` passes, because torch and JAX modules both import.

- [ ] **Step 7: Commit**

```bash
git add src/abstract_rl/algorithms/mixed/vanilla_gradient.py src/abstract_rl/value_functions/fc_v_networks.py \
    src/abstract_rl/value_functions/fc_q_networks.py tests/test_value_functions.py tests/test_parity_networks.py \
    tests/conftest.py
git commit -m "Port the value networks and their Adam fit to Equinox and Optax.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Roadmap: Plans 2 and 3

Each will be written with the writing-plans skill after this plan is merged, and pinned to the spike's version table.

**Plan 2: Environments, rollouts and TRPO (spec P3 and P4, R4, R5, R7, R8, P1, P2, P6)**
1. The `JaxEnv` protocol and its adapters:
   - gymnax for Pendulum and MountainCarContinuous;
   - Brax-MJX for InvertedPendulum, InvertedDoublePendulum and HalfCheetah;
   - Playground for AcrobotSwingup;
   - a JAX PointMass for tests.
   Each gets an env-contract test.
2. `Trajectory` and `TrajectoryCollection` as padded pytrees. That covers rounds, masks, `to_batch`, `sample(key, ...)`,
   and `merge` with the exact capping rule, tested against the list rule. `TrajectoryBuilder` becomes a view.
3. `MCMCEnvWrapper.execute_policy` as a jitted vmapped padded rollout. It also covers obs preprocessing, the reward
   transform, `RunningCenteredFilter` and obs-norm, all with fixture parity. `test_env_wrapper` and `test_obs_norm` are
   re-enabled.
4. The TD, GAE and Filter operators with `transform_all = vmap(transform)`, checked against the `adv_*` fixtures.
5. `TrustRegionGradient` with a jvp-of-grad FVP and the `lax.while_loop` line search, plus `TRPOAlgorithm`, checked
   against the `trpo_*` fixtures.
6. `run_seeds` (P6), `run_trpo` as a jitted and vmapped epoch with donation (P1) and `epochs_per_call` (P2), and the
   `seeds` config key. TRPO smoke tests are re-enabled with shape `[seeds, epochs, 2]`. The test that vmap equals the
   per-seed loop and the no-recompile test are added.

**Plan 3: MPO, I/O and the torch removal (spec P5–P7, P4, P5)**
1. The Resample, Retrace and VariationalRanking operators and `NonParametricVariationalPolicyBuilder` over the
   padded replay buffer, checked against the `retrace_*` fixtures.
2. `MPOAlgorithm`, checked against the `mpo_*` fixtures. It drops the logging recompute (P4).
3. `run_mpo_non_parametric` as a jitted and vmapped epoch, with the MPO smoke tests.
4. Checkpoints with `eqx.tree_serialise_leaves` per seed, written in a background thread (P5), plus `store_best`,
   `run_policy`, `eval_balance`, `eval_returns`, GIF rendering on the JAX envs, and the D3 seed fix.
5. Remove torch and gymnasium from `src`, `default_settings`, the dm-control extra and `acrobot_swingup.py`, then add
   `scripts/bench_jax.py`, update the README with benchmark and threshold re-validation, and run the final
   `test_imports`.
