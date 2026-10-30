"""Checkpoints of an MPO training, so a stopped run continues as if it had not been stopped.

A checkpoint is written after an epoch, once the Q fit of that epoch has finished. It holds everything the next epochs
read: the epoch counter and overall step, the policy, the Q network and their target networks, the states of all
optimizers (Adam of the policy and of the Lagrange multipliers, Adam of the Q fit, also in a fit worker), the
multipliers and the sampled trust regions, the temperature eta of the E-step, the replay memory with all annotations,
the logged values, the best return so far, the running statistics of the environment wrapper (observation
normalization, state and reward filters) and the random states of NumPy, MLX, Python and the environment.

The rollout collects whole episodes, so between epochs no episode is in progress and the environment is restored by
its random generator alone (Gymnasium resets MuJoCo completely). A resumed run is therefore bit identical to an
uninterrupted one.

Layout of ``<run>/checkpoint``: ``state.pkl`` (everything but the replay) and one ``.npy`` file per replay field with
the rows of all trajectories concatenated. A checkpoint is written to a temporary directory and renamed, so the last
complete checkpoint survives a crash during the save.
"""

import os
import pickle
import random
import shutil
import time
from glob import glob
from os.path import exists, isdir, join
from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx.utils import tree_unflatten
from numpy.lib.format import open_memmap

from abstract_rl.algorithms.mixed.vanilla_gradient import flat_items, flat_numpy, set_optimizer_state
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.cli_printer import CliPrinter

CHECKPOINT = "checkpoint"
VERSION = 1
# the arrays of a trajectory besides its annotations
_TRAJ_ARRAYS = ("_states", "_actions", "_rewards", "_dones", "_log_likelihoods")
_TRAJ_OTHER = ("traj_id", "raw_return", "_annotations", "_trajectory", "env", "_sd", "_ad", "cli")


def checkpoint_dir(run_root: str) -> str:
    return join(run_root, CHECKPOINT)


def has_checkpoint(run_root: str) -> bool:
    """Whether the run has a complete checkpoint (also one left as ``checkpoint.old`` by a crash during a save)."""
    return any(exists(join(d, "state.pkl")) for d in (checkpoint_dir(run_root), checkpoint_dir(run_root) + ".old"))


def latest_resumable_run(runs_dir: str) -> str | None:
    """The newest run directory (by name, a time stamp) below runs_dir with a complete checkpoint."""
    runs = sorted(d for d in glob(join(runs_dir, "*")) if isdir(d) and has_checkpoint(d))
    return runs[-1] if runs else None


def resume_run_root(conf: dict[str, Any]) -> str | None:
    """The run directory to resume as selected by ``resume`` of the run configuration, None for a new run."""
    resume = conf.get("resume", False)
    if resume in (None, False, "false"):
        return None
    if resume in (True, "true", "auto"):
        return latest_resumable_run(join(conf.get("data_root", "data"), conf["env_name"], conf["alg"]))
    assert has_checkpoint(str(resume)), f"{resume} has no checkpoint"
    return str(resume)


def _module_params(module: Any) -> list[tuple[str, np.ndarray]]:
    return flat_numpy(module.parameters())


def _set_module_params(module: Any, flat: list[tuple[str, np.ndarray]]) -> None:
    assert sorted(k for k, _ in flat) == sorted(k for k, _ in flat_items(module.parameters())), "parameter mismatch"
    module.update(tree_unflatten([(k, mx.array(v)) for k, v in flat]))


def _mx_key() -> tuple[int, int]:
    k0, k1 = (int(k) for k in np.array(mx.random.state[0]))  # type: ignore[index]
    return k0, k1


def _write_replay(path: str, trajectories: list[Trajectory]) -> dict[str, Any]:
    """Write the arrays of all trajectories, one file per field, and return their layout."""
    lengths = np.array([len(t) for t in trajectories], dtype=np.int64)
    fields: dict[str, tuple[str, int]] = {}
    for name in _TRAJ_ARRAYS:
        fields[name] = ("np", 0)
    for name in trajectories[0]._annotations:
        fields[name] = ("mx" if isinstance(trajectories[0][name], mx.array) else "np", 0)
    for t in trajectories:
        extra = set(t.__dict__) - set(_TRAJ_ARRAYS) - set(_TRAJ_OTHER)
        assert not extra, f"a checkpoint does not know the trajectory attributes {extra}"
        assert set(t._annotations) == set(trajectories[0]._annotations), "trajectories with different annotations"

    def rows(t: Trajectory, name: str) -> np.ndarray:
        return np.asarray(getattr(t, name) if name in _TRAJ_ARRAYS else t[name])

    for i, name in enumerate(fields):
        first = rows(trajectories[0], name)
        total = sum(rows(t, name).shape[0] for t in trajectories)
        out = open_memmap(join(path, f"f{i}.npy"), mode="w+", dtype=first.dtype, shape=(total, *first.shape[1:]))
        offset = 0
        for t in trajectories:
            a = rows(t, name)
            out[offset : offset + a.shape[0]] = a
            offset += a.shape[0]
        out.flush()
        del out
        fields[name] = (fields[name][0], i)
    meta = [(t.traj_id, t.raw_return) for t in trajectories]
    return {"lengths": lengths, "fields": fields, "meta": meta}


def _read_replay(path: str, layout: dict[str, Any], env: Any) -> list[Trajectory]:
    lengths = layout["lengths"]
    cli = CliPrinter().instance
    trajectories = []
    for traj_id, raw_return in layout["meta"]:
        t = Trajectory.__new__(Trajectory)
        t.traj_id, t.raw_return, t.env, t.cli = traj_id, raw_return, env, cli
        t._sd, t._ad, t._annotations = env.observation_dim, env.action_dim, {}
        trajectories.append(t)
    for name, (kind, i) in layout["fields"].items():
        data = np.load(join(path, f"f{i}.npy"), mmap_mode="r")
        # the states hold one row more than the transitions, the last next state
        sizes = lengths + 1 if data.shape[0] == lengths.sum() + len(lengths) else lengths
        assert data.shape[0] == sizes.sum(), f"replay field {name} has {data.shape[0]} rows"
        flat = mx.array(np.ascontiguousarray(data)) if kind == "mx" else None
        offset = 0
        for t, n in zip(trajectories, sizes, strict=True):
            rows = flat[offset : offset + n] if flat is not None else np.array(data[offset : offset + n])
            if name in _TRAJ_ARRAYS:
                setattr(t, name, rows)
            else:
                t[name] = rows
            offset += n
        del data
    for t, n in zip(trajectories, lengths, strict=True):
        # only the length of the data frame is read after the construction of a trajectory
        t._trajectory = pd.DataFrame(index=pd.RangeIndex(int(n) + 1))
    return trajectories


def save_checkpoint(mc: Any, epoch: int) -> float:
    """Write the checkpoint of the run after ``epoch`` epochs. Returns the time it took in seconds."""
    t0 = time.perf_counter()
    q_network = mc.get("q_network")
    # collects a pending q fit, the same parameters the next epoch would collect after its rollout
    q_opt_state = q_network.optimizer_state()
    alg, rnk, env, log, tc = mc["alg"], mc["rnk"], mc.env, mc["log"], mc["tc"]
    norm = env._norm
    gym_env = env.env.unwrapped
    state = {
        "version": VERSION,
        "epoch": epoch,
        "overall_step": mc.overall_step,
        "best_return": mc.best_return,
        "syncables": {
            k: {"model": _module_params(o), "target": _module_params(t)} for k, (_, o, t) in mc.syncables.items()
        },
        "q_opt": q_opt_state,
        "alg": {
            "opt_outer": flat_numpy(alg.opt_outer.state),
            "opt_inner": flat_numpy(alg.opt_inner.state),
            "lm": [np.array(x) for x in alg._lm_state["lm"]],
            "v": alg.v,
            "v_state": [np.array(x) for x in alg._v_state["v"]],
        },
        "rnk": {"eta": np.array(rnk.b.eta), "eta_mx": np.array(rnk.b.eta_mx)},
        "log": {"header": dict(log._header), "data": {k: list(v) for k, v in log._data.items()}},
        "env": {
            "epoch": env.epoch,
            "seed": env._seed,
            "best_reward": env.best_reward,
            "max_traj_len": env.max_traj_len,
            "created_trajectories": env.created_trajectories,
            "total_steps": getattr(env, "total_steps", None),
            "state_filter": dict(env.state_filter.__dict__, cli=None),
            "reward_filter": dict(env.reward_filter.__dict__, cli=None),
            "obs_rms": None
            if norm is None
            else (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count, norm.update_running_mean),
            "np_random": gym_env.np_random.bit_generator.state,
            "action_space_random": env.act_sp.np_random.bit_generator.state,
        },
        "random": {
            "numpy": np.random.get_state(),  # noqa: NPY002
            "mlx": _mx_key(),
            "python": random.getstate(),
        },
        "replay": None,
    }

    final = checkpoint_dir(mc.run_root)
    tmp = f"{final}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    state["replay"] = _write_replay(tmp, tc.trajectories())
    tc_max_l = tc.max_l
    state["tc_max_l"] = tc_max_l
    with open(join(tmp, "state.pkl"), "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    # the complete new checkpoint replaces the old one, a crash in between leaves checkpoint.old (see load)
    old = f"{final}.old"
    shutil.rmtree(old, ignore_errors=True)
    if exists(final):
        os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
    return time.perf_counter() - t0


def load_checkpoint(mc: Any) -> int:
    """Restore the run from its checkpoint, the objects of mc have to be built as for a new run. Returns the epoch."""
    path = checkpoint_dir(mc.run_root)
    if not exists(join(path, "state.pkl")):
        path = f"{path}.old"
    with open(join(path, "state.pkl"), "rb") as f:
        state = pickle.load(f)
    assert state["version"] == VERSION, f"checkpoint version {state['version']} is not {VERSION}"

    mc.overall_step = state["overall_step"]
    mc.best_return = state["best_return"]
    for k, (_, o, t) in mc.syncables.items():
        _set_module_params(o, state["syncables"][k]["model"])
        _set_module_params(t, state["syncables"][k]["target"])
    mc.get("q_network").set_optimizer_state(state["q_opt"])

    alg = mc["alg"]
    set_optimizer_state(alg.opt_outer, state["alg"]["opt_outer"])
    set_optimizer_state(alg.opt_inner, state["alg"]["opt_inner"])
    alg._lm_state["lm"][:] = [mx.array(x) for x in state["alg"]["lm"]]
    alg.v = state["alg"]["v"]
    alg._v_state["v"] = [mx.array(x) for x in state["alg"]["v_state"]]

    rnk = mc["rnk"]
    rnk.b.eta = state["rnk"]["eta"]
    rnk.b.eta_mx = mx.array(state["rnk"]["eta_mx"])

    log = mc["log"]
    assert set(log._header) == set(state["log"]["header"]), "the logged fields changed"
    log._data = {k: list(v) for k, v in state["log"]["data"].items()}
    # the log files of the epochs after the checkpoint are written again
    for key, values in log._data.items():
        with open(join(log._filepath, f"{key}.txt"), "w") as handle:
            for value in values:
                np.savetxt(handle, value)

    env = mc.env
    es = state["env"]
    env.epoch, env._seed, env.best_reward = es["epoch"], es["seed"], es["best_reward"]
    env.max_traj_len, env.created_trajectories = es["max_traj_len"], es["created_trajectories"]
    if es["total_steps"] is not None:
        env.total_steps = es["total_steps"]
    for name in ("state_filter", "reward_filter"):
        f = getattr(env, name)
        f.__dict__.update({k: v for k, v in es[name].items() if k != "cli"})
    if es["obs_rms"] is not None:
        mean, var, count, update = es["obs_rms"]
        env._norm.obs_rms.mean, env._norm.obs_rms.var, env._norm.obs_rms.count = mean.copy(), var.copy(), count
        env._norm.update_running_mean = update
    env.env.unwrapped.np_random.bit_generator.state = es["np_random"]
    env.act_sp.np_random.bit_generator.state = es["action_space_random"]

    tc = mc["tc"]
    assert tc.max_l == state["tc_max_l"], "the size of the replay memory changed"
    tc.extend(_read_replay(path, state["replay"], env))

    np.random.set_state(state["random"]["numpy"])  # noqa: NPY002
    k0, k1 = state["random"]["mlx"]
    mx.random.seed((k0 << 32) | k1)
    random.setstate(state["random"]["python"])
    return int(state["epoch"])
