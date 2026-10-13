"""On-policy rollouts for the classic-control JAX steps.

``lax.scan`` walks one episode and ``vmap`` walks the parallel environments. A ``lax.while_loop``
starts another round of episodes until the transition budget is met. The last round is finished,
so the collection can run past the budget, as the Gymnasium vector env does. Time-limit stops are
not dones. The policy density is the JAX Gaussian, the same functions the update uses.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.env.jax_classic import (
    MOUNTAIN_CAR,
    PENDULUM,
    mountain_obs,
    mountain_reset,
    mountain_step,
    pendulum_obs,
    pendulum_reset,
    pendulum_step,
)
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, act_deterministic, act_stochastic


class _Round(NamedTuple):
    obs: Array
    act: Array
    shaped: Array
    logp: Array
    done: Array
    length: Array
    raw: Array
    final: Array


class _Buf(NamedTuple):
    key: Array
    cursor: Array
    n_ep: Array
    obs: Array
    act: Array
    shaped: Array
    logp: Array
    done: Array
    ep_start: Array
    ep_len: Array
    ep_raw: Array
    ep_final: Array


def _write_rows(dest: Array, start: Array, values: Array, length: Array) -> Array:
    """Write ``values[:length]`` at ``start``. Invalid rows land on a sentinel that is never read."""
    steps = values.shape[0]
    positions = start + jnp.arange(steps)
    valid = jnp.arange(steps) < length
    sentinel = dest.shape[0] - 1
    safe = jnp.where(valid, positions, sentinel)
    valid_b = valid.reshape((-1,) + (1,) * (values.ndim - 1))
    mixed = jnp.where(valid_b, values, dest[safe])
    return dest.at[safe].set(mixed)


def _episode(
    key: PRNGKeyArray,
    params: GaussianParams,
    max_steps: int,
    deterministic: bool,
    normalize: bool,
    obs_scale: Float[Array, " obs"],
    reward_scale: Float[Array, ""],
    reward_offset: Float[Array, ""],
    env_name: str,
) -> _Round:
    match env_name:
        case _ if env_name == PENDULUM:
            reset, step, observe = pendulum_reset, pendulum_step, pendulum_obs
        case _ if env_name == MOUNTAIN_CAR:
            reset, step, observe = mountain_reset, mountain_step, mountain_obs
        case _:
            raise ValueError(f"no JAX step for {env_name}")

    key, reset_key = jax.random.split(key)
    physics = reset(reset_key)

    def preprocess(obs: Float[Array, " obs"]) -> Float[Array, " obs"]:
        match normalize:
            case True:
                return obs / obs_scale
            case False:
                return obs

    def one(
        carry: tuple[Array, Array, Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array, Array, Array], tuple[Array, Array, Array, Array, Array]]:
        phys, t, alive, step_key, raw_sum = carry
        step_key, action_key = jax.random.split(step_key)
        proc = preprocess(observe(phys))
        match deterministic:
            case True:
                action, logp = act_deterministic(params, proc[None, :])
            case False:
                action, logp = act_stochastic(params, proc[None, :], action_key)
        action = action[0]
        logp = logp[0]
        nxt, reward, terminated = step(phys, action)
        record = alive
        phys_out = jnp.where(record, nxt, phys)
        raw_out = raw_sum + jnp.where(record, reward, jnp.zeros((), dtype=jnp.float32))
        t_out = t + record.astype(jnp.int32)
        alive_out = alive & ~terminated & ((t + 1) < max_steps)
        shaped = reward_scale * reward + reward_offset
        done = (terminated & record).astype(jnp.float32)
        return (phys_out, t_out, alive_out, step_key, raw_out), (proc, action, shaped, logp, done)

    init = (
        physics,
        jnp.int32(0),
        jnp.bool_(True),
        key,
        jnp.float32(0.0),
    )
    final, traced = jax.lax.scan(one, init, None, length=max_steps)
    phys, length, _alive, _key, raw_sum = final
    proc, action, shaped, logp, done = traced
    return _Round(proc, action, shaped, logp, done, length, raw_sum, preprocess(observe(phys)))


@eqx.filter_jit
def collect_classic(
    params: GaussianParams,
    key: PRNGKeyArray,
    obs_scale: Float[Array, " obs"],
    reward_scale: Float[Array, ""],
    reward_offset: Float[Array, ""],
    env_name: str,
    max_steps: int,
    batch_size: int,
    n_envs: int,
    deterministic: bool,
    normalize: bool,
) -> _Buf:
    """Collect at least ``batch_size`` transitions. ``max_steps``, ``batch_size`` and ``n_envs`` are static."""
    obs_dim = obs_scale.shape[0]
    act_dim = len(params.action_center)
    cap = batch_size + n_envs * max_steps
    max_ep = batch_size + n_envs
    buf = _Buf(
        key=key,
        cursor=jnp.int32(0),
        n_ep=jnp.int32(0),
        obs=jnp.zeros((cap + 1, obs_dim), dtype=jnp.float32),
        act=jnp.zeros((cap + 1, act_dim), dtype=jnp.float32),
        shaped=jnp.zeros((cap + 1,), dtype=jnp.float32),
        logp=jnp.zeros((cap + 1,), dtype=jnp.float32),
        done=jnp.zeros((cap + 1,), dtype=jnp.float32),
        ep_start=jnp.zeros((max_ep,), dtype=jnp.int32),
        ep_len=jnp.zeros((max_ep,), dtype=jnp.int32),
        ep_raw=jnp.zeros((max_ep,), dtype=jnp.float32),
        ep_final=jnp.zeros((max_ep, obs_dim), dtype=jnp.float32),
    )

    def cond(state: _Buf) -> Array:
        return (state.cursor < batch_size) & (state.n_ep + n_envs <= max_ep)

    def body(state: _Buf) -> _Buf:
        keys = jax.random.split(state.key, n_envs + 1)
        rolled = jax.vmap(
            lambda episode_key: _episode(
                episode_key,
                params,
                max_steps,
                deterministic,
                normalize,
                obs_scale,
                reward_scale,
                reward_offset,
                env_name,
            )
        )(keys[1:])

        def write_one(index: int, current: _Buf) -> _Buf:
            length = rolled.length[index]
            start = current.cursor
            slot = current.n_ep
            return current._replace(
                cursor=start + length,
                n_ep=slot + jnp.int32(1),
                obs=_write_rows(current.obs, start, rolled.obs[index], length),
                act=_write_rows(current.act, start, rolled.act[index], length),
                shaped=_write_rows(current.shaped, start, rolled.shaped[index], length),
                logp=_write_rows(current.logp, start, rolled.logp[index], length),
                done=_write_rows(current.done, start, rolled.done[index], length),
                ep_start=current.ep_start.at[slot].set(start),
                ep_len=current.ep_len.at[slot].set(length),
                ep_raw=current.ep_raw.at[slot].set(rolled.raw[index]),
                ep_final=current.ep_final.at[slot].set(rolled.final[index]),
            )

        updated = jax.lax.fori_loop(0, n_envs, write_one, state)
        return updated._replace(key=keys[0])

    return jax.lax.while_loop(cond, body, buf)


def trajectories_from_buffer(env: object, buf: _Buf) -> list[Trajectory]:
    """Host copy of a collected buffer. Episode order matches a serial collector."""
    n_ep = int(buf.n_ep)
    obs = np.asarray(buf.obs)
    act = np.asarray(buf.act)
    shaped = np.asarray(buf.shaped)
    logp = np.asarray(buf.logp)
    done = np.asarray(buf.done)
    ep_start = np.asarray(buf.ep_start)
    ep_len = np.asarray(buf.ep_len)
    ep_raw = np.asarray(buf.ep_raw)
    ep_final = np.asarray(buf.ep_final)
    built: list[Trajectory] = []
    for index in range(n_ep):
        length = int(ep_len[index])
        if length <= 0:
            continue
        start = int(ep_start[index])
        stop = start + length
        states = np.concatenate([obs[start:stop], ep_final[index : index + 1]], axis=0)
        trajectory = _trajectory_from_arrays(
            env,
            states,
            act[start:stop],
            shaped[start:stop],
            logp[start:stop],
            done[start:stop],
        )
        trajectory.raw_return = float(ep_raw[index])
        built.append(trajectory)
    return built


def _trajectory_from_arrays(
    env: object,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    logps: np.ndarray,
    dones: np.ndarray,
) -> Trajectory:
    """Same column layout as ``TrajectoryBuilder.finalize``."""
    num = states.shape[0]
    sd = states.shape[1]
    ad = actions.shape[1]
    frame_actions = np.zeros((num, ad), dtype=np.float64)
    frame_actions[:-1] = actions
    reward_col = np.zeros((num, 1), dtype=np.float64)
    reward_col[:-1, 0] = rewards
    likelihood = np.zeros((num, 1), dtype=np.float64)
    likelihood[:-1, 0] = logps
    done_col = np.zeros((num, 1), dtype=np.float64)
    done_col[:-1, 0] = dones
    data = np.concatenate(
        [np.asarray(states, dtype=np.float64), frame_actions, reward_col, likelihood, done_col], axis=1
    )
    columns = [f"s_{i}" for i in range(sd)]
    columns.extend(f"a_{i}" for i in range(ad))
    columns.extend(["r", "ll", "d"])
    frame = pd.DataFrame(data, columns=columns)
    # ids are assigned by the caller via the trajectory object after construction; use 0 as a placeholder
    return Trajectory(0, env, frame)
