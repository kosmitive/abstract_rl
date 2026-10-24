"""Shard a JAX rollout across host CPU devices and land it in one MLX arena.

One XLA host device uses a handful of cores. The env axis is sharded with
``shard_map`` (``pmap`` is the fallback shape: the same function, the env axis
split explicitly). The policy vector is passed in, replicated, and is a
zero-copy view of the MLX buffer when the device pointers match.

XLA does not write into a caller buffer here. Donation left the arena
untouched on JAX 0.12 CPU, and the per-device results are not one contiguous
allocation, so concatenating them is a copy. Each shard is copied once into
its slice of a preallocated MLX arena. Training then reads that arena.
``mx.from_dlpack(..., copy=False)`` is the zero-copy alternative on machines
where MLX accepts a JAX CPU buffer (an M3 Max does; this Linux CPU build does
not). The arena copy is the path that works on both.

Set ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before the process
imports JAX. ``env_jax_devices`` cannot exceed the devices JAX already sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from abstract_rl.interop.shared import SharedParams, buffer_address
from abstract_rl.interop.spec import GaussianSpec


def _shard_map() -> Any:
    """``jax.shard_map`` on current JAX, the experimental function on older JAX."""
    import importlib

    try:
        fn = importlib.import_module("jax").shard_map
    except AttributeError:
        fn = importlib.import_module("jax.experimental.shard_map").shard_map
    return fn


_MESHES: dict[int, Any] = {}
_COMPILED: dict[tuple[Any, ...], Any] = {}


def _mesh(n_devices: int) -> Any:
    """The host mesh of ``n_devices``, created once so a cached ``shard_map`` keeps its mesh."""
    import jax
    from jax.sharding import Mesh

    mesh = _MESHES.get(n_devices)
    if mesh is None:
        devices = jax.devices("cpu")
        if n_devices > len(devices):
            raise RuntimeError(
                f"env_jax_devices={n_devices} but this process has {len(devices)} JAX CPU devices. "
                "Set XLA_FLAGS=--xla_force_host_platform_device_count="
                f"{n_devices} before Python imports JAX, then start the run again."
            )
        mesh = Mesh(np.array(devices[:n_devices]), ("i",))
        _MESHES[n_devices] = mesh
    return mesh


def _compiled_rollout(
    env: Any,
    spec: GaussianSpec,
    horizon: int,
    n_envs: int,
    n_devices: int,
    deterministic: bool,
    normalize: bool,
    reward_scale: float,
    reward_offset: float,
    rolled: Any,
) -> Any:
    """The jitted sharded scan. A new shape recompiles; a repeated call reuses the jit."""
    import jax
    from jax.sharding import PartitionSpec

    key = (
        id(env),
        id(spec),
        horizon,
        n_envs,
        n_devices,
        bool(deterministic),
        bool(normalize),
        float(reward_scale),
        float(reward_offset),
    )
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled
    # check_vma rejects a vmapped scan whose branches mix a per-env value with a
    # Python constant (the observation scale, a done flag). JAX documents turning
    # the check off. Older JAX has no such argument.
    shard = _shard_map()
    kwargs: dict[str, Any] = {
        "mesh": _mesh(n_devices),
        "in_specs": (PartitionSpec(), PartitionSpec(), PartitionSpec("i")),
        "out_specs": tuple(PartitionSpec("i") for _ in range(8)),
    }
    if "check_vma" in shard.__code__.co_varnames:
        kwargs["check_vma"] = False
    compiled = jax.jit(shard(rolled, **kwargs))
    _COMPILED[key] = compiled
    return compiled


@dataclass
class ArenaField:
    """One float32 tensor in an aligned MLX allocation."""

    base: Any
    view: Any
    numpy: np.ndarray
    shape: tuple[int, ...]

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> ArenaField:
        """Allocate an aligned field of the given shape."""
        from abstract_rl.interop.shared import _aligned_slice

        n = int(np.prod(shape))
        base, view, numpy = _aligned_slice(n)
        return cls(base, view.reshape(shape), numpy.reshape(shape), shape)


@dataclass
class ShardedRollout:
    """Per-env trajectories as views of one arena, plus what the copy cost was."""

    states: Any
    actions: Any
    rewards: Any
    raw_rewards: Any
    log_likelihoods: Any
    terminated: Any
    alive: Any
    lengths: np.ndarray
    raw_returns: np.ndarray
    params_are_views: bool
    n_keep: int
    horizon: int
    _keep: tuple[ArenaField, ...]

    def episode(self, index: int) -> dict[str, Any]:
        """Views of one episode, length ``lengths[index]``.

        Args:
            index: The env index, below ``n_keep``.

        Returns:
            ``states`` includes the final next state, so its leading dimension is ``L + 1``.
        """
        length = int(self.lengths[index])
        return {
            "states": self.states[index, : length + 1],
            "actions": self.actions[index, :length],
            "rewards": self.rewards[index, :length],
            "raw_rewards": self.raw_rewards[index, :length],
            "log_likelihoods": self.log_likelihoods[index, :length],
            "terminated": self.terminated[index, :length],
            "raw_return": float(self.raw_returns[index]),
        }


def _as_varying(tree: Any, obs: Any) -> Any:
    """Mark every leaf as varying along the env axis. The values stay the same.

    A constant built inside ``shard_map`` is invariant in that axis, and a real
    env step is not. ``lax.cond`` rejects the mix, including a done flag that
    is a Python ``False`` on one branch and a per-env boolean on the other.
    Adding a zero taken from this env's observation (or or-ing a false boolean
    built the same way) makes the leaf varying and leaves the stored value
    unchanged. JAX 0.11 has no ``pcast``, and this is that cast.
    """
    import jax
    import jax.numpy as jnp

    anchor = obs.reshape(-1)[0]
    spur = anchor * 0
    spur_bool = (anchor == anchor) & False

    def tie(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            return leaf
        if jnp.issubdtype(dtype, jnp.bool_):
            return leaf | spur_bool
        if jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.integer):
            return leaf + spur.astype(dtype)
        return leaf

    return jax.tree.map(tie, tree)


def _freeze_like(state: Any, obs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """A no-op step: repeat the observation and the state, and record zeros."""
    import jax.numpy as jnp

    zero = jnp.float32(0.0)
    return obs, state, zero, zero, jnp.bool_(False), jnp.bool_(False)


def _fill(field: ArenaField, sharded: Any) -> None:
    """Copy each device shard into its slice. ``shard.data`` on CPU is already a view."""
    sharded.block_until_ready()
    for shard in sharded.addressable_shards:
        field.numpy[shard.index] = np.asarray(shard.data, dtype=np.float32)


def _params_are_views(placed: Any, shared: SharedParams) -> bool:
    """Whether every replica still points at the MLX buffer."""
    want = buffer_address(shared.as_mlx())
    return all(buffer_address(shard.data) == want for shard in placed.addressable_shards)


def sharded_rollout(
    env: Any,
    spec: GaussianSpec,
    shared: SharedParams,
    fixed_log_sigma: np.ndarray,
    keys: Any,
    horizon: int,
    n_devices: int,
    *,
    deterministic: bool,
    normalize: bool,
    obs_scale: np.ndarray,
    reward_scale: float,
    reward_offset: float,
    n_keep: int,
    copy_params: bool = False,
) -> ShardedRollout:
    """Roll ``keys.shape[0]`` envs out for ``horizon`` steps, sharded on the env axis.

    Args:
        env: A JAX environment from ``make_jax_env``.
        spec: The policy layout. Closed over by the compiled function; it is static.
        shared: The policy parameters. Replicated, not donated.
        fixed_log_sigma: Shape ``(A,)`` when the covariance is frozen, otherwise unused.
        keys: One JAX key per env, shape ``(n_envs,)``. ``n_envs`` is a multiple of ``n_devices``.
        horizon: Steps per env. Early termination freezes that env; it does not reset.
        n_devices: How many host CPU devices share the env axis.
        deterministic: Use the mode instead of a sample.
        normalize: Divide observations by ``obs_scale``, as ``RolloutEnvWrapper`` does.
        obs_scale: Per-dimension positive scale.
        reward_scale: Training reward scale.
        reward_offset: Training reward offset.
        n_keep: How many leading envs are real. The rest pad the device count and are dropped.
        copy_params: Copy the vector into a fresh NumPy buffer before ``device_put``.
            The benchmark uses this as the duplicated-parameters baseline. The
            training path leaves it false, which is the shared view.

    Returns:
        The arena and the per-env lengths. Padded envs are present in the arrays and omitted by ``n_keep``.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from abstract_rl.interop.forward import act_one
    from abstract_rl.interop.namespace import JAX

    n_envs = int(keys.shape[0])
    if n_devices < 1:
        raise ValueError("env_jax_devices must be positive")
    if n_envs % n_devices != 0:
        raise ValueError(f"n_envs ({n_envs}) must be a multiple of env_jax_devices ({n_devices})")
    mesh = _mesh(n_devices)
    spec_p = PartitionSpec
    replicated = NamedSharding(mesh, spec_p())
    sharded = NamedSharding(mesh, spec_p("i"))
    host = np.array(shared.numpy_view(), copy=True) if copy_params else shared.numpy_view()
    theta = jax.device_put(host, replicated)
    theta.block_until_ready()
    views = _params_are_views(theta, shared)
    fixed_np = np.asarray(fixed_log_sigma if spec.cov_type == "fixed" else np.zeros(1, np.float32), dtype=np.float32)
    fixed = jax.device_put(fixed_np, replicated)
    key_array = jax.device_put(keys, sharded)

    # Python floats, turned into JAX constants inside the scan. A closed-over device
    # array would be a device-0 input and shard_map would reject it.
    scale_values = tuple(float(v) for v in np.asarray(obs_scale, dtype=np.float64).reshape(-1))
    use_fixed = spec.cov_type == "fixed"

    def prepare(obs: Any) -> Any:
        if not normalize:
            return obs
        return obs / jnp.asarray(scale_values, dtype=jnp.float32)

    def one(theta_local: Any, fixed_local: Any, key: Any) -> tuple[Any, ...]:
        obs0, state = env.reset(key)
        obs0 = prepare(obs0)
        state = _as_varying(state, obs0)

        def body(carry: tuple[Any, Any, Any, Any], _step: Any) -> tuple[tuple[Any, Any, Any, Any], tuple[Any, ...]]:
            state_c, obs, key_c, alive = carry
            key_c, noise_key = jax.random.split(key_c)
            noise = None
            if not deterministic:
                if spec.kind == "univariate":
                    noise = jax.random.normal(noise_key, (1, 1), dtype=jnp.float32)
                else:
                    noise = jax.random.normal(noise_key, (1, 1, spec.action_dim), dtype=jnp.float32)
            action, logp = act_one(
                JAX, spec, theta_local, obs, noise, deterministic, fixed_local if use_fixed else None
            )

            def do_step(state_s: Any, action_s: Any, obs_s: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                obs2, state2, rew, term, trunc = env.step(state_s, action_s)
                stored = rew * jnp.float32(reward_scale) + jnp.float32(reward_offset)
                return _as_varying((prepare(obs2), state2, rew, stored, term, trunc), obs_s)

            def hold(state_s: Any, action_s: Any, obs_s: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                return _as_varying(_freeze_like(state_s, obs_s), obs_s)

            obs2, state2, raw, stored, term, trunc = jax.lax.cond(alive, do_step, hold, state_c, action, obs)
            alive2 = alive & ~(term | trunc)
            step_out = (obs, action, stored, raw, logp, term.astype(jnp.float32), alive.astype(jnp.float32))
            return (state2, obs2, key_c, alive2), step_out

        init = (state, obs0, key, _as_varying(jnp.bool_(True), obs0))
        final, scanned = jax.lax.scan(body, init, None, length=horizon)
        obs_seq, actions, stored, raw, logp, term, alive_f = scanned
        return obs_seq, final[1], actions, stored, raw, logp, term, alive_f

    def rolled(theta_g: Any, fixed_g: Any, keys_g: Any) -> tuple[Any, ...]:
        return jax.vmap(lambda key: one(theta_g, fixed_g, key))(keys_g)

    compiled = _compiled_rollout(
        env,
        spec,
        horizon,
        n_envs,
        n_devices,
        deterministic,
        normalize,
        reward_scale,
        reward_offset,
        rolled,
    )
    obs_seq, final_obs, actions, rewards, raw, logp, term, alive = compiled(theta, fixed, key_array)

    obs_dim, act_dim = spec.obs_dim, spec.action_dim
    states_f = ArenaField.zeros((n_envs, horizon + 1, obs_dim))
    actions_f = ArenaField.zeros((n_envs, horizon, act_dim))
    rewards_f = ArenaField.zeros((n_envs, horizon))
    raw_f = ArenaField.zeros((n_envs, horizon))
    logp_f = ArenaField.zeros((n_envs, horizon))
    term_f = ArenaField.zeros((n_envs, horizon))
    alive_f = ArenaField.zeros((n_envs, horizon))
    _fill(actions_f, actions)
    _fill(rewards_f, rewards)
    _fill(raw_f, raw)
    _fill(logp_f, logp)
    _fill(term_f, term)
    _fill(alive_f, alive)
    # observations before each step, then the final next observation, so a prefix is one episode
    obs_seq.block_until_ready()
    final_obs.block_until_ready()
    for shard in obs_seq.addressable_shards:
        states_f.numpy[shard.index[0], :horizon] = np.asarray(shard.data, dtype=np.float32)
    for shard in final_obs.addressable_shards:
        states_f.numpy[shard.index[0], horizon] = np.asarray(shard.data, dtype=np.float32)

    alive_np = alive_f.numpy
    lengths = alive_np.sum(axis=1).astype(np.int32)
    raw_returns = (raw_f.numpy * alive_np).sum(axis=1)
    return ShardedRollout(
        states=states_f.view,
        actions=actions_f.view,
        rewards=rewards_f.view,
        raw_rewards=raw_f.view,
        log_likelihoods=logp_f.view,
        terminated=term_f.view,
        alive=alive_f.view,
        lengths=lengths,
        raw_returns=raw_returns,
        params_are_views=views,
        n_keep=n_keep,
        horizon=horizon,
        _keep=(states_f, actions_f, rewards_f, raw_f, logp_f, term_f, alive_f),
    )
