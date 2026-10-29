from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.host_gym import HostGymEnv
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


def _gym_seed(key: PRNGKeyArray) -> int:
    """A non-negative Gymnasium reset seed drawn from ``key``."""
    return int(jax.device_get(jax.random.bits(key))) & 0x7FFFFFFF


@jax.jit
def _update_rms(rms: RunningMeanStd, xs: Array, mask: Array) -> RunningMeanStd:
    return rms.update(xs, mask)


@jax.jit
def _update_filter(filt: RunningCenteredFilter, xs: Array, mask: Array) -> RunningCenteredFilter:
    return filt.register(xs, mask)


@eqx.filter_jit
def _host_sample(policy: Any, obs: Array, key: PRNGKeyArray) -> tuple[Array, Array]:
    """Sample one action per row and its log likelihood."""
    stats = policy.forward(obs)
    actions = policy.sample_actions(key, suff_stats=stats).reshape(obs.shape[0], -1)
    return actions, policy.log_prob(actions, suff_stats=stats).reshape(obs.shape[0])


@eqx.filter_jit
def _host_mode(policy: Any, obs: Array) -> tuple[Array, Array]:
    """The mode of one action per row and its log likelihood."""
    stats = policy.forward(obs)
    actions = policy.mode(suff_stats=stats).reshape(obs.shape[0], -1)
    return actions, policy.log_prob(actions, suff_stats=stats).reshape(obs.shape[0])


class RolloutEnvWrapper:
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
        self._lanes: list[HostGymEnv] | None = None
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

    @property
    def uses_host(self) -> bool:
        """True when the environment is a Gymnasium env stepped outside ``jax.jit`` (Humanoid-v5)."""
        return isinstance(self.env, HostGymEnv)

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
        num_envs: int | None = None,
        update_stats: bool = True,
        record: bool = False,
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
        :param num_envs: The number of lanes E, or None for ``self.num_envs(batch_size, max_steps)``.
        :param update_stats: Update the observation statistics and filters if True, otherwise freeze them.
        :param record: Annotate the raw next observations ``raw_next_states`` and, for MuJoCo environments, the joint
            positions ``qpos`` after each step (evaluation and rendering only, training rollouts do not carry them).
        :return: The new state, the trajectory collection and the rollout metrics.
        """
        env = self.env
        if isinstance(env, HostGymEnv):
            raise RuntimeError(
                f"{self.env_name} is stepped on the host with Gymnasium; call execute_policy_host "
                "(the jitted rollout only runs JAX environments)"
            )
        e = num_envs if num_envs is not None else self.num_envs(batch_size, max_steps)
        length = -(-batch_size // e) + max_steps
        sd, ad = self.observation_dim, self.action_dim
        obs_rms, state_filter, reward_filter = (
            state.get("obs_rms"),
            state.get("state_filter"),
            state.get("reward_filter"),
        )

        key, k_reset = jax.random.split(key)
        env_state, raw = jax.vmap(env.reset)(jax.random.split(k_reset, e))
        active = jnp.ones(e, bool)
        if self._obs_norm and update_stats:
            obs_rms = obs_rms.update(raw, active)
        cs = self._preprocess(raw, obs_rms)
        if update_stats:
            state_filter = state_filter.register(cs, active)

        def zeros(*shape: int, dtype: Any = DTYPE) -> Array:
            return jnp.zeros((e, length, *shape), dtype)

        buf = {
            "states": zeros(sd),
            "next_states": zeros(sd),
            "actions": zeros(ad),
            "rewards": zeros(1),
            "raw_rewards": zeros(),
            "dones": zeros(1),
            "log_likelihoods": zeros(1),
            "ends": zeros(dtype=bool),
            "mask": zeros(dtype=bool),
            "time": zeros(dtype=jnp.int32),
        }
        # static check: only MuJoCo adapters expose the model and the joint positions of their states
        record_qpos = record and hasattr(env, "mj_model")
        if record:
            buf["raw_next_states"] = zeros(sd)
        if record_qpos:
            buf["qpos"] = zeros(env.qpos(env_state).shape[-1])  # type: ignore[attr-defined]
        carry = (
            jnp.asarray(0),
            key,
            env_state,
            cs,
            jnp.zeros(e, jnp.int32),
            active,
            jnp.asarray(0),
            buf,
            obs_rms,
            state_filter,
            reward_filter,
        )

        def cond(c: Any) -> Array:
            i, _, _, _, _, active, *_ = c
            return (i < length) & jnp.any(active)

        def body(c: Any) -> Any:
            i, key, env_state, cs, t, active, count, buf, obs_rms, state_filter, reward_filter = c
            key, k_act, k_step, k_reset = jax.random.split(key, 4)

            # sample action and calc log likelihood
            suff_stats = policy.forward(cs)
            a = (
                policy.mode(suff_stats=suff_stats)
                if not exploration
                else policy.sample_actions(k_act, suff_stats=suff_stats)
            )
            a = a.reshape(e, ad)
            ll = policy.log_prob(a, suff_stats=suff_stats).reshape(e)

            stepped, raw_nxt, r, terminated, truncated = jax.vmap(env.step)(env_state, a, jax.random.split(k_step, e))
            if self._obs_norm and update_stats:
                obs_rms = obs_rms.update(raw_nxt, active)
            nxt = self._preprocess(raw_nxt, obs_rms)
            if update_stats:
                state_filter = state_filter.register(nxt, active)
            r_train = self._reward_scale * r + self._reward_offset
            if update_stats:
                reward_filter = reward_filter.register(r_train[:, None], active)

            t = t + 1
            # only true terminations are stored as done, so truncated episodes can be bootstrapped
            end = terminated | truncated | (t >= max_steps)
            step_vals = {
                "states": cs,
                "next_states": nxt,
                "actions": a,
                "rewards": r_train[:, None],
                "raw_rewards": r,
                "dones": terminated.astype(DTYPE)[:, None],
                "log_likelihoods": ll[:, None],
                "ends": end,
                "mask": active,
                "time": jnp.full(e, i, jnp.int32),
            }
            if record:
                step_vals["raw_next_states"] = jnp.asarray(raw_nxt, DTYPE)
            if record_qpos:
                step_vals["qpos"] = jnp.asarray(env.qpos(stepped), DTYPE)  # type: ignore[attr-defined]
            # index i is written once, so selecting on the slice instead of the whole buffer gives identical outputs
            buf = {
                k: buf[k].at[:, i].set(jnp.where(active.reshape(active.shape + (1,) * (v.ndim - 1)), v, buf[k][:, i]))
                for k, v in step_vals.items()
            }
            count = count + jnp.sum(active)

            # finished lanes start a new episode while the batch is not complete, otherwise they stop
            finished = active & end
            cont = finished & (count < batch_size)
            active = active & ~(finished & (count >= batch_size))
            reset_state, reset_raw = jax.vmap(env.reset)(jax.random.split(k_reset, e))
            if self._obs_norm and update_stats:
                obs_rms = obs_rms.update(reset_raw, cont)
            reset_cs = self._preprocess(reset_raw, obs_rms)
            if update_stats:
                state_filter = state_filter.register(reset_cs, cont)
            env_state = _lane_where(cont, reset_state, _lane_where(active, stepped, env_state))
            cs = jnp.where(cont[:, None], reset_cs, nxt)
            t = jnp.where(cont, 0, t)
            return (i + 1, key, env_state, cs, t, active, count, buf, obs_rms, state_filter, reward_filter)

        _, _, _, _, _, _, count, buf, obs_rms, state_filter, reward_filter = jax.lax.while_loop(cond, body, carry)
        trajectories = Trajectory(
            buf["states"],
            buf["next_states"],
            buf["actions"],
            buf["rewards"],
            buf["raw_rewards"],
            buf["dones"],
            buf["log_likelihoods"],
            buf["ends"],
            buf["mask"],
            annotations={k: buf[k] for k in ("time", "raw_next_states", "qpos") if k in buf},
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

    def execute_policy_host(
        self,
        state: ModelState,
        key: PRNGKeyArray,
        policy: Any,
        max_steps: int,
        batch_size: int,
        exploration: bool = True,
        num_envs: int | None = None,
        update_stats: bool = True,
        record: bool = False,
    ) -> tuple[ModelState, TrajectoryCollection, dict[str, Array]]:
        """Roll out a host Gymnasium environment with the same lane rules as ``execute_policy``.

        Actions and log likelihoods come from the JAX policy. The environment itself steps on the CPU. Observation
        statistics and the running filters are updated with the same masked updates as the jitted rollout. Lanes are
        reused across calls.

        Args:
            state: Traced model state of one seed (statistics are read and updated).
            key: Random key of resets and action samples.
            policy: Policy of this seed.
            max_steps: Maximum steps of one episode.
            batch_size: Minimum number of transitions to collect.
            exploration: Sample from the policy if True, otherwise use its mode.
            num_envs: Lane count, or None for ``self.num_envs(batch_size, max_steps)``.
            update_stats: Update observation statistics and filters if True.
            record: Store raw next observations and, for MuJoCo, joint positions.

        Returns:
            The new state, the trajectory collection and the rollout metrics.
        """
        env = self.env
        if not isinstance(env, HostGymEnv):
            raise RuntimeError(f"{self.env_name} is a JAX environment; call execute_policy")
        e = num_envs if num_envs is not None else self.num_envs(batch_size, max_steps)
        length = -(-batch_size // e) + max_steps
        lanes = self._lane_envs(e)
        sd, ad = self.observation_dim, self.action_dim
        obs_rms, state_filter, reward_filter = (
            state.get("obs_rms"),
            state.get("state_filter"),
            state.get("reward_filter"),
        )
        low, high = env.action_low, env.action_high

        def write(buf: np.ndarray, value: np.ndarray, active_rows: np.ndarray) -> None:
            mask = active_rows.reshape(active_rows.shape + (1,) * (value.ndim - 1))
            buf[:, i] = np.where(mask, value, buf[:, i])

        key, k_reset = jax.random.split(key)
        reset_keys = jax.random.split(k_reset, e)
        raw = np.stack([lanes[lane].reset_obs(_gym_seed(reset_keys[lane])) for lane in range(e)])
        active = np.ones(e, dtype=bool)
        if self._obs_norm and update_stats:
            obs_rms = _update_rms(obs_rms, jnp.asarray(raw), jnp.asarray(active))
        cs = np.asarray(jax.device_get(self._preprocess(jnp.asarray(raw), obs_rms)))
        if update_stats:
            state_filter = _update_filter(state_filter, jnp.asarray(cs), jnp.asarray(active))

        def zeros(*shape: int, dtype: Any = np.float32) -> np.ndarray:
            return np.zeros((e, length, *shape), dtype)

        buf = {
            "states": zeros(sd),
            "next_states": zeros(sd),
            "actions": zeros(ad),
            "rewards": zeros(1),
            "raw_rewards": zeros(),
            "dones": zeros(1),
            "log_likelihoods": zeros(1),
            "ends": zeros(dtype=bool),
            "mask": zeros(dtype=bool),
            "time": zeros(dtype=np.int32),
        }
        record_qpos = record and hasattr(env, "mj_model")
        if record:
            buf["raw_next_states"] = zeros(sd)
        if record_qpos:
            buf["qpos"] = zeros(int(env.mj_model.nq))
        t = np.zeros(e, np.int32)
        count = 0
        i = 0
        while i < length and bool(active.any()):
            key, k_act, k_reset = jax.random.split(key, 3)
            obs = jnp.asarray(cs, DTYPE)
            actions_j, ll_j = _host_sample(policy, obs, k_act) if exploration else _host_mode(policy, obs)
            actions = np.clip(np.asarray(jax.device_get(actions_j), np.float32), low, high)
            ll = np.asarray(jax.device_get(ll_j), np.float32)

            raw_nxt = np.zeros((e, sd), np.float32)
            rewards = np.zeros(e, np.float32)
            terminated = np.zeros(e, dtype=bool)
            truncated = np.zeros(e, dtype=bool)
            qpos = np.zeros((e, buf["qpos"].shape[-1]), np.float32) if record_qpos else None
            for lane in range(e):
                if not active[lane]:
                    continue
                raw_nxt[lane], rewards[lane], terminated[lane], truncated[lane] = lanes[lane].step_obs(actions[lane])
                if qpos is not None:
                    qpos[lane] = lanes[lane].qpos()

            active_j = jnp.asarray(active)
            if self._obs_norm and update_stats:
                obs_rms = _update_rms(obs_rms, jnp.asarray(raw_nxt), active_j)
            nxt = np.asarray(jax.device_get(self._preprocess(jnp.asarray(raw_nxt), obs_rms)))
            r_train = self._reward_scale * rewards + self._reward_offset
            if update_stats:
                state_filter = _update_filter(state_filter, jnp.asarray(nxt), active_j)
                reward_filter = _update_filter(reward_filter, jnp.asarray(r_train[:, None]), active_j)

            t = t + 1
            end = terminated | truncated | (t >= max_steps)
            write(buf["states"], cs, active)
            write(buf["next_states"], nxt, active)
            write(buf["actions"], actions, active)
            write(buf["rewards"], r_train[:, None], active)
            write(buf["raw_rewards"], rewards, active)
            write(buf["dones"], terminated.astype(np.float32)[:, None], active)
            write(buf["log_likelihoods"], ll[:, None], active)
            write(buf["ends"], end, active)
            write(buf["mask"], active, active)
            write(buf["time"], np.full(e, i, np.int32), active)
            if record:
                write(buf["raw_next_states"], raw_nxt, active)
            if qpos is not None:
                write(buf["qpos"], qpos, active)
            count += int(active.sum())

            finished = active & end
            cont = finished & (count < batch_size)
            active = active & ~(finished & (count >= batch_size))
            if bool(cont.any()):
                reset_keys = jax.random.split(k_reset, e)
                reset_raw = np.zeros_like(raw_nxt)
                for lane in range(e):
                    if cont[lane]:
                        reset_raw[lane] = lanes[lane].reset_obs(_gym_seed(reset_keys[lane]))
                cont_j = jnp.asarray(cont)
                if self._obs_norm and update_stats:
                    obs_rms = _update_rms(obs_rms, jnp.asarray(reset_raw), cont_j)
                reset_cs = np.asarray(jax.device_get(self._preprocess(jnp.asarray(reset_raw), obs_rms)))
                if update_stats:
                    state_filter = _update_filter(state_filter, jnp.asarray(reset_cs), cont_j)
                cs = np.where(cont[:, None], reset_cs, nxt)
            else:
                cs = nxt
            t = np.where(cont, 0, t)
            i += 1

        def as_jax(name: str, dtype: Any) -> Array:
            return jnp.asarray(buf[name], dtype)

        annotations: dict[str, Array] = {"time": as_jax("time", jnp.int32)}
        if record:
            annotations["raw_next_states"] = as_jax("raw_next_states", DTYPE)
        if record_qpos:
            annotations["qpos"] = as_jax("qpos", DTYPE)
        trajectories = Trajectory(
            as_jax("states", DTYPE),
            as_jax("next_states", DTYPE),
            as_jax("actions", DTYPE),
            as_jax("rewards", DTYPE),
            as_jax("raw_rewards", DTYPE),
            as_jax("dones", DTYPE),
            as_jax("log_likelihoods", DTYPE),
            as_jax("ends", bool),
            as_jax("mask", bool),
            annotations=annotations,
        )
        tc = TrajectoryCollection(self, trajectories)
        stats = tc.episode_stats()
        metrics = {
            "steps": jnp.asarray(count, jnp.int32),
            "episode_returns": stats["returns"],
            "episode_valid": stats["valid"],
            "episode_lengths": stats["lengths"],
            "max_traj_len": jnp.max(jnp.where(stats["valid"], stats["lengths"], 0)),
        }
        state = state.set("obs_rms", obs_rms).set("state_filter", state_filter).set("reward_filter", reward_filter)
        return state, tc, metrics

    def _lane_envs(self, n: int) -> list[HostGymEnv]:
        """``n`` Gymnasium envs, the wrapper's own env first, kept for the next rollout."""
        lanes = self._lanes
        if lanes is not None and len(lanes) == n:
            return lanes
        if lanes is not None:
            for lane in lanes[1:]:
                lane.close()
        env = self.env
        if not isinstance(env, HostGymEnv):
            raise RuntimeError(f"{self.env_name} has no host environment")
        self._lanes = [env, *[HostGymEnv(self.env_name) for _ in range(n - 1)]]
        return self._lanes

    def execute_policy_once(
        self,
        state: ModelState,
        key: PRNGKeyArray,
        max_steps: int,
        policy: Any,
        opt: bool = False,
        update_stats: bool = True,
        record: bool = False,
    ) -> tuple[ModelState, Trajectory]:
        """
        Execute a policy once for the maximum number of steps or until the environment terminates.
        :return: The new state and the single-lane trajectory of the episode.
        """
        rollout = self.execute_policy_host if self.uses_host else self.execute_policy
        state, tc, _ = rollout(
            state, key, policy, max_steps, 1, exploration=not opt, num_envs=1, update_stats=update_stats, record=record
        )
        return state, jax.tree.map(lambda x: x[0], tc.trajectories())

    def close(self) -> None:
        """Close host Gymnasium environments. JAX environments hold no resources."""
        lanes = self._lanes
        if lanes is None:
            env = self.env
            if isinstance(env, HostGymEnv):
                env.close()
            return
        for lane in lanes:
            lane.close()
        self._lanes = None

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
