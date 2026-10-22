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
        num_envs: int | None = None,
        update_stats: bool = True,
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
        :return: The new state, the trajectory collection and the rollout metrics.
        """
        e = num_envs if num_envs is not None else self.num_envs(batch_size, max_steps)
        length = -(-batch_size // e) + max_steps
        sd, ad = self.observation_dim, self.action_dim
        obs_rms, state_filter, reward_filter = (
            state.get("obs_rms"),
            state.get("state_filter"),
            state.get("reward_filter"),
        )

        key, k_reset = jax.random.split(key)
        env_state, raw = jax.vmap(self.env.reset)(jax.random.split(k_reset, e))
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

            stepped, raw_nxt, r, terminated, truncated = jax.vmap(self.env.step)(
                env_state, a, jax.random.split(k_step, e)
            )
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
            reset_state, reset_raw = jax.vmap(self.env.reset)(jax.random.split(k_reset, e))
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
            annotations={"time": buf["time"]},
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
        self,
        state: ModelState,
        key: PRNGKeyArray,
        max_steps: int,
        policy: Any,
        opt: bool = False,
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
