import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tests run on cpu, also when the mps extra is installed

from jaxtyping import install_import_hook

# modules ported to JAX are checked against their jaxtyping annotations at runtime (beartype)
JAXTYPED_MODULES = [
    "abstract_rl.misc.typing",
    "abstract_rl.algorithms.mixed.conjugate_gradient",
    "abstract_rl.misc.flat_params",
    "abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon",
    "abstract_rl.data_structures.temporal_difference_data.batch",
    "abstract_rl.data_structures.abstract_conf.model_state",
    "abstract_rl.neural_modules.neural_network",
    "abstract_rl.policy.policy",
    "abstract_rl.policy.continuous.continuous_policy",
    "abstract_rl.policy.continuous.univariate_gaussian_policy",
    "abstract_rl.policy.continuous.diagonal_gaussian_policy",
    "abstract_rl.policy.continuous.select_policy",
    "abstract_rl.algorithms.mixed.vanilla_gradient",
    "abstract_rl.value_functions.fc_v_networks",
    "abstract_rl.value_functions.fc_q_networks",
    "abstract_rl.env.jax_envs",
    "abstract_rl.misc.running_centered_filter",
    "abstract_rl.misc.running_mean_std",
    "abstract_rl.data_structures.temporal_difference_data.trajectory",
    "abstract_rl.data_structures.temporal_difference_data.trajectory_collection",
    "abstract_rl.env.mcmc_env_wrapper",
    "abstract_rl.operator.trajectory_operator",
    "abstract_rl.operator.td_operator",
    "abstract_rl.operator.gae_operator",
    "abstract_rl.operator.filter_operator",
]
install_import_hook(JAXTYPED_MODULES, "beartype.beartype")

import copy
from typing import NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.env.jax_envs import register_env
from abstract_rl.scripts.mpo_pendulum import run_config as mpo_config
from abstract_rl.scripts.trpo_pendulum import run_config as trpo_config


class PointMass(gym.Env):
    """Two dimensional point mass which has to reach the origin; actions are accelerations with asymmetric bounds."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float64)
    action_space = gym.spaces.Box(np.array([-1.0, -2.0], np.float32), np.array([1.0, 2.0], np.float32))

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1, 1, 2)
        self.vel = np.zeros(2)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.pos, self.vel])

    def step(self, action):
        a = np.clip(action, self.action_space.low, self.action_space.high)
        self.vel = 0.9 * self.vel + 0.1 * a
        self.pos = self.pos + 0.1 * self.vel
        self.t += 1
        reward = -float(self.pos @ self.pos) - 0.01 * float(a @ a)
        return self._obs(), reward, False, self.t >= 30, {}


gym.register("TestPointMass-v0", entry_point=PointMass, max_episode_steps=30)


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


@pytest.fixture
def trpo_conf(tmp_path):
    """A tiny TRPO configuration on Pendulum-v1, writing into a temporary directory."""
    conf = copy.deepcopy(trpo_config)
    conf.update({"num_epochs": 2, "policy_batch_size": 400, "val_opt_steps": 5, "data_root": str(tmp_path)})
    return conf


@pytest.fixture
def mpo_conf(tmp_path):
    """A tiny MPO configuration on Pendulum-v1, writing into a temporary directory."""
    conf = copy.deepcopy(mpo_config)
    conf.update(
        {
            "num_epochs": 2,
            "env_sample": 400,
            "val_opt_steps": 5,
            "policy_opt_steps": 5,
            "data_root": str(tmp_path),
        }
    )
    return conf


@pytest.fixture
def make_pendulum_mc(trpo_conf):
    """Factory for a model configuration with a Pendulum-v1 RolloutEnvWrapper registered, built like run_trpo does.
    Keyword arguments override entries of the tiny TRPO configuration, e.g. ``policy_cov_type="state"``."""
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
    from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
    from abstract_rl.misc.data_logger import DataLogger

    def make(**overrides):
        conf = {**trpo_conf, "alg": "trpo", **overrides}
        mc = ModelConfiguration(conf)
        mc.add_main("log", DataLogger(mc.data_dir))
        with mc["conf"].ns("env"):
            mc.add_environment(RolloutEnvWrapper(mc))
        return mc

    return make
