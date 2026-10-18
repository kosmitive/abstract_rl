import copy

import gymnasium as gym
import numpy as np
import pytest

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


@pytest.fixture
def trpo_conf(tmp_path):
    """A tiny TRPO configuration on Pendulum-v1, writing into a temporary directory."""
    conf = copy.deepcopy(trpo_config)
    conf.update({"num_epochs": 2, "policy_batch_size": 400, "data_root": str(tmp_path)})
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
    """Factory for a model configuration with a Pendulum-v1 EnvRunner registered, built like run_trpo does.
    Keyword arguments override entries of the tiny TRPO configuration, e.g. ``policy_cov_type="state"``."""
    from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
    from abstract_rl.env.env_runner import EnvRunner
    from abstract_rl.misc.data_logger import DataLogger

    def make(**overrides):
        conf = {**trpo_conf, "alg": "trpo", **overrides}
        mc = ModelConfiguration(conf)
        mc.add_main("log", DataLogger(mc.data_dir))
        with mc["conf"].ns("env"):
            mc.add_environment(EnvRunner(mc))
        return mc

    return make
