"""``obs_norm`` is Gymnasium's NormalizeObservation, reloaded frozen for eval."""

from os.path import exists, join

import numpy as np
import pytest
from gymnasium.wrappers import NormalizeObservation

from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.policy.continuous.select_policy import select_policy

# the torch policies / value nets build the ported (Equinox) MLP;
# re-enabled when their port lands (plan 1 tasks 8-10 / plan 2)
pytestmark = pytest.mark.skip(reason="porting to JAX: torch modules on the ported MLP")


def test_obs_norm_saves_training_stats_and_freezes_them(make_pendulum_mc):
    plain = make_pendulum_mc()
    assert not isinstance(plain.env.env, NormalizeObservation)
    plain.env.close()
    mc = make_pendulum_mc(obs_norm=True, env_name="TestPointMass-v0")
    env = mc.env
    assert isinstance(env.env, NormalizeObservation) and env.env.update_running_mean
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
    env.execute_policy(policy, 5, 8)
    path = join(mc.run_root, "model", "obs_rms.npz")
    assert exists(path)
    saved = np.load(path)
    with mc["conf"].ns("env"):
        frozen = RolloutEnvWrapper(mc)
    assert frozen.env.update_running_mean is False
    assert np.allclose(frozen.env.obs_rms.mean, saved["mean"])
    count = frozen.env.obs_rms.count
    frozen.reset()
    assert frozen.env.obs_rms.count == count
    raw = np.zeros(env.observation_dim, np.float32)
    got = frozen.env.observation(raw)
    expected = (raw - frozen.env.obs_rms.mean) / np.sqrt(frozen.env.obs_rms.var + frozen.env.epsilon)
    assert np.allclose(got, expected)
    env.close()
    frozen.close()
