import numpy as np

from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def _env_and_policy(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("policy"):
        return mc.env, UnivariateGaussianPolicy(mc)


def test_truncated_episode_is_not_stored_as_done(make_pendulum_mc):
    env, policy = _env_and_policy(make_pendulum_mc)
    traj = env.execute_policy_once(None, policy)
    assert len(traj) == 200  # Pendulum-v1 TimeLimit truncates after 200 steps
    assert not np.any(traj.dones)


def test_max_steps_cut_is_not_stored_as_done(make_pendulum_mc):
    env, policy = _env_and_policy(make_pendulum_mc)
    traj = env.execute_policy_once(10, policy)
    assert len(traj) == 10
    assert not np.any(traj.dones)
