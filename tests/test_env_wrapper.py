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


def test_vector_collection_finishes_episodes_and_keeps_truncation(make_pendulum_mc):
    mc = make_pendulum_mc(env_vector_envs=2)
    env, policy = mc.env, None
    with mc["conf"].ns("policy"):
        policy = UnivariateGaussianPolicy(mc)
    collection = env.execute_policy(policy, 50, 120, exploration=True)
    assert sum(len(traj) for traj in collection) >= 120
    assert all(len(traj) <= 50 for traj in collection)
    assert all(not np.any(traj.dones) for traj in collection)
