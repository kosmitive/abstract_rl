import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.operator.filter_operator import FilterOperator


def _setup(make_pendulum_mc, **conf):
    mc = make_pendulum_mc(**conf)
    with mc["conf"].ns("filter"):
        op = FilterOperator(mc)
    rng = np.random.default_rng(0)
    states = jnp.asarray(rng.normal(2.0, 3.0, size=(8, 3)), jnp.float32)
    next_states = jnp.asarray(rng.normal(2.0, 3.0, size=(8, 3)), jnp.float32)
    rewards = jnp.asarray(rng.normal(-5.0, 2.0, size=(8, 1)), jnp.float32)
    state_filter = RunningCenteredFilter("states", 3).register(jnp.concatenate([states, next_states]))
    reward_filter = RunningCenteredFilter("rewards", 1).register(rewards)
    state = ModelState.create({"state_filter": state_filter, "reward_filter": reward_filter}, {})
    z = jnp.zeros((8, 1))
    ones = jnp.ones(8, bool)
    traj = Trajectory(states, next_states, z, rewards, rewards[:, 0], z, z, jnp.zeros(8, bool), ones)
    return op, state, traj, state_filter, reward_filter


def test_states_next_states_and_rewards_are_filtered(make_pendulum_mc):
    op, state, traj, state_filter, reward_filter = _setup(make_pendulum_mc, filter_states=True, filter_rewards=True)
    out = op.transform(state, traj)
    np.testing.assert_allclose(out.states, state_filter(traj.states), rtol=1e-6)
    np.testing.assert_allclose(out.next_states, state_filter(traj.next_states), rtol=1e-6)
    np.testing.assert_allclose(out.rewards, reward_filter(traj.rewards), rtol=1e-6)
    assert not np.allclose(out.states, traj.states) and not np.allclose(out.rewards, traj.rewards)
    np.testing.assert_array_equal(out["unfiltered_states"], traj.states)
    np.testing.assert_array_equal(out["unfiltered_rewards"], traj.rewards)


def test_a_second_transform_does_not_filter_again(make_pendulum_mc):
    op, state, traj, _, _ = _setup(make_pendulum_mc, filter_states=True, filter_rewards=True)
    once = op.transform(state, traj)
    twice = op.transform(state, once)
    np.testing.assert_array_equal(twice.states, once.states)
    np.testing.assert_array_equal(twice.next_states, once.next_states)
    np.testing.assert_array_equal(twice.rewards, once.rewards)
    np.testing.assert_array_equal(twice["unfiltered_states"], traj.states)
    np.testing.assert_array_equal(twice["unfiltered_rewards"], traj.rewards)


def test_disabled_filters_leave_the_trajectory_alone(make_pendulum_mc):
    op, state, traj, _, _ = _setup(make_pendulum_mc, filter_states=False, filter_rewards=False)
    out = op.transform(state, traj)
    np.testing.assert_array_equal(out.states, traj.states)
    assert "unfiltered_states" not in out and "unfiltered_rewards" not in out
