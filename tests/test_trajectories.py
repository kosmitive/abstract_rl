import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


def _lanes():
    # lane 0: episodes of length 2 and 3, then padding; lane 1: one episode of length 4, then padding
    ends = jnp.array([[0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]], bool)
    mask = jnp.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], bool)
    e, ell = ends.shape
    raw = jnp.arange(e * ell, dtype=jnp.float32).reshape(e, ell)
    z = jnp.zeros((e, ell, 1))
    s = jnp.arange(e * ell * 3, dtype=jnp.float32).reshape(e, ell, 3)
    t = Trajectory(s, s + 1, jnp.zeros((e, ell, 2)), raw[..., None], raw, z, z, ends, mask, {"a": raw})
    return TrajectoryCollection(None, t)


def test_to_batch_flattens_lanes_and_masks_padding():
    tc = _lanes()
    b = tc.to_batch(["a"])
    assert b.states.shape == (12, 3) and b["a"].shape == (12,) and b.mask.shape == (12,)
    assert int(b.mask.sum()) == 9
    assert float(b["a"][5]) == 0.0  # the padded row is zeroed by Batch


def test_sample_only_draws_valid_transitions():
    tc = _lanes()
    b = jax.jit(lambda c, k: c.sample(k, 2000, ["a"]))(tc, jax.random.key(0))
    assert b.mask is None and b.states.shape == (2000, 3)
    drawn = set(np.asarray(b["a"]).astype(int).tolist())
    assert drawn == {0, 1, 2, 3, 4, 6, 7, 8, 9}


def test_episode_stats_sum_raw_rewards_per_episode():
    st = jax.jit(lambda c: c.episode_stats())(_lanes())
    valid = np.asarray(st["valid"])
    assert valid.sum() == 3
    np.testing.assert_allclose(sorted(np.asarray(st["returns"])[valid]), sorted([0 + 1, 2 + 3 + 4, 6 + 7 + 8 + 9]))
    np.testing.assert_allclose(sorted(np.asarray(st["lengths"])[valid]), [2, 3, 4])


def test_trajectory_set_returns_a_new_lane():
    t = _lanes().trajectories()
    t2 = t.set("tv", jnp.ones((2, 6)))
    assert "tv" in t2 and "tv" not in t
