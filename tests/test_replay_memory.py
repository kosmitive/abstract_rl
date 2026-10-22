import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection


def _rollout(lengths_per_lane, insert_value, length):
    """A rollout collection whose lanes hold whole episodes of the given lengths; states carry an episode tag."""
    e = len(lengths_per_lane)
    states = np.zeros((e, length, 1), np.float32)
    ends = np.zeros((e, length), bool)
    mask = np.zeros((e, length), bool)
    time = np.zeros((e, length), np.int32)
    for lane, lengths in enumerate(lengths_per_lane):
        t = 0
        for k, n in enumerate(lengths):
            states[lane, t : t + n, 0] = insert_value * 1000 + lane * 100 + k
            ends[lane, t + n - 1] = True
            mask[lane, t : t + n] = True
            time[lane, t : t + n] = np.arange(t, t + n)
            t += n
    z = jnp.zeros((e, length, 1))
    traj = Trajectory(
        jnp.asarray(states),
        jnp.asarray(states),
        z,
        z,
        jnp.zeros((e, length)),
        z,
        z,
        jnp.asarray(ends),
        jnp.asarray(mask),
        {"time": jnp.asarray(time)},
    )
    return TrajectoryCollection(None, traj)


def _episodes(memory):
    t = jax.tree.map(lambda x: np.asarray(x[0]), memory.trajectories())
    tags = t.states[t.mask, 0]
    out, last = [], None
    for tag in tags:
        if tag != last:
            out.append(int(tag))
            last = tag
    return out


def _list_rule(collected, max_l):
    """torch TrajectoryCollection.extend: drop the oldest whole trajectories until the rest fits, keep the newest."""
    k, i = 0, len(collected)
    while i > 0 and k + collected[i - 1][1] <= max_l:
        i -= 1
        k += collected[i][1]
    return collected[min(i, len(collected) - 1) :]


def test_merge_capping_matches_the_list_rule():
    rng = np.random.default_rng(0)
    memory = TrajectoryCollection.memory(None, 400, 1, 1)
    collected = []
    for epoch in range(25):
        lanes = [list(rng.integers(1, 12, size=rng.integers(1, 4))) for _ in range(3)]
        rollout = _rollout(lanes, epoch, 40)
        memory = jax.jit(lambda m, r: m.merge(r, 60))(memory, rollout)
        # collection order: by the time the episode ended, ties by lane
        finished = []
        for lane, lengths in enumerate(lanes):
            t = 0
            for k, n in enumerate(lengths):
                t += n
                finished.append((t - 1, lane, epoch * 1000 + lane * 100 + k, n))
        collected += [(tag, n) for _, _, tag, n in sorted(finished)]
        collected = _list_rule(collected, 60)
        assert sorted(_episodes(memory)) == sorted(tag for tag, _ in collected)
        assert int(memory.trajectories().mask.sum()) == sum(n for _, n in collected)


def test_insert_never_splits_an_episode():
    memory = TrajectoryCollection.memory(None, 50, 1, 1)
    for epoch in range(8):
        memory = memory.merge(_rollout([[7, 7]], epoch, 20), 1000)
        t = jax.tree.map(lambda x: np.asarray(x[0]), memory.trajectories())
        valid = np.nonzero(t.mask)[0]
        # every valid run of rows that shares a tag is contiguous and ends with an end flag
        for tag in set(t.states[t.mask, 0].tolist()):
            rows = valid[t.states[valid, 0] == tag]
            assert np.all(np.diff(rows) == 1) and t.ends[rows[-1]]


def test_sampling_draws_only_kept_rows():
    memory = TrajectoryCollection.memory(None, 200, 1, 1)
    for epoch in range(6):
        memory = memory.merge(_rollout([[10, 10], [5]], epoch, 30), 40)
    batch = memory.sample(jax.random.key(0), 5000, [])
    kept = set(_episodes(memory))
    assert set(np.asarray(batch.states[:, 0]).astype(int).tolist()) <= kept


def test_with_annotations_adds_zero_rows():
    memory = TrajectoryCollection.memory(None, 10, 1, 1).with_annotations(
        {"add_q": jax.ShapeDtypeStruct((4,), jnp.float32)}
    )
    assert memory.trajectories()["add_q"].shape == (1, 10, 4)
    with pytest.raises(KeyError):
        memory.trajectories()["missing"]
