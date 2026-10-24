"""The flat replay on the device draws the same batches as the trajectory collection."""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from abstract_rl.policy.continuous.select_policy import select_policy


def _collection(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
    # several trajectories, so the drawn indices cross trajectory boundaries
    tc = mc.env.execute_policy(policy, 37, 400)
    rng = np.random.default_rng(3)
    for traj in tc:
        traj["q"] = rng.standard_normal((len(traj), 1))
        traj["add_act"] = rng.standard_normal((len(traj), 5, 1)).astype(np.float32)
    return mc, tc


def test_flat_replay_matches_list_sampling(make_pendulum_mc):
    mc, tc = _collection(make_pendulum_mc)
    fields = ["q", "add_act"]
    replay = tc.flatten(fields)
    assert replay.num_data_points == sum(len(t) for t in tc)
    # a batch size larger than the collection is clipped as well
    for batch_size in (1, 64, 250, 10_000):
        np.random.seed(batch_size)  # noqa: NPY002
        expected = [tc.sample(batch_size, fields).to_tree(fields) for _ in range(5)]
        np.random.seed(batch_size)  # noqa: NPY002
        got = [replay.sample(batch_size) for _ in range(5)]
        for e, g in zip(expected, got, strict=True):
            e_flat, g_flat = dict(tree_flatten(e)), dict(tree_flatten(g))
            assert e_flat.keys() == g_flat.keys()
            for k in e_flat:
                assert g_flat[k].dtype == e_flat[k].dtype == mx.float32
                assert np.array_equal(np.array(g_flat[k]), np.array(e_flat[k])), k
    mc.env.close()
