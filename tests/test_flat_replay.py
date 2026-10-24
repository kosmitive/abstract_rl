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


def test_compiled_value_fit_matches_one_step_at_a_time(make_pendulum_mc, capsys):
    from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet

    mc, tc = _collection(make_pendulum_mc)
    for traj in tc:
        traj["tq"] = traj["q"] + 0.1
    with mc["conf"].ns("val"):
        fused, stepped = FullyQNeuralNet(mc), FullyQNeuralNet(mc)
    stepped.update(fused.parameters())
    capsys.readouterr()

    # 23 steps: two full compiled calls and a shorter one
    np.random.seed(0)  # noqa: NPY002
    fused.perform_mult_sgd(23, tc, 64, ["tq"])
    fused_log = capsys.readouterr().out
    np.random.seed(0)  # noqa: NPY002
    stepped.cli.print(stepped.titlestr % ("k", "pre_objective", "post_objective")).line()
    for s in range(23):
        stepped.perform_adam_step(s, tc.sample(64, ["tq"]), ["tq"])
    stepped_log = capsys.readouterr().out

    for (k, f), (_, s) in zip(tree_flatten(fused.parameters()), tree_flatten(stepped.parameters()), strict=True):
        assert np.array_equal(np.array(f), np.array(s)), k
    assert fused_log == stepped_log
    # the title and the steps 0, 10 and 20
    assert fused_log.count("|") == 4 * 2
    mc.env.close()
