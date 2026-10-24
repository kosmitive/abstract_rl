"""The flat replay on the device draws the same batches as the trajectory collection."""

import mlx.core as mx
import numpy as np
import pytest
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


def test_draw_batches_matches_draw(make_pendulum_mc):
    _, tc = _collection(make_pendulum_mc)
    replay = tc.flatten([])
    for batch_size in (1, 64, 10_000):
        np.random.seed(7)  # noqa: NPY002
        expected = np.stack([replay.draw(batch_size) for _ in range(30)])
        after = np.random.rand()  # noqa: NPY002
        np.random.seed(7)  # noqa: NPY002
        drawn = replay.draw_batches(30, batch_size)
        assert np.random.rand() == after  # noqa: NPY002
        np.testing.assert_array_equal(drawn, expected)


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


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_background_fit_matches_main_process(make_pendulum_mc, capsys, device):
    """A q fit in a worker process (BackgroundFit) is bit identical to the fit in the main process."""
    from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet

    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("no Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mc, tc = _collection(make_pendulum_mc)
    for traj in tc:
        traj["tq"] = traj["q"] + 0.1
    with mc["conf"].ns("val"):
        main, worker = FullyQNeuralNet(mc), FullyQNeuralNet(mc)
    worker.update(main.parameters())
    worker.fit_in_background()
    capsys.readouterr()

    # two fits, the second one continues with the moments of Adam of the first one
    for fit in range(2):
        np.random.seed(fit)  # noqa: NPY002
        main.perform_mult_sgd(23, tc, 64, ["tq"])
        main_log = capsys.readouterr().out
        np.random.seed(fit)  # noqa: NPY002
        worker.start_mult_sgd(23, tc, 64, ["tq"])
        worker.finish_mult_sgd()
        worker_log = capsys.readouterr().out
        for (k, a), (_, b) in zip(tree_flatten(main.parameters()), tree_flatten(worker.parameters()), strict=True):
            assert isinstance(b, mx.array)
            assert np.array_equal(np.array(a), np.array(b)), (fit, k)
        assert main_log == worker_log
        assert main_log.count("|") == 4 * 2
    worker.close()
    mc.env.close()
    mx.set_default_device(previous)
