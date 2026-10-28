"""Annotating only the sampled transitions (mpo_sampled_annotation) against annotating the whole memory."""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

import abstract_rl.run.run_mpo_non_parametric as run_mpo
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from test_batched_operators import _trajectories
from test_torch_parity import build_mpo


def _zero_noise(policy):
    """The noise of sample_actions is 0, so both paths sample the same actions."""
    noise = policy.sample_noise
    policy.sample_noise = lambda n, k=1: mx.zeros_like(noise(n, k))


def test_sampled_rows_match_the_full_annotation(tmp_path):
    mc = build_mpo(tmp_path)
    policy = mc.get("policy", False)
    # the evaluated policy differs from the target policy (and the behaviour), so the trace coefficients vary
    policy.update(tree_map(lambda p: p * 1.05, policy.parameters()))
    _zero_noise(policy)
    _zero_noise(mc.get("policy", True))
    conf = mc["conf"]
    with conf.ns("res"):
        res = ResampleOperator(mc)
    with conf.ns("ret"):
        ret = RetraceOperator(mc)
    with conf.ns("rnk"):
        rnk = VariationalRankingOperator(mc)
    tc = TrajectoryCollection(mc.env)
    tc.extend(_trajectories(mc.env, np.random.default_rng(0)))

    res.transform_all(tc)
    rnk.hook(tc.annotation("add_q"))
    rnk.transform_all(tc)
    ret.transform_all(tc)
    full = {k: np.array(tc.annotation(k)) for k in ["add_act", "add_q", "add_pi", "grad_ranking", "q", "tq"]}

    # rows at the start, inside and at the end of trajectories, including ones of length 1 and 3
    rows = np.unique(np.random.default_rng(1).choice(tc.num_data_points, 300, replace=False))
    rows = np.union1d(rows, np.concatenate([[0], tc.trajectory_ends()[:-1], tc.trajectory_ends() - 1]))
    ann = res.annotate_states(tc.batch_at(rows).states)
    sampled = {**ann, "grad_ranking": rnk.b.log_weights(ann["add_q"])}
    sampled["q"], sampled["tq"] = ret.targets_at(tc, rows)
    for k, v in sampled.items():
        tol = 1e-3 if k == "add_pi" else 1e-5
        np.testing.assert_allclose(np.array(v), full[k][rows], rtol=tol, atol=tol, err_msg=k)
    # the targets use several steps
    assert not np.allclose(full["tq"][rows], full["q"][rows])


def test_sampled_epoch_cost_does_not_grow_with_the_memory(mpo_conf, monkeypatch):
    import abstract_rl.operator.resample_actions_operator as res_module
    import abstract_rl.operator.retrace_operator as ret_module

    # padded_size is called once per annotation with the number of evaluated rows
    rows: dict[str, list[int]] = {"res": [], "ret": []}
    for key, module in [("res", res_module), ("ret", ret_module)]:
        orig = module.padded_size
        monkeypatch.setattr(module, "padded_size", lambda n, key=key, orig=orig: rows[key].append(n) or orig(n))
    bootstrap = mpo_conf["mem_bootstrap_steps"]
    for size in [400, 2000]:
        for sampled in [False, True]:
            rows["res"].clear()
            rows["ret"].clear()
            # the memory is full from the first epoch
            conf = {**mpo_conf, "num_epochs": 2, "env_sample": size, "tc_size": size, "mpo_sampled_annotation": sampled}
            conf.update({"policy_opt_steps": 2, "policy_batch_size": 16, "val_opt_steps": 2, "val_batch_size": 16})
            run_mpo.run_mpo_non_parametric(conf)
            if not sampled:
                # every stored transition in every epoch
                assert rows == {"res": [size, size], "ret": [size, size]}
            else:
                # at most the drawn transitions, and the next steps of their traces
                assert len(rows["res"]) == len(rows["ret"]) == 2
                assert all(0 < n <= 2 * 16 for n in rows["res"])
                assert all(0 < n <= 2 * 16 * bootstrap for n in rows["ret"])
