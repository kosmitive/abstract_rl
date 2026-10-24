"""The annotation and Retrace over all trajectories at once against one call per trajectory (the earlier version)."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

import abstract_rl.operator.trajectory_operator as trajectory_operator
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.misc.mixed import set_device, to_mx
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.operator.trajectory_operator import materialize
from test_torch_parity import build_mpo

LENGTHS = [200, 37, 1, 3, 200, 120, 5, 64, 199]


def _trajectories(env, rng):
    out = []
    for i, n in enumerate(LENGTHS):
        b = TrajectoryBuilder(i, env, rng.standard_normal(3))
        for t in range(n):
            done = int(t == n - 1 and i % 3 == 1)
            b.observe(
                rng.uniform(-2, 2, 1),
                float(rng.standard_normal()),
                rng.standard_normal(3),
                float(rng.uniform(-3, 0)),
                done,
            )
        out.append(b.finalize())
    return out


def _old_annotation(op, trajectory):
    t_states = to_mx(trajectory.states)
    f_stats = op.policy.forward(t_states)
    sampled_actions = op.policy.sample_actions(suff_stats=f_stats, num_actions=op.add_act)
    trajectory["add_act"] = sampled_actions
    trajectory["add_q"] = op.q_net.q_val_multi(t_states, sampled_actions)
    trajectory["add_pi"] = op.policy.log_prob(sampled_actions, suff_stats=f_stats)


def _old_retrace(op, trajectory):
    tl = len(trajectory)
    states, actions = to_mx(trajectory.states), to_mx(trajectory.actions)
    q_vals = op.q_network.q_val(states, actions).reshape(-1)
    dones = to_mx(trajectory.dones).reshape(-1)
    nxt_v_vals = op.q_network.v_val(to_mx(trajectory.next_states), policy=op.policy).reshape(-1) * (1 - dones)
    pi_ll = op.policy.log_prob(actions, states=states)
    c = mx.minimum(mx.exp(pi_ll - to_mx(trajectory.log_likelihoods).reshape([-1, 1])), 1.0) * op.ret_lambda
    c = c.reshape(-1)
    td_errors_q = to_mx(trajectory.rewards).reshape(-1) + op.discount * nxt_v_vals - q_vals
    tq = q_vals + td_errors_q
    coeff = mx.ones(tl)
    for k in range(1, min(op.ret_bootstrap, tl)):
        coeff = coeff[: tl - k] * op.discount * c[k:]
        tq = tq + mx.concatenate([coeff * td_errors_q[k:], mx.zeros(k)])
    trajectory["q"] = q_vals.reshape(-1, 1)
    trajectory["tq"] = tq.reshape(-1, 1)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("chunk", [64, 4096])
def test_batched_annotation_and_retrace_match_per_trajectory(tmp_path, monkeypatch, chunk, device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("needs MLX with Metal (Apple silicon)")
    monkeypatch.setattr(trajectory_operator, "CHUNK_STATES", {"gpu": chunk, "cpu": chunk})
    set_device(device)
    try:
        _check(tmp_path)
    finally:
        set_device("cpu")


def _check(tmp_path):
    mc = build_mpo(tmp_path)
    # the evaluated policy differs from the target policy (and the behaviour), so the trace coefficients vary
    policy = mc.get("policy", False)
    policy.update(tree_map(lambda p: p * 1.05, policy.parameters()))
    rng = np.random.default_rng(0)
    conf = mc["conf"]
    with conf.ns("res"):
        res = ResampleOperator(mc)
    with conf.ns("ret"):
        ret = RetraceOperator(mc)
    old, new = _trajectories(mc.env, rng), _trajectories(mc.env, np.random.default_rng(0))

    for op, old_fn, keys in [(res, _old_annotation, ["add_act", "add_pi", "add_q"]), (ret, _old_retrace, ["q", "tq"])]:
        mx.random.seed(7)
        for t in old:
            old_fn(op, t)
        materialize(old)
        mx.random.seed(7)
        op.transform_all(new)
        for t_old, t_new in zip(old, new, strict=True):
            for k in keys:
                assert t_new[k].shape == t_old[k].shape, k
                # the log likelihood of squashed actions near the bound (arctanh) amplifies the rounding of tanh
                tol = 1e-3 if k == "add_pi" else 1e-5
                np.testing.assert_allclose(t_new[k], t_old[k], rtol=tol, atol=tol, err_msg=k)
    # the trace coefficients are not all 1 and some targets use several steps
    assert any(not np.allclose(t["tq"], t["q"]) for t in new)
