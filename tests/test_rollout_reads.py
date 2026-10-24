"""The rollout reads the log likelihoods once per episode; they equal the per-step values of policy.act."""

from typing import Any

import mlx.core as mx
import numpy as np

from test_torch_parity import build_mpo


class RecordingPolicy:
    def __init__(self, policy: Any) -> None:
        self.policy, self.steps = policy, []

    def act(self, states: mx.array, opt: bool = False) -> tuple[mx.array, mx.array]:
        a, ll = self.policy.act(states, opt)
        self.steps.append((a, ll))
        return a, ll


def test_rollout_stores_the_per_step_actions_and_log_likelihoods(tmp_path):
    mc = build_mpo(tmp_path)
    policy = RecordingPolicy(mc.get("policy", target=False))
    mx.random.seed(3)
    tc = mc.env.execute_policy(policy, 30, 70)
    trajectories = tc.trajectories()
    assert [len(t) for t in trajectories] == [30, 30, 30]
    actions = np.concatenate([t.actions for t in trajectories])
    lls = np.concatenate([t.log_likelihoods for t in trajectories])
    assert len(policy.steps) == len(actions)
    for i, (a, ll) in enumerate(policy.steps):
        assert actions[i, 0] == np.float64(np.array(a).reshape(-1)[0])
        assert lls[i, 0] == float(ll.reshape(-1)[0].item())
