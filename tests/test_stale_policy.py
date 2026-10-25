"""The opt-in stale-policy rollout is one update behind; the default epoch is unchanged."""

import copy
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.policy_snapshot import (
    BehaviorSnapshot,
    PendingRollout,
    behavior_snapshot,
    fork_random_key,
    read_key,
    write_key,
)
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.scripts.hydra_conf import load, run_dict
from abstract_rl.scripts.mpo_pendulum import run_config as mpo_config

# Two epochs of TestPointMass MPO captured on this commit's parent with the flag absent (CPU, seed 1). The CPU
# kernels of MLX round differently on Linux and macOS, so each platform has its own bit-exact reference.
_GOLDENS = {
    "linux": [[-28.301695227131603, 17.43803590418358], [-13.245070878901272, 5.277140340529596]],
    "darwin": [[-28.301695621975195, 17.438036014413395], [-13.245070751384018, 5.277140248044079]],
}
_GOLDEN = np.array(_GOLDENS.get(sys.platform, _GOLDENS["linux"]), dtype=np.float64)


def _fingerprint(module: object) -> np.ndarray:
    flat = tree_flatten(module.parameters())  # type: ignore[attr-defined]
    return np.concatenate([np.array(value).reshape(-1) for _name, value in flat])


def _conf(tmp_path, **overrides):
    conf = copy.deepcopy(mpo_config)
    conf.update(
        {
            "env_name": "TestPointMass-v0",
            "num_epochs": 2,
            "max_steps": 30,
            "env_sample": 90,
            "val_opt_steps": 2,
            "policy_opt_steps": 3,
            "rnk_max_steps": 8,
            "data_root": str(tmp_path),
            "seed": 1,
            "device": "cpu",
            "val_background_fit": False,
            "stale_policy_rollout": False,
        }
    )
    conf.update(overrides)
    return conf


def test_hydra_stale_policy_rollout_defaults_off():
    halfcheetah = run_dict(load(["env=halfcheetah", "alg=mpo"]))
    assert halfcheetah["stale_policy_rollout"] is False
    enabled = run_dict(load(["env=halfcheetah", "alg=mpo", "alg.stale_policy_rollout=true"]))
    assert enabled["stale_policy_rollout"] is True


def test_fork_splits_the_global_key_and_keeps_the_subkey():
    mx.random.seed(123)
    before = np.array(mx.random.state[0]).copy()
    sub = fork_random_key()
    new, sub_mx = mx.random.split(mx.array(before))
    assert np.array_equal(np.array(mx.random.state[0]), np.array(new))
    assert sub == tuple(int(v) for v in np.array(sub_mx))
    write_key(read_key())
    assert np.array_equal(np.array(mx.random.state[0]), np.array(new))


def test_snapshot_is_isolated_from_later_updates(make_pendulum_mc):
    mx.random.seed(7)
    np.random.seed(7)  # noqa: NPY002
    mc = make_pendulum_mc(
        env_name="TestPointMass-v0",
        policy_structure=[16, 16],
        policy_act_fn=["tanh", "tanh", "none"],
        policy_squash=True,
    )
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
    before_key = np.array(mx.random.state[0]).copy()
    before = _fingerprint(policy)
    np.random.seed(1)  # noqa: NPY002
    snap = behavior_snapshot(policy, 16)
    frozen_weights = []
    if snap.host is not None:
        frozen_weights = [(weight.copy(), bias.copy()) for weight, bias in snap.host.weights]
    drawn = np.random.rand()  # noqa: NPY002
    np.random.seed(1)  # noqa: NPY002
    assert np.random.rand() == drawn  # noqa: NPY002
    new, sub = mx.random.split(mx.array(before_key))
    assert np.array_equal(np.array(mx.random.state[0]), np.array(new))
    assert snap.rollout_key == tuple(int(v) for v in np.array(sub))
    assert snap.overlaps_learner() == (snap.host is not None)

    policy.update(tree_map(lambda p: p + 1, policy.parameters()))
    assert not np.array_equal(_fingerprint(policy), before)
    if snap.clone is not None:
        assert np.array_equal(_fingerprint(snap.clone), before)
        state = mx.array(np.zeros((1, mc.env.observation_dim), np.float32))
        write_key(snap.rollout_key)
        stale, _ll = snap.clone.act(state, False)
        write_key(snap.rollout_key)
        live, _ll = policy.act(state, False)
        assert not np.array_equal(np.array(stale), np.array(live))
    else:
        assert snap.host is not None
        for (weight, bias), (weight0, bias0) in zip(snap.host.weights, frozen_weights, strict=True):
            assert np.array_equal(weight, weight0)
            assert np.array_equal(bias, bias0)


def test_default_path_matches_the_synchronous_epoch(tmp_path, monkeypatch):
    import abstract_rl.run.run_mpo_non_parametric as mod

    def boom(*_args, **_kwargs):
        raise AssertionError("the default path must not snapshot the policy")

    # one epoch never has a next rollout, and the flag-off loop never schedules one
    monkeypatch.setattr(mod, "behavior_snapshot", boom)
    one = run_mpo_non_parametric(_conf(tmp_path / "off1", num_epochs=1))
    flagged = run_mpo_non_parametric(_conf(tmp_path / "on1", num_epochs=1, stale_policy_rollout=True))
    two = run_mpo_non_parametric(_conf(tmp_path / "off2"))
    assert np.array_equal(one, _GOLDEN[:1])
    assert np.array_equal(flagged, one)
    assert np.array_equal(two, _GOLDEN)


def test_host_rollout_runs_on_a_thread_until_join(make_pendulum_mc):
    """A NumPy snapshot is sampled off the main thread; log likelihoods are filled after join."""
    import threading

    mc = make_pendulum_mc(env_name="TestPointMass-v0")
    mc["log"].create_field("sreward", 2)
    mc["log"].create_field("steps", 1)
    started = threading.Event()
    release = threading.Event()

    class _Actor:
        def __call__(self, state: np.ndarray) -> np.ndarray:
            started.set()
            assert release.wait(timeout=2)
            return np.zeros(mc.env.action_dim, np.float64)

    class _Host:
        def __init__(self) -> None:
            self.n = 0
            self.weights: list[tuple[np.ndarray, np.ndarray]] = []

        def spawn(self) -> _Actor:
            self.n += 1
            return _Actor()

        def materialize(self, trajectories: list[object]) -> None:
            assert self.n == len(trajectories)
            for traj in trajectories:
                traj._log_likelihoods[:] = 0.25  # type: ignore[attr-defined]

    snap = BehaviorSnapshot(_Host(), None, (1, 2))
    pending = PendingRollout(mc.env, object(), snap, 4, 4, False, "sreward")
    assert started.wait(timeout=2)
    release.set()
    collection = pending.join()
    assert snap.host is not None and collection.trajectories()
    assert np.all(collection.trajectories()[0].log_likelihoods == 0.25)


def test_stale_rollout_uses_the_policy_from_before_the_m_step(tmp_path, monkeypatch):
    recorded: list[dict[str, object]] = []
    original = RolloutEnvWrapper.execute_policy

    def record(self, policy, *args, **kwargs):
        live = self.mc.get("policy", target=False)
        frozen = kwargs.get("frozen")
        recorded.append(
            {
                "frozen": None if frozen is None else [(weight.copy(), bias.copy()) for weight, bias in frozen.weights],
                "policy": _fingerprint(policy),
                "live": _fingerprint(live),
                "layers": [
                    (np.array(layer.weight).copy(), np.array(layer.bias).copy()) for layer in live.network.layers
                ],
            }
        )
        return original(self, policy, *args, **kwargs)

    monkeypatch.setattr(RolloutEnvWrapper, "execute_policy", record)
    returns = run_mpo_non_parametric(_conf(tmp_path, stale_policy_rollout=True))
    assert returns.shape == (2, 2) and np.all(np.isfinite(returns))
    assert np.array_equal(returns[0], _GOLDEN[0])
    assert not np.array_equal(returns, _GOLDEN)
    assert len(recorded) == 2
    assert recorded[0]["frozen"] is None
    assert np.array_equal(recorded[0]["policy"], recorded[0]["live"])
    frozen = recorded[1]["frozen"]
    if frozen is None:
        # cloned module: rolled out at the join, after the M-step changed the live policy
        assert not np.array_equal(recorded[1]["live"], recorded[0]["live"])
        # the rollout argument is the snapshot, not the live policy
        assert np.array_equal(recorded[1]["policy"], recorded[0]["policy"])
        assert not np.array_equal(recorded[1]["policy"], recorded[1]["live"])
    else:
        layers = recorded[0]["layers"]
        assert isinstance(layers, list)
        for (weight, bias), (weight0, bias0) in zip(frozen, layers, strict=True):
            assert np.array_equal(weight, weight0)
            assert np.array_equal(bias, bias0)


@pytest.mark.skipif(not mx.metal.is_available(), reason="the host snapshot is built on the CPU stream of a Mac")
def test_host_snapshot_matches_act_and_ignores_later_updates(make_pendulum_mc):
    mx.random.seed(4)
    np.random.seed(4)  # noqa: NPY002
    mc = make_pendulum_mc(
        env_name="TestPointMass-v0",
        policy_structure=[16, 16],
        policy_act_fn=["tanh", "tanh", "none"],
        policy_squash=True,
        policy_cov_type="single",
    )
    with mc["conf"].ns("policy"):
        policy = DiagonalGaussianPolicy(mc)
    with mx.stream(mx.cpu):
        snap = behavior_snapshot(policy, 8)
    assert snap.host is not None and snap.overlaps_learner()
    held = [(weight.copy(), bias.copy()) for weight, bias in snap.host.weights]
    actor = snap.host.spawn()
    state = np.zeros((1, mc.env.observation_dim), np.float32)
    write_key(snap.rollout_key)
    expected, _ll = policy.act(mx.array(state), False)
    assert np.array_equal(actor(state), np.array(expected).reshape(-1))
    policy.update(tree_map(lambda p: p + 1, policy.parameters()))
    for (weight, bias), (weight0, bias0) in zip(snap.host.weights, held, strict=True):
        assert np.array_equal(weight, weight0)
        assert np.array_equal(bias, bias0)
