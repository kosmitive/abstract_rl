"""Generate the golden fixture of the parity tests from the PyTorch implementation.

Run inside a checkout of ``main`` at eb41282 (the last PyTorch commit), with its own environment::

    git worktree add /tmp/ar_torch eb41282
    cd /tmp/ar_torch && uv sync && uv run python <this repo>/tests/golden/generate_golden.py <this repo>/tests/golden

Inputs are drawn with a seeded numpy generator. All parameters are stored under their MLX names
(``layer_<i>`` of the PyTorch MLP is ``layers.<i>``), the parity tests load them into the MLX modules, run the same
computation and compare the outputs. The retrace operator estimates V by sampling actions, the fixture replaces this
by the value of the policy mode on both sides, so the comparison stays deterministic.
"""

import copy
import sys
from os.path import join
from typing import Any

import numpy as np
import torch

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)
from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm
from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.mpo_pendulum import run_config as mpo_config
from abstract_rl.scripts.trpo_pendulum import run_config as trpo_config
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet

OUT: dict[str, np.ndarray] = {}
RNG = np.random.default_rng(1234)


def f32(*shape: int, scale: float = 1.0) -> np.ndarray:
    return (scale * RNG.standard_normal(shape)).astype(np.float32)


def mlx_name(name: str) -> str:
    return name.replace("layer_", "layers.")


def params(prefix: str, named: dict[str, torch.Tensor]) -> None:
    for k, v in named.items():
        OUT[f"{prefix}/{mlx_name(k)}"] = v.detach().numpy().copy()


def policy_params(policy: Any) -> dict[str, torch.Tensor]:
    named = {f"network.{k}": v for k, v in policy.network.named_parameters()}
    named["cov"] = policy.cov
    return named


def set_params(module: torch.nn.Module, prefix: str, named: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, v in named.items():
            v.copy_(torch.as_tensor(OUT[f"{prefix}/{mlx_name(k)}"]))


def make_mc(base: dict[str, Any], alg: str, root: str) -> ModelConfiguration:
    conf = copy.deepcopy(base)
    conf.update({"alg": alg, "data_root": root})
    mc = ModelConfiguration(conf)
    mc.add_main("log", DataLogger(mc.data_dir))
    with mc["conf"].ns("env"):
        mc.add_environment(RolloutEnvWrapper(mc))
    return mc


def rollout(prefix: str, env: Any, length: int) -> TrajectoryBuilder:
    """A synthetic Pendulum like trajectory, stored so the MLX side can rebuild it."""
    s = f32(length + 1, 3)
    a = f32(length, 1)
    r = -np.abs(f32(length, 1, scale=3.0))
    ll = f32(length, 1, scale=0.5) - 1.0
    d = np.zeros((length, 1), dtype=np.int64)
    d[-1] = 1 if prefix.endswith("done") else 0
    for k, v in {"s": s, "a": a, "r": r, "ll": ll, "d": d}.items():
        OUT[f"{prefix}/{k}"] = v
    builder = TrajectoryBuilder(0, env, s[0].astype(np.float64))
    for t in range(length):
        builder.observe(
            a[t].astype(np.float64), float(r[t, 0]), s[t + 1].astype(np.float64), float(ll[t, 0]), int(d[t, 0])
        )
    return builder


def trpo_part(root: str) -> None:
    torch.manual_seed(0)
    mc = make_mc(trpo_config, "trpo", root)
    conf = mc["conf"]
    with conf.ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with conf.ns("val"):
        v_net = FullyVNeuralNet(mc)
        mc.add_syncable("v_network", v_net, 1)
    with conf.ns("tdop"):
        td_op = TemporalDifferenceOperator(mc)
    with conf.ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
    with conf.ns("trpo"):
        alg = TRPOAlgorithm(mc)
    t_policy = mc.get("policy", True)

    params("trpo/policy", policy_params(policy))
    params("trpo/v", dict(v_net.net.named_parameters()))
    set_params(t_policy, "trpo/policy", policy_params(t_policy))
    set_params(mc.get("v_network", True), "trpo/v", dict(mc.get("v_network", True).net.named_parameters()))

    # forward, log likelihood, mode
    n = 1000
    s, a = f32(n, 3), f32(n, 1)
    OUT["trpo/s"], OUT["trpo/act"] = s, a
    ts = torch.as_tensor(s)
    with torch.no_grad():
        OUT["trpo/stats"] = policy.forward(ts).numpy()
        OUT["trpo/ll"] = policy.log_prob(torch.as_tensor(a), states=ts).numpy()
        OUT["trpo/mode"] = policy.mode(states=ts).numpy()

    # KL between the policy and a perturbed target
    with torch.no_grad():
        for v in policy_params(t_policy).values():
            v.add_(0.05 * torch.as_tensor(f32(*v.shape)))
    params("trpo/policy_perturbed", policy_params(t_policy))
    batch = Batch(s, a, OUT["trpo/ll"], np.zeros((n, 1), np.float32), s, np.zeros((n, 1), np.float32), {})
    with torch.no_grad():
        OUT["trpo/kl_m_mean"] = policy.kl_divergence(batch, mode="m-projection", est="mean").numpy()
        OUT["trpo/kl_i_sum"] = policy.kl_divergence(batch, mode="i-projection", est="sum").numpy()
    set_params(t_policy, "trpo/policy", policy_params(t_policy))

    # conjugate gradient on a fixed SPD matrix
    m = RNG.standard_normal((20, 20))
    spd = (m @ m.T / 20 + np.eye(20)).astype(np.float32)
    b = f32(20)
    OUT["cg/a"], OUT["cg/b"] = spd, b
    OUT["cg/x"] = conjugate_gradient(lambda v: torch.as_tensor(spd) @ v, torch.as_tensor(b), 10, 1e-10).numpy()

    # one TRPO step, policy == target as after a sync
    adv = f32(n)
    OUT["trpo/adv"] = adv
    batch = Batch(s, a, OUT["trpo/ll"], np.zeros((n, 1), np.float32), s, np.zeros((n, 1), np.float32), {"a": adv})
    alg.perform_tr_sgd(0, batch, batch)
    params("trpo/policy_after_step", policy_params(policy))

    # five Adam steps of the value function
    tv = f32(256, scale=5.0)
    vs = f32(256, 3)
    OUT["v/s"], OUT["v/tv"] = vs, tv
    vb = Batch(vs, a[:256], a[:256], a[:256], vs, a[:256] * 0, {"tv": tv})
    for k in range(5):
        v_net.perform_adam_step(k, vb)
    params("trpo/v_after_adam", dict(v_net.net.named_parameters()))

    # td and gae on a truncated and on a terminated trajectory with the (target) value function of the start
    for name in ("traj_trunc", "traj_done"):
        traj = rollout(name, mc.env, 200).finalize()
        td_op.transform(traj)
        gae.transform(traj)
        OUT[f"{name}/tv"] = np.asarray(traj["tv"], dtype=np.float32)
        OUT[f"{name}/td"] = np.asarray(traj["td"], dtype=np.float32)
        OUT[f"{name}/adv"] = np.asarray(traj["a"], dtype=np.float32)


def mpo_part(root: str) -> None:
    torch.manual_seed(1)
    np.random.seed(1)  # noqa: NPY002
    mc = make_mc(mpo_config, "mpo", root)
    conf = mc["conf"]
    with conf.ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with conf.ns("mpo"):
        alg = MPOAlgorithm(mc)
    with conf.ns("val"):
        q_net = FullyQNeuralNet(mc)
        mc.add_syncable("q_network", q_net, 1)
    with conf.ns("ret"):
        ret = RetraceOperator(mc)
    t_q = mc.get("q_network", True)
    params("mpo/policy", policy_params(policy))
    params("mpo/q", dict(q_net.fc_net.named_parameters()))

    # retrace with the deterministic value of the policy mode
    def v_mode(states: torch.Tensor, exp_actions: int = 10, policy: Any = None) -> torch.Tensor:
        return t_q.q_val(states, policy.mode(states=states))

    t_q.v_val = v_mode  # type: ignore[method-assign]
    for name in ("mtraj_trunc", "mtraj_done"):
        traj = rollout(name, mc.env, 200).finalize()
        ret.transform(traj)
        OUT[f"{name}/q"] = traj["q"].numpy()
        OUT[f"{name}/tq"] = traj["tq"].numpy()

    # temperature dual
    q = f32(1000, 20, scale=3.0)
    OUT["dual/q"] = q
    OUT["dual/eta"] = find_optimal_temperature(torch.as_tensor(q), 0.05, 1.0, 0.05).numpy()

    # three M-steps
    n = 256
    s = f32(n, 3)
    add_act = f32(n, 20)
    logits = f32(n, 20, scale=2.0)
    grad_ranking = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    OUT["mstep/s"], OUT["mstep/add_act"], OUT["mstep/grad_ranking"] = s, add_act, grad_ranking.astype(np.float32)
    batch = Batch(
        s,
        add_act[:, :1],
        add_act[:, :1],
        add_act[:, :1],
        s,
        add_act[:, :1] * 0,
        {"add_act": add_act, "grad_ranking": OUT["mstep/grad_ranking"]},
    )
    with torch.no_grad():
        OUT["mstep/ll"] = policy.log_prob(torch.as_tensor(add_act[:, :1]), states=torch.as_tensor(s)).numpy()
    batch._log_likelihoods = OUT["mstep/ll"]
    for k in range(3):
        alg.perform_step(k, batch)
    params("mpo/policy_after_mstep", policy_params(policy))
    OUT["mstep/lm"] = np.concatenate([lm.detach().numpy() for lm in alg.lagrange_mults])

    # three Adam steps of the q function
    qs, qa, tq = f32(256, 3), f32(256, 1), f32(256, 1, scale=5.0)
    OUT["qfit/s"], OUT["qfit/a"], OUT["qfit/tq"] = qs, qa, tq
    qb = Batch(qs, qa, qa, qa, qs, qa * 0, {"tq": tq})
    for k in range(3):
        q_net.perform_adam_step(k, qb)
    params("mpo/q_after_adam", dict(q_net.fc_net.named_parameters()))


if __name__ == "__main__":
    out_dir = sys.argv[1]
    torch.set_num_threads(1)
    trpo_part("/tmp/ar_golden")
    mpo_part("/tmp/ar_golden")
    np.savez_compressed(join(out_dir, "torch_eb41282.npz"), **OUT)
    print(f"{len(OUT)} arrays written")
