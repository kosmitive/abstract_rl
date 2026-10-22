"""Record reference outputs of the PyTorch implementation for the JAX parity tests (tests/test_parity_*.py).

Run on the last torch commit. After the port removes torch, re-record from a worktree of that commit:
    git worktree add ../abstract_rl_torch ab552fb
    cd ../abstract_rl_torch
    uv run python tests/fixtures/record_torch_golden.py <repo>/tests/fixtures/torch_golden.npz
"""

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conftest  # noqa: F401  (registers TestPointMass-v0)
from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)
from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import (
    TRPOAlgorithm,
)
from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import (
    find_optimal_temperature,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_builder import TrajectoryBuilder
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.mpo_pendulum import run_config as mpo_config
from abstract_rl.scripts.trpo_pendulum import run_config as trpo_config
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet

OUT: dict[str, np.ndarray] = {}


def put(prefix: str, **arrays) -> None:
    for k, v in arrays.items():
        OUT[f"{prefix}/{k}"] = v.detach().cpu().numpy().copy() if torch.is_tensor(v) else np.asarray(v)


def put_params(prefix: str, module: torch.nn.Module, grads: bool = False, sub: str = "params") -> None:
    for n, p in module.named_parameters():
        OUT[f"{prefix}/{sub}/{n}"] = p.detach().numpy().copy()
        if grads:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            OUT[f"{prefix}/grads/{n}"] = g.detach().numpy().copy()


def make_mc(root: str, base: dict, **overrides) -> ModelConfiguration:
    conf = {**copy.deepcopy(base), "data_root": root, **overrides}
    mc = ModelConfiguration(conf)
    mc.add_main("log", DataLogger(mc.data_dir))
    with mc["conf"].ns("env"):
        mc.add_environment(RolloutEnvWrapper(mc))
    return mc


def perturb(module: torch.nn.Module, scale: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))


def make_traj(env, length: int, done_last: int, rng: np.random.Generator):
    high = np.asarray(env.action_space.high, dtype=np.float64)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    builder = TrajectoryBuilder(0, env, rng.normal(size=env.observation_dim))
    for t in range(length):
        act = rng.uniform(low * 0.9, high * 0.9)
        done = int(done_last and t == length - 1)
        builder.observe(act, float(rng.normal()), rng.normal(size=env.observation_dim), float(rng.normal() - 1), done)
    return builder.finalize()


def record_mlps(root: str) -> None:
    mc = make_mc(root, trpo_config, alg="trpo")
    cases = {
        "tanh_tanh_none": ([3, 64, 64, 2], ["tanh", "tanh", "none"], False, "xavier_gaussian"),
        "ln_elu": ([3, 32, 32, 1], ["elu", "elu", "none"], True, "xavier_gaussian"),
        "split_head": ([3, 16, 2], ["tanh", ["none", "softplus"]], False, "xavier_gaussian"),
        "single_act": ([3, 8, 8, 1], ["relu"], False, "orthogonal"),
        "prelu_lrelu": ([3, 8, 8, 1], ["prelu", "lrelu", "sigmoid"], False, "gaussian"),
    }
    for name, (structure, act_fn, layer_norm, init) in cases.items():
        torch.manual_seed(0)
        net = MultilayerPerceptron(mc, np.array(structure), act_fn, layer_norm, init)
        x = torch.randn(16, structure[0])
        put(f"mlp_{name}", x=x, y=net.forward(x))
        put_params(f"mlp_{name}", net)


POLICY_CASES = {
    "uni_single_squash": ("Pendulum-v1", {"policy_cov_type": "single", "policy_squash": True}),
    "uni_state_nosquash": ("Pendulum-v1", {"policy_cov_type": "state", "policy_squash": False}),
    "uni_fixed_squash": ("Pendulum-v1", {"policy_cov_type": "fixed", "policy_squash": True, "policy_sigma": 0.5}),
    "diag_single_squash": ("TestPointMass-v0", {"policy_cov_type": "single", "policy_squash": True}),
    "diag_state_squash": ("TestPointMass-v0", {"policy_cov_type": "state", "policy_squash": True}),
    "diag_full_squash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": True}),
    "diag_full_nosquash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": False}),
}


def record_policy(root: str, name: str, env: str, overrides: dict) -> None:
    torch.manual_seed(1)
    mc = make_mc(root, trpo_config, alg="trpo", env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    target = mc.get("policy", True)
    perturb(target, 0.05)
    n = 32
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        stats = policy.forward(states)
        actions = policy.sample_actions(suff_stats=stats, num_actions=5)
        single = policy.sample_actions(suff_stats=stats, num_actions=1).reshape(n, -1)
        lp = policy.log_prob(actions, suff_stats=stats)
        lp1 = policy.log_prob(single, states=states)
        mode = policy.mode(states=states)
    lls = lp1.reshape(n, 1) + 0.1 * torch.randn(n, 1)
    a = torch.randn(n)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), single.numpy(), lls.numpy(), zeros, states.numpy(), zeros, {"a": a.numpy()})
    kls = {
        f"kl_{m.split('-')[0]}_{e}": policy.kl_divergence(batch, mode=m, est=e).detach()
        for m in ("i-projection", "m-projection")
        for e in ("mean", "sum")
    }
    put(
        name, states=states, stats=stats, actions=actions, lp=lp, single=single, lp1=lp1, mode=mode, lls=lls, a=a, **kls
    )
    if not policy.squash:
        put(name, entropy=policy.entropy(batch).detach())
    policy.zero_grad()
    policy.kl_divergence(batch, mode="m-projection", est="mean").backward()
    put_params(name, policy, grads=True)
    put_params(name, target, sub="target")
    policy.zero_grad()
    lr = policy.likelihood_ratio(batch)
    lr.backward()
    put(name, lr=lr.detach())
    put_params(f"{name}_lr", policy, grads=True)


def record_values(root: str) -> None:
    torch.manual_seed(2)
    mc = make_mc(root, trpo_config, alg="trpo")
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc)
    states, tv = torch.randn(40, 3), torch.randn(40)
    zeros = np.zeros((40, 1))
    batch = Batch(states.numpy(), zeros, zeros, zeros, states.numpy(), zeros, {"tv": tv.numpy()})
    obj = v.tr_obj(batch)
    obj.backward()
    put("v_net", states=states, tv=tv, forward=v.forward(states).detach(), obj=obj.detach())
    put_params("v_net", v, grads=True)
    for s in range(3):
        v.perform_adam_step(s, batch)
    put_params("v_net", v, sub="adam3/params")

    torch.manual_seed(2)
    mc = make_mc(root, mpo_config, alg="mpo")
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc)
    states, actions, add_actions = torch.randn(40, 3), torch.rand(40, 1) * 4 - 2, torch.rand(40, 4) * 4 - 2
    tq = torch.randn(40, 1)
    batch = Batch(states.numpy(), actions.numpy(), zeros, zeros, states.numpy(), zeros, {"tq": tq.numpy()})
    obj = q.tr_obj(batch)
    obj.backward()
    put(
        "q_net",
        states=states,
        actions=actions,
        add_actions=add_actions,
        tq=tq,
        q_val=q.q_val(states, actions).detach(),
        q_val_multi=q.q_val_multi(states, add_actions).detach(),
        obj=obj.detach(),
    )
    put_params("q_net", q, grads=True)


def record_advantages(root: str) -> None:
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    mc = make_mc(root, trpo_config, alg="trpo")
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc)
        mc.add_syncable("v_network", v, 1)
    with mc["conf"].ns("tdop"):
        td = TemporalDifferenceOperator(mc)
    with mc["conf"].ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
    for case, (length, done_last) in {"truncated": (12, 0), "terminated": (7, 1)}.items():
        traj = make_traj(mc.env, length, done_last, rng)
        td.transform(traj)
        gae.transform(traj)
        put(f"adv_{case}", all_states=traj.all_states, rewards=traj.rewards, dones=traj.dones,
            tv=traj["tv"], td=traj["td"], a=traj["a"])
    put_params("adv_v", v)


def _deterministic_rsample(self, sample_shape=torch.Size()):  # noqa: B008
    return self.loc.expand(self._extended_shape(sample_shape)).clone()


def record_retrace(root: str) -> None:
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    Normal.rsample, MultivariateNormal.rsample = _deterministic_rsample, _deterministic_rsample
    mc = make_mc(root, mpo_config, alg="mpo")
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc)
        mc.add_syncable("q_network", q, 1)
    with mc["conf"].ns("ret"):
        ret = RetraceOperator(mc)
    with mc["conf"].ns("res"):
        res = ResampleOperator(mc)
    for case, (length, done_last) in {"truncated": (12, 0), "terminated": (7, 1)}.items():
        traj = make_traj(mc.env, length, done_last, rng)
        res.transform(traj)
        ret.transform(traj)
        put(f"retrace_{case}", all_states=traj.all_states, actions=traj.actions, rewards=traj.rewards,
            dones=traj.dones, log_likelihoods=traj.log_likelihoods, q=traj["q"], tq=traj["tq"],
            add_act=traj["add_act"], add_q=traj["add_q"], add_pi=traj["add_pi"])
    put_params("retrace_policy", policy)
    put_params("retrace_q", q)


def record_cg() -> None:
    torch.manual_seed(5)
    n = 8
    m = torch.randn(n, n)
    a = m @ m.T + n * torch.eye(n)
    b = torch.randn(n)
    put("cg", A=a, b=b, x=conjugate_gradient(lambda v: a @ v, b, 10, 1e-10))


def record_trpo(root: str, name: str, env: str) -> None:
    torch.manual_seed(6)
    mc = make_mc(root, trpo_config, alg="trpo", env_name=env)
    with mc["conf"].ns("val"):
        mc.add_syncable("v_network", FullyVNeuralNet(mc), 1)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    with mc["conf"].ns("trpo"):
        alg = TRPOAlgorithm(mc)
    n = 64
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        actions = policy.sample_actions(states=states).reshape(n, -1)
        lls = policy.log_prob(actions, states=states).reshape(n, 1)
    a = torch.randn(n)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), actions.numpy(), lls.numpy(), zeros, states.numpy(), zeros, {"a": a.numpy()})
    put(name, states=states, actions=actions, lls=lls, a=a)
    put_params(name, policy, sub="before/params")
    alg.perform_tr_sgd(0, batch, batch)
    put_params(name, policy, sub="after/params")


def record_dual() -> None:
    torch.manual_seed(7)
    q = torch.randn(256, 20) * 10
    for case, eta0 in {"cold": 1.0, "warm": 3.0}.items():
        put(f"dual_{case}", q=q, eta0=np.float64(eta0), eta=find_optimal_temperature(q, 0.1, eta0, 0.05))


def record_mpo(root: str, name: str, overrides: dict) -> None:
    torch.manual_seed(8)
    mc = make_mc(root, mpo_config, alg="mpo", **overrides)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 1)
    target = mc.get("policy", True)
    perturb(target, 0.05)
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    n = 64
    states = torch.randn(n, mc.env.observation_dim)
    with torch.no_grad():
        add_act = target.sample_actions(states=states, num_actions=20)
    grad_ranking = torch.log_softmax(torch.randn(n, 20), dim=1)
    zeros = np.zeros((n, 1))
    batch = Batch(states.numpy(), zeros, zeros, zeros, states.numpy(), zeros,
                  {"add_act": add_act.numpy(), "grad_ranking": grad_ranking.numpy(), "add_pi": np.zeros((n, 20))})
    put(name, states=states, add_act=add_act, grad_ranking=grad_ranking)
    put_params(name, target, sub="target/params")
    put_params(name, policy, sub="step0/params")
    put(f"{name}/step0", lagrange=torch.stack([lm.detach() for lm in alg.lagrange_mults]))
    for s in range(3):
        alg.perform_step(s, batch)
        put_params(name, policy, sub=f"step{s + 1}/params")
        put(f"{name}/step{s + 1}", lagrange=torch.stack([lm.detach() for lm in alg.lagrange_mults]))


def record_filters() -> None:
    rng = np.random.default_rng(9)
    xs = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([0.0, 2.0, -1.0])
    f = RunningCenteredFilter("states", 3)
    means, stds, outs = [], [], []
    for x in xs:
        outs.append(f(x))
        means.append(f.mean.copy())
        stds.append(f.std.copy())
    put("filter", xs=xs, means=np.stack(means), stds=np.stack(stds), outs=np.stack(outs))

    from gymnasium.wrappers.utils import RunningMeanStd

    rms = RunningMeanStd(shape=(3,))
    means, variances, outs = [], [], []
    for x in xs:
        rms.update(np.array([x]))
        outs.append((x - rms.mean) / np.sqrt(rms.var + 1e-8))
        means.append(rms.mean.copy())
        variances.append(rms.var.copy())
    put("obsnorm", xs=xs, means=np.stack(means), vars=np.stack(variances), outs=np.stack(outs))


def main(out_path: str) -> None:
    with tempfile.TemporaryDirectory() as root:
        record_mlps(root)
        for name, (env, overrides) in POLICY_CASES.items():
            record_policy(root, name, env, overrides)
        record_values(root)
        record_advantages(root)
        record_cg()
        record_trpo(root, "trpo_uni", "Pendulum-v1")
        record_trpo(root, "trpo_diag", "TestPointMass-v0")
        record_dual()
        record_mpo(root, "mpo_uni", {})
        record_mpo(root, "mpo_split", {"mpo_coordinate_ascent_objective": True,
                                       "mpo_coordinate_ascent_constraint": True,
                                       "mpo_trust_regions": [0.01, 0.02]})
        record_mpo(root, "mpo_diag", {"env_name": "TestPointMass-v0"})
        record_filters()
        record_retrace(root)  # last: it patches torch sampling to be deterministic
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    OUT["meta/commit"] = np.array(commit)
    np.savez_compressed(out_path, **OUT)
    print(f"wrote {len(OUT)} arrays to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
