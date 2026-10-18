"""Acme-style MPO: temperature dual, decoupled KL, stopped gradients and floors."""

import math

import torch
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)
from abstract_rl.algorithms.continuous.policy_gradient.mpo_loss import (
    ALPHA_FLOOR,
    ETA_FLOOR,
    decoupled_kl_diagonal,
    decoupled_kl_full,
    dual_descent_loss,
    inverse_softplus,
    normalized_weights,
    project_positive,
)
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import temperature_dual
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.hydra_conf import make_config
from abstract_rl.value_functions.distributional import c51_projection
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet


def test_temperature_dual_matches_the_log_mean_exp_formula():
    q = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    eta = torch.tensor(0.5)
    manual = eta * 0.1 + eta * torch.log(torch.exp(q / eta).mean(dim=1)).mean()
    assert torch.allclose(temperature_dual(q, eta, 0.1), manual, atol=1e-5)


def test_dual_gradient_decreases_g():
    torch.manual_seed(0)
    q = torch.randn(64, 20) * 4
    raw = torch.nn.Parameter(inverse_softplus(torch.tensor(1.5)))
    loss = temperature_dual(q, project_positive(raw, ETA_FLOOR), 0.1)
    loss.backward()
    grad = raw.grad.detach().clone()
    assert grad.abs().item() > 0
    with torch.no_grad():
        stepped = project_positive((raw.detach() - 1e-3 * grad).clamp(min=-20), ETA_FLOOR)
    decreased = temperature_dual(q, stepped, 0.1)
    assert decreased < loss.detach()


def test_normalized_weights_are_a_detached_softmax():
    q = torch.tensor([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    eta = torch.tensor(0.5, requires_grad=True)
    weights = normalized_weights(q, eta)
    assert not weights.requires_grad
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(weights, torch.softmax(q.detach() / eta.detach(), dim=1), atol=1e-6)
    # the weight tensor itself is a constant, so a downstream loss cannot reach Q or η
    weights.sum().backward()
    assert q.grad is None and eta.grad is None


def test_decoupled_kl_matches_the_mixed_gaussians_including_per_dimension():
    mu_o = torch.tensor([[0.2, -0.4], [0.0, 0.1]], requires_grad=True)
    sig_o = torch.tensor([[0.5, 1.2], [0.8, 0.7]], requires_grad=True)
    mu_t = torch.tensor([[-0.1, 0.3], [0.2, -0.2]])
    sig_t = torch.tensor([[0.9, 0.4], [1.1, 0.6]])
    online = torch.cat([mu_o, sig_o], dim=-1)
    target = torch.cat([mu_t, sig_t], dim=-1)

    kl_mean, kl_std = decoupled_kl_diagonal(online, target, per_dim=False)
    expected_mean = kl_divergence(Normal(mu_t, sig_t), Normal(mu_o, sig_t)).sum(-1).mean()
    expected_std = kl_divergence(Normal(mu_t, sig_t), Normal(mu_t, sig_o)).sum(-1).mean()
    assert torch.allclose(kl_mean, expected_mean, atol=1e-6)
    assert torch.allclose(kl_std, expected_std, atol=1e-6)
    # KL(π_old || π) is not the reverse direction once the means differ
    reverse = kl_divergence(Normal(mu_o, sig_t), Normal(mu_t, sig_t)).sum(-1).mean()
    assert not torch.allclose(kl_mean, reverse, atol=1e-4)

    per_mean, per_std = decoupled_kl_diagonal(online, target, per_dim=True)
    assert per_mean.shape == per_std.shape == (2,)
    assert torch.allclose(per_mean.sum(), kl_mean, atol=1e-5)
    assert torch.allclose(per_std.sum(), kl_std, atol=1e-5)
    # the joint KL(target || online) is a different quantity from the sum of the decoupled terms
    coupled = kl_divergence(Normal(mu_t, sig_t), Normal(mu_o, sig_o)).sum(-1).mean()
    assert not torch.allclose(kl_mean + kl_std, coupled, atol=1e-3)

    kl_mean.backward(retain_graph=True)
    assert mu_o.grad.abs().sum() > 0
    assert sig_o.grad is None or sig_o.grad.abs().sum() == 0
    mu_o.grad = None
    kl_std.backward()
    assert sig_o.grad.abs().sum() > 0
    assert mu_o.grad is None or mu_o.grad.abs().sum() == 0


def test_full_covariance_decoupled_kl_matches_the_analytic_gaussian():
    eye = torch.eye(2)
    factor_t = torch.stack([eye * 0.8, eye * 1.1])
    factor_o = torch.stack(
        [
            torch.tensor([[0.5, 0.0], [0.2, 0.7]]),
            torch.tensor([[0.9, 0.0], [-0.3, 0.4]]),
        ]
    )
    mu_t = torch.tensor([[0.2, -0.1], [0.0, 0.4]])
    mu_o = torch.tensor([[-0.3, 0.5], [0.2, -0.2]], requires_grad=True)
    factor_o = factor_o.detach().requires_grad_(True)
    kl_mean, kl_std = decoupled_kl_full(mu_o, factor_o, mu_t, factor_t, per_dim=False)
    target = MultivariateNormal(mu_t, scale_tril=factor_t)
    expected_mean = kl_divergence(target, MultivariateNormal(mu_o, scale_tril=factor_t)).mean()
    expected_std = kl_divergence(target, MultivariateNormal(mu_t, scale_tril=factor_o)).mean()
    assert torch.allclose(kl_mean, expected_mean, atol=1e-5)
    assert torch.allclose(kl_std, expected_std, atol=1e-5)

    per_mean, per_std = decoupled_kl_full(mu_o, factor_o, mu_t, factor_t, per_dim=True)
    assert per_mean.shape == (2,)
    assert per_std.ndim == 0
    assert torch.allclose(per_mean.sum(), kl_mean, atol=1e-5)
    assert torch.allclose(per_std, kl_std, atol=1e-5)


def test_multiplier_grows_when_the_kl_exceeds_the_bound():
    raw = torch.nn.Parameter(torch.zeros(()))
    before = project_positive(raw, ALPHA_FLOOR).detach()
    dual_descent_loss(torch.tensor(0.5), project_positive(raw, ALPHA_FLOOR), 5e-3).backward()
    assert raw.grad is not None and raw.grad.item() < 0
    with torch.no_grad():
        raw -= 0.5 * raw.grad
    assert project_positive(raw, ALPHA_FLOOR) > before


def test_eta_and_multipliers_stay_above_their_floors():
    raw = torch.tensor(-80.0)
    assert project_positive(raw, ETA_FLOOR).item() >= ETA_FLOOR
    assert project_positive(raw, ALPHA_FLOOR).item() >= ALPHA_FLOOR
    # softplus(-80) underflows; the floor is what keeps η usable in the dual
    assert math.isclose(project_positive(raw, ETA_FLOOR).item(), ETA_FLOOR)


def _stack(conf):
    mc = ModelConfiguration(conf)
    mc.add_main("log", DataLogger(mc.data_dir))
    with mc["conf"].ns("env"):
        mc.add_environment(MCMCEnvWrapper(mc))
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, 100)
    with mc["conf"].ns("mpo"):
        alg = MPOAlgorithm(mc)
    with mc["conf"].ns("val"):
        q_net = FullyQNeuralNet(mc)
        mc.add_syncable("q_network", q_net, 100)
    return mc, alg


def test_target_networks_and_e_step_weights_get_no_gradients(mpo_conf):
    mc, alg = _stack(mpo_conf)
    policy = mc.get("policy", False)
    target_policy = mc.get("policy", True)
    target_q = mc.get("q_network", True)
    before_policy = [p.detach().clone() for p in target_policy.parameters()]
    before_q = [p.detach().clone() for p in target_q.parameters()]
    seen = {}
    original = target_policy.sample_actions

    def spy(states=None, suff_stats=None, num_actions=1):
        seen["num_actions"] = num_actions
        return original(states=states, suff_stats=suff_stats, num_actions=num_actions)

    target_policy.sample_actions = spy
    obs = mc.env.observation_dim
    states = torch.randn(8, obs).numpy()
    batch = Batch(states, states[:, :1], states[:, :1], states[:, :1], states, states[:, :1])
    alg.perform_step(0, batch)

    assert seen["num_actions"] == 20
    for old, new in zip(before_policy, target_policy.parameters()):
        assert torch.equal(old, new)
        assert new.grad is None
    for old, new in zip(before_q, target_q.parameters()):
        assert torch.equal(old, new)
        assert new.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    # the dual step trained η, and the projection still respects the floor
    alg.raw_eta.data.fill_(-80)
    alg.raw_alpha_mean.data.fill_(-80)
    alg.raw_alpha_std.data.fill_(-80)
    assert float(alg.eta()) >= ETA_FLOOR
    assert float(alg.alpha_mean().min()) >= ALPHA_FLOOR
    assert float(alg.alpha_std().min()) >= ALPHA_FLOOR


def test_mpo_defaults_follow_the_acme_numbers():
    conf = make_config("InvertedPendulum-v5", "mpo")
    assert conf["rnk_eps"] == 0.1
    assert conf["res_add_acts"] == 20
    assert conf["mpo_epsilon_mean"] == 0.005
    assert conf["mpo_epsilon_stddev"] == 1e-5
    assert conf["mpo_outer_lr"] == 1e-4
    assert conf["val_lr"] == 1e-4
    assert conf["mpo_inner_lr"] == 1e-2
    assert conf["mpo_grad_clip"] == 40.0
    assert conf["val_grad_clip"] == 40.0
    assert conf["mpo_per_dim_constraint"] is True
    assert conf["mpo_eta_solver"] == "dual"
    assert conf["mpo_samples_per_insert"] == 32.0
    assert conf["policy_sync_steps"] == 100 and conf["val_sync_steps"] == 100
    assert conf["mpo_target_update"] == "hard"
    assert conf["val_distributional"] == "none"
    assert conf["ret_operator"] == "retrace"
    # Pendulum keeps the previously tuned E-step epsilon
    assert make_config("Pendulum-v1", "mpo")["rnk_eps"] == 0.05


def test_c51_projection_is_a_distribution_and_a_one_hot_on_an_atom():
    atoms = torch.linspace(-1.0, 1.0, 5)
    on_atom = c51_projection(torch.tensor([atoms[2]]), atoms)
    assert torch.allclose(on_atom.sum(-1), torch.ones(1))
    assert on_atom.argmax() == 2
    # atoms are -1, -0.5, 0, 0.5, 1; 0.25 sits halfway between 0 and 0.5
    between = c51_projection(torch.tensor([0.25]), atoms)
    assert torch.allclose(between.sum(-1), torch.ones(1), atol=1e-5)
    assert between[0, 2] > 0 and between[0, 3] > 0


def test_vector_envs_collect_the_requested_transitions(make_pendulum_mc):
    mc = make_pendulum_mc(env_num_envs=2, env_normalize=False)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
    collection = mc.env.execute_policy(policy, max_steps=5, batch_size=12, exploration=True)
    assert sum(len(traj) for traj in collection.trajectories()) >= 12
    for traj in collection.trajectories():
        assert len(traj) <= 5
        assert not traj.dones.any()
        assert torch.isfinite(torch.as_tensor(traj.rewards)).all()
    mc.env.close()
