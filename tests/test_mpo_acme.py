"""MPO dual step, decoupled KL, stopped gradients, floors, mixture critic and vector envs."""

import torch
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
    decoupled_kl_diagonal,
    decoupled_kl_full,
    inverse_softplus,
    project_positive,
)
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import temperature_dual
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet


def test_dual_gradient_decreases_g():
    torch.manual_seed(0)
    q = torch.randn(64, 20) * 4
    raw = torch.nn.Parameter(inverse_softplus(torch.tensor(1.5)))
    loss = temperature_dual(q, project_positive(raw), 0.1)
    loss.backward()
    assert raw.grad is not None and raw.grad.abs().item() > 0
    with torch.no_grad():
        stepped = project_positive(raw.detach() - 1e-3 * raw.grad)
    assert temperature_dual(q, stepped, 0.1) < loss.detach()


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
    # equal variances make the mean KL symmetric; the covariance KL is not the reverse direction
    reverse_std = kl_divergence(Normal(mu_t, sig_o), Normal(mu_t, sig_t)).sum(-1).mean()
    assert not torch.allclose(kl_std, reverse_std, atol=1e-4)

    per_mean, per_std = decoupled_kl_diagonal(online, target, per_dim=True)
    assert per_mean.shape == per_std.shape == (2,)
    assert torch.allclose(per_mean.sum(), kl_mean, atol=1e-5)
    assert torch.allclose(per_std.sum(), kl_std, atol=1e-5)
    coupled = kl_divergence(Normal(mu_t, sig_t), Normal(mu_o, sig_o)).sum(-1).mean()
    assert not torch.allclose(kl_mean + kl_std, coupled, atol=1e-3)

    kl_mean.backward(retain_graph=True)
    assert mu_o.grad is not None and mu_o.grad.abs().sum() > 0
    assert sig_o.grad is None or sig_o.grad.abs().sum() == 0
    mu_o.grad = None
    kl_std.backward()
    assert sig_o.grad is not None and sig_o.grad.abs().sum() > 0
    assert mu_o.grad is None or mu_o.grad.abs().sum() == 0


def test_full_covariance_decoupled_kl_is_one_scalar_per_term():
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
    kl_mean, kl_std = decoupled_kl_full(mu_o, factor_o, mu_t, factor_t)
    assert kl_mean.ndim == 0 and kl_std.ndim == 0
    target = MultivariateNormal(mu_t, scale_tril=factor_t)
    expected_mean = kl_divergence(target, MultivariateNormal(mu_o, scale_tril=factor_t)).mean()
    expected_std = kl_divergence(target, MultivariateNormal(mu_t, scale_tril=factor_o)).mean()
    assert torch.allclose(kl_mean, expected_mean, atol=1e-5)
    assert torch.allclose(kl_std, expected_std, atol=1e-5)
    coupled = kl_divergence(target, MultivariateNormal(mu_o, scale_tril=factor_o)).mean()
    assert not torch.allclose(kl_mean + kl_std, coupled, atol=1e-3)


def test_eta_and_multipliers_stay_above_their_floors():
    raw = torch.tensor(-80.0)
    eta = project_positive(raw)
    floor = eta.new_tensor(1e-8)
    assert eta >= floor
    assert torch.all(project_positive(torch.tensor([-80.0, -80.0])) >= floor)


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
    alg.raw_eta.data.fill_(-80)
    alg.raw_alpha_mean.data.fill_(-80)
    alg.raw_alpha_std.data.fill_(-80)
    floor = alg.eta().new_tensor(1e-8)
    assert alg.eta().detach() >= floor
    assert alg.alpha_mean().detach().min() >= floor
    assert alg.alpha_std().detach().min() >= floor


def test_mog_critic_uses_the_mixture_mean_and_does_not_train_the_target(mpo_conf):
    conf = {**mpo_conf, "val_critic_type": "mog", "val_mog_value_samples": 4, "res_add_acts": 2}
    mc, _alg = _stack(conf)
    q_net = mc.get("q_network", False)
    target_q = mc.get("q_network", True)
    target_policy = mc.get("policy", True)
    obs = mc.env.observation_dim
    act = mc.env.action_dim
    states = torch.randn(4, obs)
    actions = torch.randn(4, act)
    mixture = q_net._mixture(states, actions)
    assert torch.allclose(q_net.q_val(states, actions).view(-1), mixture.mean)
    assert torch.all(mixture.component_distribution.stddev >= 1e-4)
    batch = Batch(
        states.numpy(),
        actions.numpy(),
        torch.zeros(4, 1).numpy(),
        torch.randn(4, 1).numpy(),
        torch.randn(4, obs).numpy(),
        torch.zeros(4, 1).numpy(),
    )
    before = [p.detach().clone() for p in target_q.parameters()]
    loss = q_net.tr_obj(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q_net.parameters())
    for old, new in zip(before, target_q.parameters()):
        assert torch.equal(old, new)
        assert new.grad is None
    assert all(p.grad is None for p in target_policy.parameters())


def test_vector_envs_keep_open_episodes_and_normalize_with_gymnasium(make_pendulum_mc):
    mc = make_pendulum_mc(env_num_envs=2, env_normalize=False)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
    collection = mc.env.execute_policy(policy, max_steps=5, batch_size=12, exploration=True)
    assert sum(len(traj) for traj in collection.trajectories()) >= 12
    for traj in collection.trajectories():
        assert len(traj) <= 5
        assert not traj.dones.any()
        assert torch.isfinite(torch.as_tensor(traj.rewards)).all()
    vec = mc.env._vector_env()
    calls: list[object] = []
    original_reset = vec.reset

    def spy(*args: object, **kwargs: object) -> object:
        options = kwargs.get("options")
        calls.append(None if not isinstance(options, dict) else dict(options))
        return original_reset(*args, **kwargs)

    vec.reset = spy
    mc.env.execute_policy(policy, max_steps=5, batch_size=4, exploration=True)
    # a later collection continues the open copies, so every reset is a partial one
    assert calls
    assert all(isinstance(options, dict) and "reset_mask" in options for options in calls)
    mc.env.close()

    mc_norm = make_pendulum_mc(env_num_envs=2, obs_norm=True, env_normalize=False)
    with mc_norm["conf"].ns("policy"):
        policy = select_policy(mc_norm)
    mc_norm.env.execute_policy(policy, max_steps=5, batch_size=6, exploration=True)
    assert mc_norm.env._norm is not None
    assert float(mc_norm.env._norm.obs_rms.count) > 1.0
    mc_norm.env.close()
