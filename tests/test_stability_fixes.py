"""Multiplier clamp, episode entropy, and grad_clip."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
    _mpo_scan,
    _step_columns,
    initial_multiplier,
    multiplier_alpha,
    project_multiplier,
)
from abstract_rl.algorithms.mixed.vanilla_gradient import adam_transform
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.misc.flat_params import weight_spec
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, entropy_from_sigma, log_prob, sample
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts.hydra_conf import make_config
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


def _finite(tree: object) -> None:
    for leaf in jax.tree.leaves(tree):
        if eqx.is_array(leaf):
            assert np.all(np.isfinite(np.asarray(leaf)))


def _adam_on_clipped(lr: float, grad_clip: float, params: jax.Array, grads: jax.Array) -> jax.Array:
    scale = grad_clip / float(jnp.linalg.norm(grads))
    clipped = grads if scale >= 1.0 else grads * scale
    plain = optax.adam(lr)
    updates, _state = plain.update(clipped, plain.init(params), params)
    return updates


def test_log_columns_name_alpha_kl_and_slack():
    lm = initial_multiplier(1.0, 2)
    slack = jnp.array([0.02, -0.1], dtype=jnp.float32)
    epsilon = jnp.array([0.1, 1.0e-4], dtype=jnp.float32)
    alpha_mu, alpha_sigma, kl_mu, kl_sigma, slack_mu, slack_sigma = _step_columns(lm, slack, epsilon)
    alpha = np.asarray(multiplier_alpha(lm))
    assert alpha_mu == pytest.approx(float(alpha[0]), abs=1e-6)
    assert alpha_sigma == pytest.approx(float(alpha[1]), abs=1e-6)
    assert kl_mu == pytest.approx(0.1 - 0.02)
    assert kl_sigma == pytest.approx(1.0e-4 + 0.1)
    assert slack_mu == pytest.approx(0.02)
    assert slack_sigma == pytest.approx(-0.1)


def test_unreachable_kl_keeps_multipliers_finite_and_inside_the_clamp():
    """A covariance KL that stays above its bound must not freeze alpha or Adam."""
    key = jax.random.PRNGKey(0)
    k_net, k_scan = jax.random.split(key)
    network = MultilayerPerceptron(k_net, [2, 4, 1], ["tanh", "none"], False, "xavier_gaussian")
    policy = GaussianParams(
        network=network,
        log_sigma=jnp.zeros((1,), dtype=jnp.float32),
        cov_type="single",
        squash=False,
        sigma_is_softplus=False,
        action_center=(0.0,),
        action_scale=(1.0,),
        fixed_log_sigma=None,
    )
    capacity = 64
    states = jnp.zeros((capacity, 2), dtype=jnp.float32)
    # sigma = 1 against a target sigma of 0.2. That covariance KL is about 10 and does not move,
    # because the policy step size is zero.
    target = jnp.broadcast_to(jnp.asarray([3.0, 0.2], dtype=jnp.float32), (capacity, 2))
    actions = jnp.zeros((capacity, 4, 1), dtype=jnp.float32)
    q_values = jnp.zeros((capacity, 4), dtype=jnp.float32)
    diff, static = eqx.partition(policy, weight_spec(policy))
    lm = initial_multiplier(1.0, 2)
    assert np.asarray(multiplier_alpha(lm)) == pytest.approx(1.0, abs=1e-5)
    tx_outer = optax.adam(0.0)
    tx_inner = optax.adam(5.0)
    eta_raw = jnp.zeros((), dtype=jnp.float32)
    tx_eta = optax.adam(1e-3)
    alpha_min, alpha_max = 1e-6, 1e3
    steps = 400
    out = _mpo_scan(
        (states, actions, target, q_values, k_scan, jnp.int32(capacity)),
        diff,
        static,
        tx_outer.init(diff),
        lm,
        tx_inner.init(lm),
        eta_raw,
        tx_eta.init(eta_raw),
        tx_outer,
        tx_inner,
        tx_eta,
        steps,
        16,
        jnp.asarray([alpha_min, alpha_min], dtype=jnp.float32),
        0.0,
        0.1,
        alpha_min,
        alpha_max,
    )
    history = np.asarray(multiplier_alpha(out[7]))
    final_slack = np.asarray(out[8][-1])
    assert history.shape == (steps, 2)
    assert np.all(np.isfinite(history))
    assert np.all(history >= alpha_min - 1e-8)
    assert np.all(history <= alpha_max * (1.0 + 1e-4))
    assert final_slack[1] < 0.0
    assert history[-1, 1] == pytest.approx(alpha_max, rel=1e-4, abs=1e-2)
    assert history[-1, 1] > history[0, 1]
    _finite(out[2])
    _finite(out[1])
    _finite(out[3])
    projected = project_multiplier(jnp.asarray(out[2]), alpha_min, alpha_max)
    assert np.allclose(np.asarray(projected), np.asarray(out[2]), rtol=1e-5, atol=1e-4)


def test_trajectory_entropy_matches_the_unsquashed_diagonal_gaussian():
    key = jax.random.PRNGKey(1)
    k_net, k_sample = jax.random.split(key)
    sigma = jnp.asarray([0.7, 1.3], dtype=jnp.float32)
    network = MultilayerPerceptron(k_net, [1, 2], ["none"], False, "xavier_gaussian")
    params = GaussianParams(
        network=network,
        log_sigma=jnp.log(sigma),
        cov_type="single",
        squash=False,
        sigma_is_softplus=False,
        action_center=(0.0, 0.0),
        action_scale=(1.0, 1.0),
        fixed_log_sigma=None,
    )
    n = 20_000
    states = jnp.zeros((n, 1), dtype=jnp.float32)
    actions = sample(params, k_sample, 1, states=states).reshape((n, 2))
    logp = np.asarray(log_prob(params, actions, states=states)).reshape(-1)
    closed = float(entropy_from_sigma(jnp.broadcast_to(sigma, (n, 2))))
    estimate = float(-np.mean(logp))
    assert estimate == pytest.approx(closed, abs=0.05)

    class _Env:
        observation_dim = 1
        action_dim = 2

    rows = n + 1
    frame = pd.DataFrame(
        {
            "s_0": np.zeros(rows),
            "a_0": np.zeros(rows),
            "a_1": np.zeros(rows),
            "r": np.zeros(rows),
            "d": np.zeros(rows),
            "ll": np.concatenate([logp, [0.0]]),
        }
    )
    entropy = Trajectory(0, _Env(), frame).entropy
    assert entropy == pytest.approx(estimate, abs=1e-6)
    assert entropy == pytest.approx(closed, abs=0.05)


def test_grad_clip_zero_matches_plain_adam():
    lr = 1e-2
    params = jnp.array([1.0, -2.0, 0.5, 4.0], dtype=jnp.float32)
    grads = jnp.array([80.0, -40.0, 10.0, -5.0], dtype=jnp.float32)
    plain = optax.adam(lr)
    off = adam_transform(lr, 0.0)
    plain_update, _plain_state = plain.update(grads, plain.init(params), params)
    off_update, _off_state = off.update(grads, off.init(params), params)
    np.testing.assert_array_equal(np.asarray(plain_update), np.asarray(off_update))


def test_positive_grad_clip_matches_adam_on_the_clipped_gradient():
    lr = 1e-3
    params = jnp.ones((4,), dtype=jnp.float32)
    grads = jnp.full((4,), 30.0, dtype=jnp.float32)
    updates, _state = adam_transform(lr, 40.0).update(grads, adam_transform(lr, 40.0).init(params), params)
    expected = _adam_on_clipped(lr, 40.0, params, grads)
    np.testing.assert_allclose(np.asarray(updates), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_configured_optimisers_apply_grad_clip(tmp_path, make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        value = FullyVNeuralNet(mc)
    assert value.grad_clip == 40.0
    probe = jnp.ones((8,), dtype=jnp.float32)
    large = jnp.full((8,), 100.0, dtype=jnp.float32)
    got, _state = value.tx.update(large, value.tx.init(probe), probe)
    expected = _adam_on_clipped(value.lr, 40.0, probe, large)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-6)

    conf = make_config("pendulum", "mpo", 1)
    conf.update({"alg": "mpo", "data_root": str(tmp_path)})
    assert conf["grad_clip"] == 40.0 and type(conf["grad_clip"]) is float
    assert conf["mpo_alpha_init"] == 1.0 and type(conf["mpo_alpha_init"]) is float
    assert conf["mpo_alpha_min"] == 1.0e-6 and type(conf["mpo_alpha_min"]) is float
    assert conf["mpo_alpha_max"] == 1.0e3 and type(conf["mpo_alpha_max"]) is float
    assert make_config("pendulum", "trpo", 1)["grad_clip"] == 40.0
    mpo = ModelConfiguration(conf)
    with mpo["conf"].ns("env"):
        mpo.add_environment(MCMCEnvWrapper(mpo))
    with mpo["conf"].ns("policy"):
        mpo.add_syncable("policy", select_policy(mpo), 1)
    with mpo["conf"].ns("mpo"):
        algorithm = MPOAlgorithm(mpo)
    with mpo["conf"].ns("val"):
        critic = FullyQNeuralNet(mpo)
    assert np.asarray(multiplier_alpha(algorithm.lm)) == pytest.approx(1.0, abs=1e-5)
    for transform, lr in (
        (algorithm.tx_outer, algorithm.outer_lr),
        (algorithm.tx_inner, algorithm.inner_lr),
        (critic.tx, critic.lr),
    ):
        update, _state = transform.update(large, transform.init(probe), probe)
        reference = _adam_on_clipped(lr, 40.0, probe, large)
        np.testing.assert_allclose(np.asarray(update), np.asarray(reference), rtol=1e-5, atol=1e-6)
