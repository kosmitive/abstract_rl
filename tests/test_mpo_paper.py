"""Paper MPO options: decoupled KL, Cholesky policy, and an Adam dual for η."""

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    find_optimal_temperature,
    fit_temperature_sgd,
    temperature_dual,
    temperature_raw,
)
from abstract_rl.policy.continuous.full_covariance_gaussian_policy import FullCovarianceGaussianPolicy
from abstract_rl.policy.continuous.gaussian_math import (
    cholesky_log_prob,
    kl_mean_covariance,
    kl_mean_covariance_cholesky,
    kl_normal,
    unpack_lower,
)
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric
from abstract_rl.scripts.hydra_conf import make_config


def test_diagonal_kl_parts_sum_to_the_closed_form():
    key = jax.random.PRNGKey(0)
    mu, sigma, other_mu, other_sigma = jax.random.uniform(key, (4, 6, 3), minval=0.2, maxval=1.4, dtype=jnp.float32)
    sigma = sigma + 0.3
    other_sigma = other_sigma + 0.3
    kl_mu, kl_sigma = kl_mean_covariance(mu, sigma, other_mu, other_sigma)
    closed = kl_normal(mu, sigma, other_mu, other_sigma)
    np.testing.assert_allclose(np.asarray(kl_mu + kl_sigma), np.asarray(closed), rtol=1e-5, atol=1e-5)


def _random_factor(key: jax.Array, batch: int, action_dim: int) -> np.ndarray:
    raw = np.asarray(jax.random.normal(key, (batch, action_dim, action_dim), dtype=jnp.float64))
    factor = np.tril(raw)
    diagonal = np.log1p(np.exp(np.clip(np.diagonal(factor, axis1=-2, axis2=-1), -40, 40))) + 0.2
    idx = np.arange(action_dim)
    factor[:, idx, idx] = diagonal
    return factor


def _numpy_kl(mu: np.ndarray, factor: np.ndarray, other_mu: np.ndarray, other_factor: np.ndarray) -> np.ndarray:
    """Closed form of KL(N(mu, L L^T) || N(other_mu, L_other L_other^T))."""
    covariance = factor @ np.swapaxes(factor, -1, -2)
    other = other_factor @ np.swapaxes(other_factor, -1, -2)
    other_inv = np.linalg.inv(other)
    trace = np.trace(other_inv @ covariance, axis1=-2, axis2=-1)
    delta = (mu - other_mu)[..., None]
    quad = np.swapaxes(delta, -1, -2) @ other_inv @ delta
    _sign, logdet = np.linalg.slogdet(covariance)
    _other_sign, other_logdet = np.linalg.slogdet(other)
    action_dim = mu.shape[-1]
    return 0.5 * (trace + quad[..., 0, 0] - action_dim + other_logdet - logdet)


def test_cholesky_kl_parts_sum_to_the_closed_form():
    key = jax.random.PRNGKey(1)
    key_l, key_o, key_mu = jax.random.split(key, 3)
    factor = _random_factor(key_l, 5, 3)
    other = _random_factor(key_o, 5, 3)
    mu = np.asarray(jax.random.normal(key_mu, (5, 3), dtype=jnp.float64))
    other_mu = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (5, 3), dtype=jnp.float64))
    kl_mu, kl_sigma = kl_mean_covariance_cholesky(
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(factor),
        jnp.asarray(other_mu),
        jnp.asarray(other),
    )
    closed = _numpy_kl(mu, factor, other_mu, other)
    got = np.asarray(kl_mu + kl_sigma)
    np.testing.assert_allclose(got, closed, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kl_mu), 0.5 * _quad(mu, other_mu, other), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kl_sigma), closed - 0.5 * _quad(mu, other_mu, other), rtol=1e-6, atol=1e-6)


def _quad(mu: np.ndarray, other_mu: np.ndarray, other_factor: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(other_factor, (mu - other_mu)[..., None])[..., 0]
    return np.sum(np.square(solved), axis=-1)


def _numpy_log_prob(values: np.ndarray, mu: np.ndarray, factor: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(factor[:, None, :, :], (values - mu[:, None, :])[..., None])[..., 0]
    log_diag = np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=-1)
    action_dim = mu.shape[-1]
    return -0.5 * np.sum(np.square(solved), axis=-1) - log_diag[:, None] - 0.5 * action_dim * np.log(2.0 * np.pi)


def test_cholesky_log_prob_matches_numpy():
    key = jax.random.PRNGKey(3)
    factor = _random_factor(key, 4, 3)
    mu = np.asarray(jax.random.normal(jax.random.PRNGKey(4), (4, 3), dtype=jnp.float64))
    values = np.asarray(jax.random.normal(jax.random.PRNGKey(5), (4, 6, 3), dtype=jnp.float64))
    got = cholesky_log_prob(jnp.asarray(values), jnp.asarray(mu), jnp.asarray(factor))
    np.testing.assert_allclose(np.asarray(got), _numpy_log_prob(values, mu, factor), rtol=1e-6, atol=1e-6)


def test_full_policy_log_prob_matches_numpy_with_and_without_squash(make_pendulum_mc):
    for squash in (False, True):
        mc = make_pendulum_mc(env_name="TestPointMass-v0", policy_cov_type="full", policy_squash=squash)
        with mc["conf"].ns("policy"):
            policy = FullCovarianceGaussianPolicy(mc)
            assert isinstance(select_policy(mc), FullCovarianceGaussianPolicy)
        states = np.zeros((3, 4), dtype=np.float32)
        _force_head(policy, [0.2, -0.4], [0.0, 0.5, -0.3])
        stats = np.asarray(policy.forward(states))
        mu = stats[:, :2]
        factor = np.asarray(unpack_lower(jnp.asarray(stats[:, 2:]), 2))
        diagonal = jax.nn.softplus(jnp.asarray([0.0, -0.3], dtype=jnp.float32))
        np.testing.assert_allclose(factor[0].diagonal(), np.asarray(diagonal), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(factor[0, 1, 0], 0.5, atol=1e-6)
        actions = np.array([[[0.1, -0.2], [0.0, 0.3]], [[-0.4, 0.2], [0.2, -0.1]], [[0.3, 0.1], [-0.2, 0.0]]])
        actions = actions.astype(np.float32)
        got = np.asarray(policy.log_prob(actions, suff_stats=jnp.asarray(stats)))
        if squash:
            center = np.asarray(policy.params.action_center, dtype=np.float32)
            scale = np.asarray(policy.params.action_scale, dtype=np.float32)
            y = np.clip((actions - center) / scale, -np.tanh(5.0), np.tanh(5.0))
            preimage = np.arctanh(y).astype(np.float64)
            log_jac = np.log(scale * (1.0 - np.square(y))).sum(axis=-1)
            expected = _numpy_log_prob(preimage, mu.astype(np.float64), factor.astype(np.float64)) - log_jac
        else:
            expected = _numpy_log_prob(actions.astype(np.float64), mu.astype(np.float64), factor.astype(np.float64))
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
        mode, log_np = policy.act(states, deterministic=True)
        log_jax = np.asarray(policy.log_prob(mode, states=states))[:, 0]
        np.testing.assert_allclose(log_np, log_jax, rtol=1e-4, atol=1e-4)


def _force_head(policy: FullCovarianceGaussianPolicy, mu: list[float], packed: list[float]) -> None:
    layers = list(policy.network.layers)
    layer = layers[-1]
    layer = eqx.tree_at(lambda lin: lin.weight, layer, jnp.zeros_like(layer.weight))
    layer = eqx.tree_at(lambda lin: lin.bias, layer, jnp.asarray([*mu, *packed], dtype=jnp.float32))
    layers[-1] = layer
    policy.network = eqx.tree_at(lambda net: net.layers, policy.network, tuple(layers))


def test_eta_sgd_converges_to_the_newton_solution():
    q_values = jax.random.normal(jax.random.PRNGKey(6), (64, 12), dtype=jnp.float64) * 3.0
    eps = 0.1
    newton = find_optimal_temperature(q_values, eps, eta0=1.0, max_steps=80)
    raw = temperature_raw(4.0).astype(jnp.float64)
    tx = optax.adam(0.05)
    _raw, _state, eta = fit_temperature_sgd(q_values, raw, tx.init(raw), tx, eps, 600)
    assert abs(float(eta) - float(newton[0])) < 1e-2
    dual_sgd = float(temperature_dual(q_values, eta, eps))
    dual_newton = float(temperature_dual(q_values, jnp.asarray(newton[0], dtype=jnp.float64), eps))
    assert abs(dual_sgd - dual_newton) < 1e-3


def test_mpo_paper_preset_keeps_the_mpo_defaults_and_turns_on_the_three_options():
    preset = make_config("pendulum", "mpo_paper")
    default = make_config("pendulum", "mpo")
    assert preset["alg"] == "mpo"
    assert preset["policy_type"] == "full"
    assert preset["policy_cov_type"] == "full"
    assert preset["mpo_kl_mode"] == "decoupled"
    assert preset["mpo_eps_mean"] == 0.1 and type(preset["mpo_eps_mean"]) is float
    assert preset["mpo_eps_cov"] == 1.0e-4 and type(preset["mpo_eps_cov"]) is float
    assert preset["eta_solver"] == "sgd"
    assert preset["num_epochs"] == 800 and preset["rnk_eps"] == 0.05
    assert "mpo_kl_mode" not in default
    assert "eta_solver" not in default
    assert "policy_type" not in default
    assert default["policy_cov_type"] == "single"


def _tiny(conf: dict) -> dict:
    updated = copy.deepcopy(conf)
    updated.update(
        {
            "num_epochs": 1,
            "env_sample": 80,
            "max_steps": 40,
            "val_opt_steps": 1,
            "policy_opt_steps": 2,
            "policy_batch_size": 16,
            "val_batch_size": 16,
            "res_add_acts": 4,
            "ret_add_acts": 4,
        }
    )
    return updated


def _finite(returns: np.ndarray) -> None:
    assert returns.shape == (1, 2)
    assert np.all(np.isfinite(returns))


def test_decoupled_kl_smoke(mpo_conf):
    conf = _tiny(mpo_conf)
    conf.update({"mpo_kl_mode": "decoupled", "mpo_eps_mean": 0.1, "mpo_eps_cov": 1.0e-4})
    _finite(run_mpo_non_parametric(conf))


def test_full_covariance_policy_smoke(mpo_conf):
    conf = _tiny(mpo_conf)
    conf.update({"env_name": "TestPointMass-v0", "max_steps": 20, "policy_type": "full", "policy_cov_type": "full"})
    _finite(run_mpo_non_parametric(conf))


def test_eta_sgd_smoke(mpo_conf):
    conf = _tiny(mpo_conf)
    conf.update({"eta_solver": "sgd"})
    _finite(run_mpo_non_parametric(conf))


def test_mpo_paper_preset_smoke(mpo_conf, tmp_path):
    conf = _tiny(make_config("pendulum", "mpo_paper"))
    conf["data_root"] = str(tmp_path)
    conf["seed"] = mpo_conf["seed"]
    _finite(run_mpo_non_parametric(conf))
