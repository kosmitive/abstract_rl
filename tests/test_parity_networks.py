import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet
from golden import golden, load_named, named_parameters, sub

MLP_CASES = {
    "tanh_tanh_none": ([3, 64, 64, 2], ["tanh", "tanh", "none"], False, "xavier_gaussian"),
    "ln_elu": ([3, 32, 32, 1], ["elu", "elu", "none"], True, "xavier_gaussian"),
    "split_head": ([3, 16, 2], ["tanh", ["none", "softplus"]], False, "xavier_gaussian"),
    "single_act": ([3, 8, 8, 1], ["relu"], False, "orthogonal"),
    "prelu_lrelu": ([3, 8, 8, 1], ["prelu", "lrelu", "sigmoid"], False, "gaussian"),
}


@pytest.mark.parametrize("name", MLP_CASES)
def test_mlp_forward_matches_torch(make_pendulum_mc, name):
    structure, act_fn, layer_norm, init = MLP_CASES[name]
    g = golden(f"mlp_{name}")
    net = MultilayerPerceptron(make_pendulum_mc(), np.array(structure), act_fn, layer_norm, init, key=jax.random.key(0))
    net = load_named(net, sub(g, "params"))
    np.testing.assert_allclose(net.forward(jnp.asarray(g["x"])), g["y"], rtol=1e-5, atol=1e-5)


POLICY_CASES = {
    "uni_single_squash": ("Pendulum-v1", {"policy_cov_type": "single", "policy_squash": True}),
    "uni_state_nosquash": ("Pendulum-v1", {"policy_cov_type": "state", "policy_squash": False}),
    "uni_fixed_squash": ("Pendulum-v1", {"policy_cov_type": "fixed", "policy_squash": True, "policy_sigma": 0.5}),
    "diag_single_squash": ("TestPointMass-v0", {"policy_cov_type": "single", "policy_squash": True}),
    "diag_state_squash": ("TestPointMass-v0", {"policy_cov_type": "state", "policy_squash": True}),
    "diag_full_squash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": True}),
    "diag_full_nosquash": ("TestPointMass-v0", {"policy_cov_type": "full", "policy_squash": False}),
}


def _golden_policy(make_pendulum_mc, name):
    env, overrides = POLICY_CASES[name]
    mc = make_pendulum_mc(env_name=env, **overrides)
    with mc["conf"].ns("policy"):
        fresh = select_policy(mc, jax.random.key(0))
    g = golden(name)
    policy, target = load_named(fresh, sub(g, "params")), load_named(fresh, sub(g, "target"))
    return policy, ModelState.create({}, {"policy": (policy, target)}), g


def _batch(g):
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    return Batch(
        jnp.asarray(g["states"]),
        jnp.asarray(g["single"]),
        jnp.asarray(g["lls"]),
        z,
        jnp.asarray(g["states"]),
        z,
        {"a": jnp.asarray(g["a"])},
    )


def _close_grads(grads, expected):
    got = named_parameters(grads)
    assert set(got) == set(expected)
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], rtol=1e-4, atol=1e-5, err_msg=k)


@pytest.mark.parametrize("name", [n for n in POLICY_CASES if n.startswith("uni")])
def test_univariate_policy_matches_torch(make_pendulum_mc, name):
    policy, state, g = _golden_policy(make_pendulum_mc, name)
    states, stats = jnp.asarray(g["states"]), policy.forward(jnp.asarray(g["states"]))
    np.testing.assert_allclose(stats, g["stats"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["actions"]), suff_stats=stats), g["lp"], atol=1e-4)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["single"]), states=states), g["lp1"], atol=1e-4)
    np.testing.assert_allclose(policy.mode(states=states), g["mode"], atol=1e-6)
    batch = _batch(g)
    for mode in ("i", "m"):
        for est in ("mean", "sum"):
            kl = policy.kl_divergence(state, batch, mode=f"{mode}-projection", est=est)
            np.testing.assert_allclose(kl, g[f"kl_{mode}_{est}"], rtol=1e-4, atol=1e-6)
    if "entropy" in g:
        np.testing.assert_allclose(policy.entropy(state, batch), g["entropy"], rtol=1e-5)
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    _close_grads(grads, sub(g, "grads"))
    lr = policy.likelihood_ratio(state, batch)
    np.testing.assert_allclose(lr, g["lr"], rtol=1e-4, atol=1e-6)
    lr_grads = eqx.filter_grad(lambda p: p.likelihood_ratio(state.set("policy", p), batch))(policy)
    _close_grads(lr_grads, sub(golden(f"{name}_lr"), "grads"))


@pytest.mark.parametrize("name", [n for n in POLICY_CASES if n.startswith("diag")])
def test_diagonal_policy_matches_torch(make_pendulum_mc, name):
    policy, state, g = _golden_policy(make_pendulum_mc, name)
    states, stats = jnp.asarray(g["states"]), policy.forward(jnp.asarray(g["states"]))
    np.testing.assert_allclose(stats, g["stats"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["actions"]), suff_stats=stats), g["lp"], atol=1e-4)
    np.testing.assert_allclose(policy.log_prob(jnp.asarray(g["single"]), states=states), g["lp1"], atol=1e-4)
    np.testing.assert_allclose(policy.mode(states=states), g["mode"], atol=1e-6)
    batch = _batch(g)
    for mode in ("i", "m"):
        for est in ("mean", "sum"):
            kl = policy.kl_divergence(state, batch, mode=f"{mode}-projection", est=est)
            np.testing.assert_allclose(kl, g[f"kl_{mode}_{est}"], rtol=1e-4, atol=1e-6)
    if "entropy" in g:
        np.testing.assert_allclose(policy.entropy(state, batch), g["entropy"], rtol=1e-5)
    grads = eqx.filter_grad(lambda p: p.kl_divergence(state, batch, mode="m-projection", est="mean"))(policy)
    _close_grads(grads, sub(g, "grads"))
    np.testing.assert_allclose(policy.likelihood_ratio(state, batch), g["lr"], rtol=1e-4, atol=1e-6)
    lr_grads = eqx.filter_grad(lambda p: p.likelihood_ratio(state.set("policy", p), batch))(policy)
    _close_grads(lr_grads, sub(golden(f"{name}_lr"), "grads"))


def test_v_net_matches_torch(make_pendulum_mc):
    g = golden("v_net")
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = load_named(FullyVNeuralNet(mc, jax.random.key(0)), sub(g, "params"))
    n = len(g["states"])
    z = jnp.zeros((n, 1))
    batch = Batch(jnp.asarray(g["states"]), z, z, z, jnp.asarray(g["states"]), z, {"tv": jnp.asarray(g["tv"])})
    np.testing.assert_allclose(v.forward(jnp.asarray(g["states"])), g["forward"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v.tr_obj(batch), g["obj"], rtol=1e-5)
    _close_grads(eqx.filter_grad(lambda m: m.tr_obj(batch))(v), sub(g, "grads"))
    opt = optax.adam(v.lr)
    opt_state = v.init_opt()
    for s in range(3):
        v, opt_state, _, _ = v.perform_adam_step(opt, opt_state, s, batch)
    got = named_parameters(v)
    for k, expected in sub(g, "adam3/params").items():
        np.testing.assert_allclose(got[k], expected, rtol=1e-4, atol=1e-6, err_msg=k)


def test_q_net_matches_torch(make_pendulum_mc):
    g = golden("q_net")
    # the golden Q net was built from the MPO conf (val_struct [256, 256], elu), the fixture conf is the TRPO one
    mc = make_pendulum_mc(alg="mpo", val_struct=[256, 256], val_act_fn=["elu", "elu", "none"])
    with mc["conf"].ns("val"):
        q = load_named(FullyQNeuralNet(mc, jax.random.key(0)), sub(g, "params"))
    s, a = jnp.asarray(g["states"]), jnp.asarray(g["actions"])
    z = jnp.zeros((len(s), 1))
    batch = Batch(s, a, z, z, s, z, {"tq": jnp.asarray(g["tq"])})
    np.testing.assert_allclose(q.q_val(s, a), g["q_val"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q.q_val_multi(s, jnp.asarray(g["add_actions"])), g["q_val_multi"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(q.tr_obj(batch), g["obj"], rtol=1e-5)
    _close_grads(eqx.filter_grad(lambda m: m.tr_obj(batch))(q), sub(g, "grads"))
