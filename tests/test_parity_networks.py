import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from golden import golden, load_named, sub

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
