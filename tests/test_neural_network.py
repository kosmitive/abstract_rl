import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


@pytest.mark.parametrize(
    "init", ["xavier_gaussian", "xavier_uniform", "orthogonal", "gaussian", "uniform", "high", "none"]
)
def test_init_zeroes_biases_and_keeps_float32(make_pendulum_mc, init):
    net = MultilayerPerceptron(
        make_pendulum_mc(), np.array([3, 16, 2]), ["tanh", "none"], False, init, key=jax.random.key(0)
    )
    assert all(float(jnp.abs(la.bias).sum()) == 0.0 for la in net.layers)
    assert all(
        la.weight.dtype == jnp.float32 and la.weight.shape == (o, i)
        for la, (i, o) in zip(net.layers, [(3, 16), (16, 2)], strict=True)
    )


def test_xavier_gaussian_has_the_torch_variance(make_pendulum_mc):
    net = MultilayerPerceptron(
        make_pendulum_mc(), np.array([200, 300]), ["none"], False, "xavier_gaussian", key=jax.random.key(1)
    )
    assert float(jnp.std(net.layers[0].weight)) == pytest.approx(np.sqrt(2 / 500), rel=0.02)


def test_unknown_init_raises(make_pendulum_mc):
    with pytest.raises(NotImplementedError):
        MultilayerPerceptron(make_pendulum_mc(), np.array([3, 2]), ["none"], False, "foo", key=jax.random.key(0))


def test_mlp_with_per_output_activations_jits(make_pendulum_mc):
    # Hydra hands nested lists, the static structure must stay hashable
    net = MultilayerPerceptron(
        make_pendulum_mc(), np.array([3, 8, 2]), ["tanh", ["none", "softplus"]], key=jax.random.key(0)
    )
    out = jax.jit(lambda n, x: n.forward(x))(net, jnp.full((4, 3), -50.0))
    assert out.shape == (4, 2) and jnp.all(out[:, 1] >= 0)


def test_two_builds_have_equal_static_structure(make_pendulum_mc):
    mc = make_pendulum_mc()
    a = MultilayerPerceptron(mc, np.array([3, 8, 1]), ["tanh", "tanh"], key=jax.random.key(0))
    b = MultilayerPerceptron(mc, np.array([3, 8, 1]), ["tanh", "tanh"], key=jax.random.key(1))
    assert jax.tree.structure(a) == jax.tree.structure(b)
