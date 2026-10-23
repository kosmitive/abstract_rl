"""The MLX features the port relies on, and the gap which keeps the full covariance policy out of it."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten


def _flat_mlp() -> tuple[nn.Module, mx.array, mx.array]:
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
    mx.eval(model.parameters())
    x = mx.random.normal((64, 3))
    flat = mx.concatenate([v.reshape(-1) for _, v in tree_flatten(model.parameters())])
    return model, x, flat


def _unflatten(model: nn.Module, flat: mx.array) -> None:
    items, i = [], 0
    for k, v in tree_flatten(model.parameters()):
        items.append((k, flat[i : i + v.size].reshape(v.shape)))
        i += v.size
    model.update(tree_unflatten(items))


def test_jvp_of_grad_is_the_hessian_vector_product():
    # forward over reverse (used for the Fisher vector products) equals reverse over reverse
    model, x, flat = _flat_mlp()

    def loss(p: mx.array) -> mx.array:
        _unflatten(model, p)
        return mx.mean(model(x) ** 2)

    v = mx.random.normal(flat.shape, key=mx.random.key(1))
    _, (fwd,) = mx.jvp(mx.grad(loss), [flat], [v])
    _, (rev,) = mx.vjp(lambda p: mx.grad(loss)(p), [flat], [v])
    np.testing.assert_allclose(np.array(fwd), np.array(rev), rtol=1e-4, atol=1e-5)


def test_adam_with_bias_correction_matches_the_reference_update():
    w0 = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    grads = [np.array([0.1, -0.2, 0.3], np.float32), np.array([-0.4, 0.1, 0.2], np.float32)]
    opt = optim.Adam(learning_rate=0.01, bias_correction=True)
    params = {"w": mx.array(w0)}
    opt.init(params)
    m = v = np.zeros(3)
    w = w0.astype(np.float64)
    for t, g in enumerate(grads, 1):
        params = opt.apply_gradients({"w": mx.array(g)}, params)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - 0.01 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    np.testing.assert_allclose(np.array(params["w"]), w, rtol=1e-5)


def test_compiled_fixed_length_loop_with_frozen_iterations():
    # MLX has no while loop, iterations after convergence are masked out with where
    def halve_until(x: mx.array) -> mx.array:
        for _ in range(20):
            x = mx.where(x > 1.0, x / 2, x)
        return x

    x = mx.array([100.0, 3.0, 0.5])
    np.testing.assert_allclose(np.array(mx.compile(halve_until)(x)), np.array(halve_until(x)))
    np.testing.assert_allclose(np.array(halve_until(x)), [0.78125, 0.75, 0.5])


def test_compile_with_module_state_as_input_and_output():
    # the parameters and the optimizer state are implicit inputs and outputs of the compiled step

    def run(compiled: bool) -> list[float]:
        mx.random.seed(3)
        model = nn.Linear(2, 1)
        opt = optim.Adam(learning_rate=0.01, bias_correction=True)
        opt.init(model.trainable_parameters())
        x, y = mx.random.normal((8, 2)), mx.random.normal((8, 1))

        def step(x: mx.array, y: mx.array) -> mx.array:
            loss, g = nn.value_and_grad(model, lambda x, y: nn.losses.mse_loss(model(x), y))(x, y)
            opt.update(model, g)
            return loss

        state = [model.state, opt.state]
        fn = mx.compile(step, inputs=state, outputs=state) if compiled else step
        return [fn(x, y).item() for _ in range(10)]

    np.testing.assert_allclose(run(True), run(False), rtol=1e-5)
    assert run(True)[-1] < run(True)[0]


def test_scatter_add_accumulates_duplicate_indices():
    idx = mx.array([0, 2, 2, 2])
    out = mx.zeros(3).at[idx].add(mx.array([1.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.array(out), [1.0, 0.0, 6.0])


def test_cholesky_has_no_vjp():
    """The full covariance policy needs the gradient of a Cholesky factorization, MLX (0.32) has none."""
    a = mx.array([[2.0, 0.5], [0.5, 1.0]])

    def f(a: mx.array) -> mx.array:
        return mx.linalg.cholesky(a, stream=mx.cpu).sum()

    try:
        mx.grad(f)(a)
    except ValueError:
        return
    pytest.skip("this MLX version differentiates the Cholesky factorization, the full covariance could be ported")
