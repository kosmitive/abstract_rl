import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.misc.running_centered_filter import RunningCenteredFilter
from abstract_rl.misc.running_mean_std import RunningMeanStd
from golden import golden


def test_running_centered_filter_matches_torch_sequence():
    g = golden("filter")
    f = RunningCenteredFilter("states", 3)
    for i, x in enumerate(g["xs"]):
        f = f.register(jnp.asarray(x))
        np.testing.assert_allclose(f(jnp.asarray(x)), g["outs"][i], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(f.mean, g["means"][i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f.std, g["stds"][i], rtol=1e-4, atol=1e-6)


def test_running_centered_filter_masked_rows_are_skipped_in_order():
    xs = jnp.asarray(np.random.default_rng(0).normal(size=(6, 2)), jnp.float32)
    mask = jnp.array([True, False, True, True, False, True])
    masked = RunningCenteredFilter("s", 2).register(xs, mask)
    plain = RunningCenteredFilter("s", 2).register(xs[np.asarray(mask)])
    np.testing.assert_allclose(masked.mean, plain.mean, rtol=1e-6)
    np.testing.assert_allclose(masked.var, plain.var, rtol=1e-5)
    assert int(masked._n) == 4
    jax.jit(lambda f, x, m: f.register(x, m))(RunningCenteredFilter("s", 2), xs, mask)


def test_running_mean_std_matches_gymnasium_sequence():
    g = golden("obsnorm")
    rms = RunningMeanStd.create(3)
    for i, x in enumerate(g["xs"]):
        rms = rms.update(jnp.asarray(x)[None])
        np.testing.assert_allclose(rms.mean, g["means"][i], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(rms.var, g["vars"][i], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(rms.normalize(jnp.asarray(x)), g["outs"][i], rtol=1e-4, atol=1e-5)


def test_running_mean_std_batch_update_equals_sequential_and_ignores_masked_rows():
    xs = jnp.asarray(np.random.default_rng(1).normal(size=(5, 3)) * 3 + 1, jnp.float32)
    seq = RunningMeanStd.create(3)
    for x in xs[:4]:
        seq = seq.update(x[None])
    batch = RunningMeanStd.create(3).update(xs, jnp.array([True, True, True, True, False]))
    np.testing.assert_allclose(batch.mean, seq.mean, rtol=1e-5)
    np.testing.assert_allclose(batch.var, seq.var, rtol=1e-4)
    np.testing.assert_allclose(batch.count, seq.count, rtol=1e-6)
    none = RunningMeanStd.create(3).update(xs, jnp.zeros(5, bool))
    np.testing.assert_allclose(none.mean, 0.0) and np.testing.assert_allclose(none.var, 1.0)
