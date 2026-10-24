"""The parameter buffer is one allocation, viewed by MLX and by JAX."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mx = pytest.importorskip("mlx.core")

from abstract_rl.interop.shared import SharedParams, buffer_address  # noqa: E402


def test_jax_views_the_mlx_buffer_and_sees_writes():
    """``device_put`` of the aligned memoryview keeps the address, and a write is visible on both sides."""
    shared = SharedParams(64)
    shared.write_numpy(np.arange(64, dtype=np.float32))
    mlx = shared.as_mlx()
    placed = shared.as_jax()
    assert buffer_address(mlx) % 64 == 0
    assert buffer_address(placed) == buffer_address(mlx)
    assert buffer_address(shared.numpy_view()) == buffer_address(mlx)
    shared.numpy_view()[0] = 7
    assert float(np.array(mlx)[0]) == 7
    assert float(np.array(placed)[0]) == 7
    # a NumPy write is what MLX kernels read
    assert float(np.array(mx.sum(mlx))) == float(np.arange(64).sum() - 0 + 7)


def test_jit_reads_the_buffer_again_instead_of_folding_it():
    """The vector is an argument. A write between two calls is visible to the second call."""
    shared = SharedParams(4)
    shared.write_numpy(np.array([1, 2, 3, 4], np.float32))
    placed = shared.as_jax()

    @jax.jit
    def total(theta):
        return theta.sum()

    assert float(total(placed)) == 10
    shared.numpy_view()[:] = 10
    assert float(total(placed)) == 40


def test_a_fresh_mlx_vector_is_copied_back_into_the_same_address():
    """Functional updates allocate. The copy lands in the original buffer and the view stays put."""
    shared = SharedParams(4)
    shared.write_numpy(np.zeros(4, np.float32))
    address = buffer_address(shared.as_mlx())
    fresh = mx.array(np.array([1, 2, 3, 4], np.float32))
    assert buffer_address(fresh) != address
    assert shared.write_from_mlx(fresh)
    assert buffer_address(shared.as_mlx()) == address
    np.testing.assert_allclose(shared.numpy_view(), [1, 2, 3, 4])
    assert not shared.write_from_mlx(shared.as_mlx())


def test_replicated_host_devices_keep_the_pointer():
    """Each host device's replica is a view when JAX has more than one CPU device."""
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("this process has one JAX CPU device; the sharded test covers several")
    shared = SharedParams(32)
    shared.write_numpy(np.arange(32, dtype=np.float32))
    placed = shared.replicated(devices)
    want = buffer_address(shared.as_mlx())
    for shard in placed.addressable_shards:
        assert buffer_address(shard.data) == want
    shared.numpy_view()[1] = 5
    assert float(np.array(placed.addressable_shards[-1].data)[1]) == 5
