import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch, masked_mean, masked_reduce, masked_std


class _Lin(eqx.Module):
    w: jax.Array

    def sync(self, v):
        return self


def test_masked_reductions_match_unpadded():
    x = jnp.arange(5.0)
    padded = jnp.concatenate([x, jnp.array([100.0, -7.0])])
    mask = jnp.arange(7) < 5
    assert masked_mean(padded, mask) == pytest.approx(float(jnp.mean(x)))
    assert masked_std(padded, mask) == pytest.approx(float(jnp.std(x, ddof=1)), rel=1e-6)
    assert masked_reduce(padded, mask, "sum") == pytest.approx(10.0)
    assert masked_mean(x, None) == pytest.approx(2.0)
    with pytest.raises(NotImplementedError):
        masked_reduce(x, None, "max")


def test_masked_reductions_ignore_nan_padding():
    padded = jnp.array([1.0, 2.0, 3.0, jnp.nan, jnp.inf])
    mask = jnp.array([True, True, True, False, False])
    assert jnp.isfinite(masked_mean(padded, mask)) and jnp.isfinite(masked_std(padded, mask))
    grad = jax.grad(lambda p: masked_mean(p * 2.0, mask))(padded)
    assert jnp.all(jnp.isfinite(grad[:3])) and jnp.all(grad[3:] == 0)


def test_batch_keeps_the_torch_accessors():
    s = jnp.ones((4, 3))
    b = Batch(s, jnp.zeros((4, 1)), jnp.zeros((4, 1)), jnp.zeros((4, 1)), s, jnp.zeros((4, 1)), {"a": jnp.arange(4.0)})
    assert len(b) == 4 and b.states.shape == (4, 3) and b["a"].shape == (4,) and b.mask is None
    assert b.log_likelihoods.shape == (4, 1) and b.dones.shape == (4, 1) and b.next_states.shape == (4, 3)
    jax.jit(lambda bb: bb["a"].sum())(b)  # a batch is a pytree


def test_get_set_and_target_access():
    st = ModelState.create({"opt": jnp.zeros(2)}, {"policy": (_Lin(jnp.ones(2)), _Lin(jnp.zeros(2)))})
    assert float(st.get("policy").w[0]) == 1.0 and float(st.get("policy", True).w[0]) == 0.0
    assert st["opt"].shape == (2,)
    st2 = st.set("policy", _Lin(jnp.full(2, 5.0))).set("opt", jnp.ones(2))
    assert float(st2.get("policy").w[0]) == 5.0 and float(st2.get("policy", True).w[0]) == 0.0
    assert float(st2["opt"][0]) == 1.0 and float(st.get("policy").w[0]) == 1.0  # immutable


def test_sync_respects_sync_steps():
    online, target = _Lin(jnp.ones(1)), _Lin(jnp.zeros(1))
    st = ModelState.create({}, {"a": (online, target), "b": (online, target)}, {"a": 1, "b": 2})
    seen = []
    for k in range(4):
        st = st.sync()
        seen.append((float(st.get("a", True).w[0]), float(st.get("b", True).w[0])))
        st = st.set("a", _Lin(jnp.full(1, k + 2.0))).set("b", _Lin(jnp.full(1, k + 2.0)))
    # ModelConfiguration.sync: copy when overall_step % s == 0, then increment overall_step
    assert seen == [(1.0, 1.0), (2.0, 1.0), (3.0, 3.0), (4.0, 3.0)]
    assert int(st.overall_step) == 4


def test_sync_is_jittable_and_vmappable():
    st = ModelState.create({}, {"a": (_Lin(jnp.ones((3, 2))), _Lin(jnp.zeros((3, 2))))})
    st = eqx.tree_at(lambda s: s.overall_step, st, jnp.zeros(3, jnp.int32))
    out = jax.jit(jax.vmap(lambda s: s.sync()))(st)
    np.testing.assert_allclose(out.get("a", True).w, 1.0)


def test_model_configuration_exports_state(make_pendulum_mc):
    mc = make_pendulum_mc()
    # add_syncable needs a Syncable model (Task 7+), a main element exercises the export
    mc.add_main("opt", jnp.zeros(3))
    st = mc.state()
    assert "opt" in st.data and "conf" not in st.data and "log" not in st.data
    mc.load_state(st.set("opt", jnp.ones(3)))
    assert float(mc["opt"][0]) == 1.0
