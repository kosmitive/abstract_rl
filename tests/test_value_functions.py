import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


class _StubCollection:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, key, batch_size, copy_fields):
        return self.batch


def _v_batch(n=32):
    s = jax.random.normal(jax.random.key(0), (n, 3))
    z = jnp.zeros((n, 1))
    return Batch(s, z, z, z, s, z, {"tv": jnp.sin(s[:, 0]) * 3})


def test_perform_mult_sgd_fits_and_threads_the_state(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = FullyVNeuralNet(mc, jax.random.key(0))
    state = ModelState.create({"v_network_opt": v.init_opt()}, {"v_network": (v, v)})
    batch = _v_batch()
    before = float(v.tr_obj(batch))
    step = jax.jit(lambda st, k: st.get("v_network").perform_mult_sgd(st, k, 50, _StubCollection(batch), 32, ["tv"]))
    state, metrics = step(state, jax.random.key(1))
    assert float(state.get("v_network").tr_obj(batch)) < before
    assert metrics["k_pre_post"].shape == (50, 2)
    assert jnp.all(jnp.isnan(metrics["k_pre_post"][1:10, 1])) and jnp.isfinite(metrics["k_pre_post"][10, 1])
    # the target is untouched by the fit
    np.testing.assert_allclose(state.get("v_network", True).net.layers[0].weight, v.net.layers[0].weight)


def test_q_val_multi_keeps_the_repeat_interleave_order(make_pendulum_mc):
    mc = make_pendulum_mc(alg="mpo", val_struct=[256, 256], val_act_fn=["elu", "elu", "none"])
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc, jax.random.key(0))
    states = jax.random.normal(jax.random.key(1), (5, 3))
    actions = jax.random.uniform(jax.random.key(2), (5, 4), minval=-2, maxval=2)
    multi = q.q_val_multi(states, actions)
    single = jnp.stack([q.q_val(states, actions[:, k : k + 1])[:, 0] for k in range(4)], axis=1)
    np.testing.assert_allclose(multi, single, atol=1e-6)
