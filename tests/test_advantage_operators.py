import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet
from golden import golden, load_named, sub


def _ops(make_pendulum_mc):
    mc = make_pendulum_mc()
    with mc["conf"].ns("val"):
        v = load_named(FullyVNeuralNet(mc, jax.random.key(0)), sub(golden("adv_v"), "params"))
    with mc["conf"].ns("tdop"):
        td = TemporalDifferenceOperator(mc)
    with mc["conf"].ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
    return ModelState.create({}, {"v_network": (v, v)}), td, gae


def _lane(g, pad=0):
    s = jnp.asarray(g["all_states"], jnp.float32)
    n = len(s) - 1
    p = lambda x: jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), x.dtype)])  # noqa: E731
    ends = jnp.zeros(n, bool).at[-1].set(True)
    return Trajectory(
        p(s[:-1]),
        p(s[1:]),
        p(jnp.zeros((n, 1))),
        p(jnp.asarray(g["rewards"], jnp.float32)),
        p(jnp.asarray(g["rewards"][:, 0], jnp.float32)),
        p(jnp.asarray(g["dones"], jnp.float32)),
        p(jnp.zeros((n, 1))),
        p(ends),
        p(jnp.ones(n, bool)),
    )


@pytest.mark.parametrize("case", ["truncated", "terminated"])
def test_td_and_gae_match_torch(make_pendulum_mc, case):
    state, td, gae = _ops(make_pendulum_mc)
    g = golden(f"adv_{case}")
    n = len(g["rewards"])
    t = gae.transform(state, td.transform(state, _lane(g, pad=3)))
    np.testing.assert_allclose(t["tv"][:n], g["tv"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["td"][:n], g["td"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["a"][:n], g["a"], rtol=1e-5, atol=1e-5)


def test_episodes_in_one_lane_do_not_leak_into_each_other(make_pendulum_mc):
    state, td, gae = _ops(make_pendulum_mc)
    g1, g2 = golden("adv_terminated"), golden("adv_truncated")
    l1, l2 = _lane(g1), _lane(g2)
    both = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), l1, l2)
    n1 = len(g1["rewards"])
    t = gae.transform(state, td.transform(state, both))
    np.testing.assert_allclose(t["a"][:n1], g1["a"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t["a"][n1:], g2["a"], rtol=1e-5, atol=1e-5)


def test_transform_all_vmaps_over_lanes(make_pendulum_mc):
    state, td, gae = _ops(make_pendulum_mc)
    g = golden("adv_truncated")
    lanes = jax.tree.map(lambda x: jnp.stack([x, x]), _lane(g))
    tc = jax.jit(lambda s, c: gae.transform_all(s, td.transform_all(s, c)))(state, TrajectoryCollection(None, lanes))
    np.testing.assert_allclose(tc.trajectories()["a"][1], g["a"], rtol=1e-5, atol=1e-5)
