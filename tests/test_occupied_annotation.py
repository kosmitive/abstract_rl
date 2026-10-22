import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_mpo_non_parametric import annotate_chunked
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet

CAPACITY = 600
FIELDS = ("add_act", "add_q", "add_pi")


def _rollout(key, lengths, obs_dim):
    """A one-lane rollout of whole episodes of the given lengths, padded to their total plus 7 rows."""
    n = sum(lengths)
    length = n + 7
    ends = np.zeros(length, bool)
    ends[np.cumsum(lengths) - 1] = True
    mask = np.arange(length) < n
    k1, k2, k3 = jax.random.split(key, 3)
    z = jnp.zeros((1, length, 1))
    traj = Trajectory(
        jax.random.normal(k1, (1, length, obs_dim)),
        jax.random.normal(k2, (1, length, obs_dim)),
        jax.random.uniform(k3, (1, length, 1), minval=-2.0, maxval=2.0),
        z,
        jnp.zeros((1, length)),
        z,
        z,
        jnp.asarray(ends)[None],
        jnp.asarray(mask)[None],
        {"time": jnp.zeros((1, length), jnp.int32)},
    )
    return TrajectoryCollection(None, traj)


def _models(mc):
    with mc["conf"].ns("policy"):
        policy = select_policy(mc, jax.random.key(0))
    with mc["conf"].ns("val"):
        q = FullyQNeuralNet(mc, jax.random.key(1))
    return ModelState.create({}, {"policy": (policy, policy), "q_network": (q, q)})


def test_occupied_rows_track_the_highest_write():
    memory = TrajectoryCollection.memory(None, CAPACITY, 3, 1)
    assert int(memory.occupied_rows()) == 0
    for lengths, expected in [([100, 100], 200), ([150], 350), ([200], 550), ([120], 550)]:
        memory = memory.merge(_rollout(jax.random.key(len(lengths)), lengths, 3), 10_000)
        assert int(memory.occupied_rows()) == expected
    # the last insert wrapped to the front: every valid row lies below the high-water mark
    assert not np.any(np.asarray(memory.trajectories().mask[0, 550:]))


def test_chunked_and_full_resample_agree_on_valid_rows(make_pendulum_mc):
    mc = make_pendulum_mc(alg="mpo", res_add_acts=20, val_struct=[32, 32], val_act_fn=["elu", "elu", "none"])
    state = _models(mc)
    with mc["conf"].ns("res"):
        res = ResampleOperator(mc)

    # three merges fill about a third of the memory
    memory = TrajectoryCollection.memory(None, CAPACITY, 3, 1)
    for i, lengths in enumerate([[40, 30], [50, 25], [45]]):
        memory = memory.merge(_rollout(jax.random.key(10 + i), lengths, 3), 10_000)
    assert int(memory.occupied_rows()) == 190
    template = jax.eval_shape(lambda m: res.transform_all(state, m, jax.random.key(0)), memory)
    memory = memory.with_annotations(
        {k: jax.ShapeDtypeStruct(template.trajectories()[k].shape[2:], jnp.float32) for k in FIELDS}
    )

    key = jax.random.key(3)
    lane = jax.tree.map(lambda x: x[0], memory.trajectories())
    # both jitted: eager and jitted log-likelihoods differ by up to 4e-5 near the squash bound
    full = jax.jit(lambda lane: res.transform(state, lane, key))(lane)
    chunked = jax.jit(
        lambda m: annotate_chunked(lambda part, k, rows: res.transform(state, part, k, rows), m, key, 64)
    )(memory)
    chunked = jax.tree.map(lambda x: x[0], chunked.trajectories())

    valid = np.asarray(lane.mask)
    for k in FIELDS:
        np.testing.assert_allclose(np.asarray(chunked[k])[valid], np.asarray(full[k])[valid], atol=1e-6)
        # rows beyond the occupied ones keep their previous (zero) annotations
        assert not np.any(np.asarray(chunked[k])[190:])
