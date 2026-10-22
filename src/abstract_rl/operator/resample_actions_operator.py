from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


class ResampleOperator(TrajectoryOperator):
    """
    Samples for each state additional actionsa and equips
    """

    # the annotations written by ``transform``
    FIELDS = ("add_act", "add_q", "add_pi")

    def __repr__(self) -> str:
        return "resample"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new retrace operator.
        :param mc: The model configuration with env and so on.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.add_act = conf["add_acts"]
        self.mc = mc

        env = mc["env"]
        self.discount = env.discount()

    def transform(
        self, state: ModelState, trajectory: Trajectory, key: Any = None, rows: Int[Array, " L"] | None = None
    ) -> Trajectory:
        """
        Transform a trajectory with the current instance of the operator.
        :param rows: The row index of every step in its lane (default ``0..L-1``). The actions of a row are sampled
            with ``fold_in(key, row)``, so annotating a lane in chunks draws the same actions as annotating it whole.
        """
        policy = state.get("policy", True)
        q_net = state.get("q_network", True)

        # sample actions for sample based, with one key per row
        t_states = trajectory.states
        f_stats = policy.forward(t_states)
        rows = jnp.arange(len(trajectory)) if rows is None else rows
        keys = jax.vmap(jax.random.fold_in, (None, 0))(key, rows)
        sampled_actions = jax.vmap(
            lambda k, s: policy.sample_actions(k, suff_stats=s[None], num_actions=self.add_act)[0]
        )(keys, f_stats)
        ll = policy.log_prob(sampled_actions, suff_stats=f_stats)
        q_vals = q_net.q_val_multi(t_states, sampled_actions)
        return trajectory.set("add_act", sampled_actions).set("add_q", q_vals).set("add_pi", ll)
