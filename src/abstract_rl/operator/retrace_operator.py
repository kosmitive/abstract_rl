from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


def _shift(x: Array, k: int) -> Array:
    """x[t + k], zero beyond the end."""
    return jnp.concatenate([x[k:], jnp.zeros((k, *x.shape[1:]), x.dtype)])


class RetraceOperator(TrajectoryOperator):
    """
    Represents the retrace operator from https://arxiv.org/abs/1606.02647.
    """

    def __repr__(self) -> str:
        return "retrace"

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new retrace operator.
        :param mc: The model configuration with env and so on.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get("conf")
        self.conf = conf
        self.mc = mc

        env = mc["env"]
        self.discount = env.discount()
        self.env = env

        self.ret_lambda = float(self.conf["lambda"])
        self.ret_bootstrap = self.conf.get_root("mem_bootstrap_steps")
        self.ret_add_acts = self.conf["add_acts"]

    def transform(self, state: ModelState, trajectory: Trajectory, key: Any = None) -> Trajectory:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """
        policy = state.get("policy", False)
        q_network = state.get("q_network", True)

        # first rolling estimate of the exp q function
        discount = self.discount
        tl = len(trajectory)

        # obtain q and v value estimates
        states = trajectory.states
        actions = trajectory.actions
        q_vals = q_network.q_val(states, actions).reshape(-1)

        # expectation of the next state under the evaluated (current) policy, the same policy defines the trace
        # coefficients c; terminal states have no value
        nxt_states = trajectory.next_states
        dones = trajectory.dones.reshape(-1)
        nxt_v_vals = q_network.v_val(state, key, nxt_states, policy=policy).reshape(-1) * (1 - dones)

        # get import. sampling coefficients
        c = self._def_bounded_importance_sampling_coefficients(state, trajectory).reshape(-1)

        # calc td errors for all t
        r = trajectory.rewards.reshape(-1)
        td_errors_q = r + discount * nxt_v_vals - q_vals
        # Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{j=t}^{t+B-1} gamma^(j-t) (prod_{i=t+1}^{j} c_i) delta_j, computed
        # vectorized over t by iterating over the offset j - t; the sum stops at the end of the episode of t
        episode = jnp.cumsum(trajectory.ends) - trajectory.ends
        tq = q_vals + td_errors_q
        coeff = jnp.ones(tl)
        for k in range(1, min(self.ret_bootstrap, tl)):
            same_episode = (_shift(episode, k) == episode) & _shift(trajectory.mask, k)
            coeff = coeff * discount * _shift(c, k) * same_episode
            tq = tq + coeff * _shift(td_errors_q, k)

        return trajectory.set("q", q_vals.reshape(-1, 1)).set("tq", tq.reshape(-1, 1))

    def _def_bounded_importance_sampling_coefficients(self, state: ModelState, trajectory: Trajectory) -> Array:
        """
        Calculated the bounded importance sampling coefficients.
        :param trajectory: The trajectory for which the coefficients have to be calculated
        :return: A vector containing the coefficients.
        """
        policy = state.get("policy", False)

        # obtain the bounded importance sampling coefficients coefficients
        beh_lls = trajectory.log_likelihoods.reshape([-1, 1])
        pi_ll = policy.log_prob(trajectory.actions, states=trajectory.states)
        r = jnp.exp(pi_ll - beh_lls)
        c = jnp.minimum(r, 1.0) * self.ret_lambda
        return c
