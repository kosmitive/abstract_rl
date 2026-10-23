import mlx.core as mx

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.misc.mixed import to_mx
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


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
        self.policy = mc.get("policy", False)
        self.q_network = mc.get("q_network", True)

        env = mc["env"]
        self.discount = env.discount()
        self.env = env

        self.ret_lambda = float(self.conf["lambda"])
        self.ret_bootstrap = self.conf.get_root("mem_bootstrap_steps")
        self.ret_add_acts = self.conf["add_acts"]

    def train(self) -> None:
        """
        Train the operator, or more specifically the q network.
        """
        self.q_network.train()

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator. The annotations stay lazy MLX
        arrays until transform_all evaluates them.
        :param trajectory: trajectory to transform.
        """

        # first rolling estimate of the exp q function
        discount = self.discount

        # get truncated length
        tl = len(trajectory)

        # obtain q and v value estimates
        states = to_mx(trajectory.states)
        actions = to_mx(trajectory.actions)
        q_vals = self.q_network.q_val(states, actions).reshape(-1)

        # expectation of the next state under the evaluated (current) policy, the same policy defines the trace
        # coefficients c; terminal states have no value
        nxt_states = to_mx(trajectory.next_states)
        dones = to_mx(trajectory.dones).reshape(-1)
        nxt_v_vals = self.q_network.v_val(nxt_states, policy=self.policy).reshape(-1) * (1 - dones)

        # get import. sampling coefficients
        c = self._def_bounded_importance_sampling_coefficients(trajectory).reshape(-1)

        # calc td errors for all t
        r = to_mx(trajectory.rewards).reshape(-1)
        td_errors_q = r + discount * nxt_v_vals - q_vals
        # Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{j=t}^{t+B-1} gamma^(j-t) (prod_{i=t+1}^{j} c_i) delta_j,
        # computed vectorized over t by iterating over the offset j - t (shifted adds, padded with zeros).
        tq = q_vals + td_errors_q
        coeff = mx.ones(tl)
        for k in range(1, min(self.ret_bootstrap, tl)):
            coeff = coeff[: tl - k] * discount * c[k:]
            tq = tq + mx.concatenate([coeff * td_errors_q[k:], mx.zeros(k)])

        trajectory["q"] = q_vals.reshape(-1, 1)
        trajectory["tq"] = tq.reshape(-1, 1)

    def _def_bounded_importance_sampling_coefficients(self, trajectory: Trajectory) -> mx.array:
        """
        Calculated the bounded importance sampling coefficients.
        :param trajectory: The trajectory for which the coefficients have to be calculated
        :return: A vector containing the coefficients.
        """

        # obtain the bounded importance sampling coefficients coefficients
        beh_lls = to_mx(trajectory.log_likelihoods).reshape([-1, 1])
        s = to_mx(trajectory.states)
        a = to_mx(trajectory.actions)

        pi_ll = self.policy.log_prob(a, states=s)
        r = mx.exp(pi_ll - beh_lls)
        c = mx.minimum(r, 1.0) * self.ret_lambda
        return c
