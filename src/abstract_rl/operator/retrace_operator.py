from collections.abc import Callable

import mlx.core as mx
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import (
    TrajectoryOperator,
    compile_for_device,
    pad_np,
    pad_rows,
    padded_size,
    run_in_chunks,
    stack_field,
    store_rows,
)

# number of actions sampled per next state for its expected value (the default of FullyQNeuralNet.v_val)
EXP_ACTIONS = 10


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
        self._rows: Callable[..., tuple[mx.array, ...]] | None = None
        self._targets: Callable[..., mx.array] | None = None

    def train(self) -> None:
        """
        Train the operator, or more specifically the q network.
        """
        self.q_network.train()

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """
        self.transform_all([trajectory])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """
        Annotates all trajectories with the Q values (q) and the Retrace targets (tq). The trajectories are
        concatenated: the network evaluations run in calls of chunk_states() states, the bootstrap sum in one
        compiled call over all rows, where the trace of a row stops at the end of its trajectory. The noise of the
        expected next-state values is drawn per trajectory as before, so the result is the same as with one call per
        trajectory, up to float rounding.
        """
        target = trajectories
        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()
        n = sum(len(t) for t in trajectories)
        n_pad = padded_size(n)
        noise = pad_rows(mx.concatenate([self.policy.sample_noise(len(t), EXP_ACTIONS) for t in trajectories]), n_pad)
        fields = ["states", "actions", "next_states", "dones", "rewards", "log_likelihoods"]
        s, a, nxt_s, dones, r, beh_lls = (stack_field(trajectories, f, n_pad) for f in fields)
        q_vals, nxt_v_vals, c = run_in_chunks(self._rows_fn(), s, a, nxt_s, dones, beh_lls, noise)

        # rows of the same trajectory share its index, the padding rows get -1
        seg = mx.array(
            np.concatenate(
                [np.repeat(np.arange(len(trajectories)), [len(t) for t in trajectories]), np.full(n_pad - n, -1)]
            ).astype(np.int32)
        )
        tq = self._targets_fn()(q_vals, nxt_v_vals, c, r.reshape(-1), seg)
        store_rows(target, {"q": q_vals[:n].reshape(-1, 1), "tq": tq[:n].reshape(-1, 1)})

    def targets_at(self, tc: TrajectoryCollection, indices: np.ndarray) -> tuple[mx.array, mx.array]:
        """
        The Q values (q) and the Retrace targets (tq) of transform_all for the rows at the sorted unique indices of
        the concatenated trajectories only. The trace of a row needs the next mem_bootstrap_steps - 1 rows of its
        trajectory, so these rows are evaluated too, as runs of consecutive rows: a trace stops at the end of its
        run, which is the end of the trajectory for every row of indices. The noise of the expected next-state
        values is drawn for all evaluated rows at once.
        :return: q and tq of the rows at indices, as [N, 1] arrays.
        """
        ends = tc.trajectory_ends()
        row_ends = ends[np.searchsorted(ends, indices, side="right")]
        window = np.arange(max(self.ret_bootstrap, 1))
        rows = (indices[:, None] + window[None, :]).reshape(-1)
        rows = np.unique(rows[rows < np.repeat(row_ends, len(window))])
        n = len(rows)
        n_pad = padded_size(n)
        batch = tc.batch_at(rows)
        noise = pad_rows(self.policy.sample_noise(n, EXP_ACTIONS), n_pad)
        fields = [batch.states, batch.actions, batch.next_states, batch.dones, batch.rewards, batch.log_likelihoods]
        s, a, nxt_s, dones, r, beh_lls = (pad_np(f, n_pad) for f in fields)
        q_vals, nxt_v_vals, c = run_in_chunks(self._rows_fn(), s, a, nxt_s, dones, beh_lls, noise)

        # consecutive rows of the same trajectory share a run index, the padding rows get -1
        run_start = np.ones(n, dtype=bool)
        run_start[1:] = (rows[1:] != rows[:-1] + 1) | np.isin(rows[1:], ends)
        seg = mx.array(np.concatenate([np.cumsum(run_start), np.full(n_pad - n, -1)]).astype(np.int32))
        tq = self._targets_fn()(q_vals, nxt_v_vals, c, r.reshape(-1), seg)
        pos = mx.array(np.searchsorted(rows, indices).astype(np.int32))
        return q_vals[pos].reshape(-1, 1), tq[pos].reshape(-1, 1)

    def _rows_fn(self) -> Callable[..., tuple[mx.array, ...]]:
        """Compiled per-row values: Q(s_t, a_t), the expected value of s_t+1 (0 if done) and the trace coefficient."""
        if self._rows is None:
            policy, q_network, ret_lambda = self.policy, self.q_network, self.ret_lambda

            def rows(
                s: mx.array, a: mx.array, nxt_s: mx.array, dones: mx.array, beh_lls: mx.array, noise: mx.array
            ) -> tuple[mx.array, ...]:
                q_vals = q_network.q_val(s, a).reshape(-1)
                # expectation of the next state under the evaluated (current) policy, the same policy defines the
                # trace coefficients c; terminal states have no value
                nxt_v_vals = q_network.v_val(nxt_s, policy=policy, noise=noise).reshape(-1) * (1 - dones.reshape(-1))
                # bounded importance sampling coefficients
                pi_ll = policy.log_prob(a, states=s)
                c = (mx.minimum(mx.exp(pi_ll - beh_lls.reshape([-1, 1])), 1.0) * ret_lambda).reshape(-1)
                return q_vals, nxt_v_vals, c

            self._rows = compile_for_device(rows, [policy.state, q_network.state])
        return self._rows

    def _targets_fn(self) -> Callable[..., mx.array]:
        """Compiled Retrace targets of all concatenated rows."""
        if self._targets is None:
            discount, bootstrap = self.discount, self.ret_bootstrap

            def targets(q_vals: mx.array, nxt_v_vals: mx.array, c: mx.array, r: mx.array, seg: mx.array) -> mx.array:
                # Q_ret(s_t, a_t) = Q(s_t, a_t) + sum_{j=t}^{t+B-1} gamma^(j-t) (prod_{i=t+1}^{j} c_i) delta_j,
                # computed vectorized over t by iterating over the offset j - t (shifted adds, padded with zeros);
                # the coefficient is 0 once j is in another trajectory
                n = q_vals.shape[0]
                td_errors_q = r + discount * nxt_v_vals - q_vals
                tq = q_vals + td_errors_q
                coeff = mx.ones(n)
                for k in range(1, min(bootstrap, n)):
                    coeff = coeff[: n - k] * discount * c[k:] * (seg[: n - k] == seg[k:])
                    tq = tq + mx.concatenate([coeff * td_errors_q[k:], mx.zeros(k)])
                return tq

            self._targets = compile_for_device(targets, [])
        return self._targets
