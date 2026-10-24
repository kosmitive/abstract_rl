from collections.abc import Callable

import mlx.core as mx

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.operator.trajectory_operator import (
    TrajectoryOperator,
    compile_for_device,
    pad_rows,
    padded_size,
    run_in_chunks,
    stack_field,
    store_rows,
)


class ResampleOperator(TrajectoryOperator):
    """
    Samples for each state additional actionsa and equips
    """

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
        self.policy = mc.get("policy", True)
        self.q_net = mc.get("q_network", True)

        env = mc["env"]
        self.discount = env.discount()
        self._annotate: Callable[..., tuple[mx.array, ...]] | None = None

    def transform(self, trajectory: Trajectory) -> None:
        """
        Transform a trajectory with the current instance of the operator.
        """
        self.transform_all([trajectory])

    def transform_all(self, trajectories: TrajectoryCollection | list[Trajectory]) -> None:
        """
        Samples add_acts actions for every state of all trajectories and annotates them with the actions (add_act),
        their log likelihood (add_pi) and their target Q value (add_q). The trajectories are concatenated and run in
        calls of chunk_states() states. The noise is drawn per trajectory as before, so the samples are the
        same as with one call per trajectory, up to float rounding.
        """
        if isinstance(trajectories, TrajectoryCollection):
            trajectories = trajectories.trajectories()
        n = sum(len(t) for t in trajectories)
        n_pad = padded_size(n)
        noise = pad_rows(mx.concatenate([self.policy.sample_noise(len(t), self.add_act) for t in trajectories]), n_pad)
        states = stack_field(trajectories, "states", n_pad)
        actions, log_pi, q_vals = run_in_chunks(self._annotate_fn(), states, noise)
        store_rows(trajectories, {"add_act": actions[:n], "add_q": q_vals[:n], "add_pi": log_pi[:n]})

    def _annotate_fn(self) -> Callable[..., tuple[mx.array, ...]]:
        if self._annotate is None:
            policy, q_net, k = self.policy, self.q_net, self.add_act

            def annotate(states: mx.array, noise: mx.array) -> tuple[mx.array, ...]:
                f_stats = policy.forward(states)
                sampled_actions = policy.sample_actions(suff_stats=f_stats, num_actions=k, noise=noise)
                ll = policy.log_prob(sampled_actions, suff_stats=f_stats)
                return sampled_actions, ll, q_net.q_val_multi(states, sampled_actions)

            # the parameters are inputs of the compiled function, so it stays valid after every update
            self._annotate = compile_for_device(annotate, [policy.state, q_net.state])
        return self._annotate
