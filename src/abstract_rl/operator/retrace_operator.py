import torch

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory import Trajectory
from abstract_rl.operator.trajectory_operator import TrajectoryOperator


def bounded_importance_coefficients(
    pi_log: torch.Tensor, behavior_log: torch.Tensor, trace_lambda: float
) -> torch.Tensor:
    """Retrace coefficients ``c = lambda * min(1, pi / mu)``.

    Args:
        pi_log: Log density of the evaluation policy, shape ``[T]`` or ``[T, 1]``.
        behavior_log: Log density of the behavior policy, same shape.
        trace_lambda: The Retrace lambda in ``[0, 1]``.

    Returns:
        Coefficients of shape ``[T]``.
    """

    ratio = torch.exp(pi_log.view(-1) - behavior_log.view(-1))
    return torch.minimum(ratio, torch.ones_like(ratio)) * float(trace_lambda)


def retrace_targets(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    coefficients: torch.Tensor,
    discount: float,
    horizon: int,
) -> torch.Tensor:
    """Retrace return truncated after ``horizon`` TD errors.

    ``Q_ret_t = Q_t + sum_k discount^k * (prod_{i=1..k} c_{t+i}) * delta_{t+k}`` with
    ``delta_t = r_t + discount * (1 - done_t) * V(s_{t+1}) - Q_t``. The product starts at ``c_{t+1}``, so ``c_t``
    does not multiply the TD error at time ``t``.

    Args:
        q_values: ``Q(s_t, a_t)``, shape ``[T]``.
        rewards: Rewards, shape ``[T]``.
        next_values: ``V(s_{t+1})``, shape ``[T]``. Terminations are masked inside.
        dones: One when the transition terminated, shape ``[T]``.
        coefficients: Trace coefficients ``c_t`` (already scaled by lambda), shape ``[T]``.
        discount: Discount factor.
        horizon: Number of TD errors, including the immediate one.

    Returns:
        Retrace targets of shape ``[T]``.
    """

    q_values = q_values.view(-1)
    rewards = rewards.view(-1)
    next_values = next_values.view(-1)
    dones = dones.view(-1)
    coefficients = coefficients.view(-1)
    length = int(q_values.shape[0])
    delta = rewards + float(discount) * next_values * (1.0 - dones) - q_values
    targets = q_values + delta
    coeff = torch.ones(length, dtype=q_values.dtype, device=q_values.device)
    for offset in range(1, min(int(horizon), length)):
        coeff = coeff[: length - offset] * float(discount) * coefficients[offset:]
        targets[: length - offset] = targets[: length - offset] + coeff * delta[offset:]
    return targets


def n_step_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    discount: float,
    n_step: int,
) -> torch.Tensor:
    """n-step return bootstrapping from ``V(s_{t+n})``.

    A termination cuts the sum. If the stored trajectory ends first, the backup uses the last available value
    ``V(s_T)``, which is how truncated episodes keep their bootstrap.

    Args:
        rewards: Rewards ``r_t``, shape ``[T]``.
        next_values: ``V(s_{t+1})``, shape ``[T]``.
        dones: Termination flags, shape ``[T]``.
        discount: Discount factor.
        n_step: Number of rewards in the sum. One is the one-step return.

    Returns:
        Targets of shape ``[T]``.
    """

    rewards = rewards.view(-1)
    next_values = next_values.view(-1)
    dones = dones.view(-1)
    length = int(rewards.shape[0])
    targets = torch.empty(length, dtype=rewards.dtype, device=rewards.device)
    for t in range(length):
        discounted = rewards.new_zeros(())
        scale = rewards.new_ones(())
        finished = False
        for k in range(int(n_step)):
            index = t + k
            if index >= length:
                if k > 0:
                    discounted = discounted + scale * next_values[index - 1]
                finished = True
                break
            discounted = discounted + scale * rewards[index]
            scale = scale * float(discount)
            if float(dones[index]) > 0.5:
                finished = True
                break
        if not finished:
            discounted = discounted + scale * next_values[t + int(n_step) - 1]
        targets[t] = discounted
    return targets


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
        self.ret_bootstrap = int(self.conf.get_root("mem_bootstrap_steps"))
        self.ret_add_acts = int(self.conf["add_acts"])
        self.kind = str(self.conf.get("operator", "retrace"))

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

        with torch.no_grad():
            states = torch.as_tensor(trajectory.states, dtype=torch.float32)
            actions = torch.as_tensor(trajectory.actions, dtype=torch.float32)
            q_vals = self.q_network.q_val(states, actions).view(-1)

            # V(s_{t+1}) is the expectation of the target critic under the online policy, which is also the policy
            # inside the trace coefficients. Terminations are masked by the backup, not here.
            nxt_states = torch.as_tensor(trajectory.next_states, dtype=torch.float32)
            dones = torch.as_tensor(trajectory.dones, dtype=torch.float32).view(-1)
            nxt_v_vals = self.q_network.v_val(nxt_states, exp_actions=self.ret_add_acts, policy=self.policy).view(-1)
            rewards = torch.as_tensor(trajectory.rewards, dtype=torch.float32).view(-1)

            match self.kind:
                case "retrace":
                    coefficients = self._def_bounded_importance_sampling_coefficients(trajectory)
                    targets = retrace_targets(
                        q_vals, rewards, nxt_v_vals, dones, coefficients, self.discount, self.ret_bootstrap
                    )
                case "nstep":
                    targets = n_step_targets(rewards, nxt_v_vals, dones, self.discount, self.ret_bootstrap)
                case _:
                    raise ValueError(f"unknown return operator {self.kind}")

            trajectory["q"] = q_vals.view(-1, 1)
            trajectory["tq"] = targets.view(-1, 1)

    def _def_bounded_importance_sampling_coefficients(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Calculated the bounded importance sampling coefficients.
        :param trajectory: The trajectory for which the coefficients have to be calculated
        :return: A vector containing the coefficients.
        """

        behavior_log = torch.as_tensor(trajectory.log_likelihoods, dtype=torch.float32)
        states = torch.as_tensor(trajectory.states, dtype=torch.float32)
        actions = torch.as_tensor(trajectory.actions, dtype=torch.float32)
        with torch.no_grad():
            pi_log = self.policy.log_prob(actions, states=states)
        return bounded_importance_coefficients(pi_log, behavior_log, self.ret_lambda)
