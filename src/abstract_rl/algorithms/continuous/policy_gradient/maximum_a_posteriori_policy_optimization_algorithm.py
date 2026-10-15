"""MPO M-step: fit the parametric policy to the E-step under a mean KL trust region.

The default path (non-parametric, no coordinate ascent) runs the policy Adam step and the
Lagrange Adam step inside one ``lax.scan``. Parametric mode and coordinate ascent use the
same objective through ``perform_step``.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    descend_temperature,
    positive_temperature,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.replay_buffer import ReplayBuffer
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc import buckets
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import l2_norm, weight_spec
from abstract_rl.misc.obs_norm import normalize_obs
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.gaussian_math import (
    GaussianParams,
    forward,
    kl_divergence,
    kl_normal,
    kl_parts,
    log_prob,
)

_STATS_CHUNK = 1024


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """Maximum a posteriori policy optimisation.

    See https://arxiv.org/abs/1806.06920. Retrace lives outside this class. One Adam step updates
    the policy (max step) and one Adam step updates the positive Lagrange multiplier (min step).

    ``mpo_kl_mode: decoupled`` splits that constraint into a mean bound and a covariance bound
    (Abdolmaleki et al., appendix, and arXiv 1812.02256). Each bound has its own multiplier, and
    the constraint weight is the exponential of the multiplier, so the weight stays positive.
    ``eta_solver: sgd`` trains the E-step temperature on the same minibatches.
    """

    def __repr__(self) -> str:
        return "Maximum A Posteriori Policy Optimization"

    def __init__(self, mc: ModelConfiguration) -> None:
        """Create the policy and dual Adam states.

        Args:
            mc: Model configuration. The active namespace is the algorithm.
        """
        super().__init__(mc)
        policy = self.mc.get("policy", target=False)
        self.outer_lr = float(self.conf["outer_lr"])
        self.inner_lr = float(self.conf["inner_lr"])
        self.mode = str(self.conf["mode"])
        self.trust_regions = self.conf["trust_regions"]
        self.split_objective = bool(self.conf["coordinate_ascent_objective"])
        self.split_constraint = bool(self.conf["coordinate_ascent_constraint"])
        self.l2_reg = float(self.conf.get("l2_reg", 0.0))
        self.cli = CliPrinter().instance
        self._check_mode()
        root = self.conf.to_dict()
        self.kl_mode = str(root.get("mpo_kl_mode", "joint"))
        if self.kl_mode not in ("joint", "decoupled"):
            raise ValueError(f"unknown mpo_kl_mode {self.kl_mode}")
        self.decoupled = self.kl_mode == "decoupled"
        if self.decoupled:
            if self.split_constraint or self.split_objective:
                raise ValueError("decoupled KL uses two constraints and does not combine with coordinate ascent")
            self.trust_regions = [float(root.get("mpo_eps_mean", 0.1)), float(root.get("mpo_eps_cov", 1.0e-4))]

        match self.decoupled:
            case True:
                self.num_params = 2
            case False:
                match self.split_constraint:
                    case True:
                        self.num_params = policy.get_suff_stats_dim()
                    case False:
                        self.num_params = 1
        match self.num_params:
            case 1:
                self.lm = jnp.ones((), dtype=jnp.float32)
            case _:
                self.lm = jnp.ones((self.num_params,), dtype=jnp.float32)
        self.v = self.sample_v()
        self.tx_outer = optax.adam(self.outer_lr)
        self.tx_inner = optax.adam(self.inner_lr)
        diff, _static = eqx.partition(policy.params, weight_spec(policy.params))
        self.opt_outer = self.tx_outer.init(diff)
        self.opt_inner = self.tx_inner.init(self.lm)
        self.policy: ContinuousPolicy = policy
        self.fmtstr = "%10i | %10.5f" + self.num_params * " | %10.5f x %10.5f"
        self.titlestr = "%10s | %10s" + self.num_params * " | %23s"

    def _check_mode(self) -> None:
        match self.mode:
            case "non-parametric" | "parametric":
                return
            case _:
                raise NotImplementedError(f"unknown MPO mode {self.mode}")

    def sample_v(self) -> list[Any]:
        """Draw the trust-region bounds for this epoch.

        A scalar bound is used as-is. A pair ``[low, high]`` is sampled uniformly.

        Returns:
            One bound per constraint. The default configuration has a single bound.
        """
        sampled: list[Any] = []
        for bound in self.trust_regions:
            if isinstance(bound, list):
                sampled.append(float(np.random.uniform(bound[0], bound[1])))
            else:
                sampled.append(float(bound))
        self.v = sampled
        return self.v

    def shuffle_up_stats(self, batch: Batch) -> list[Float[Array, "batch stats"]]:
        """Mix current and target sufficient statistics, one vector per coordinate.

        The first entry is the live policy. Each later entry replaces every coordinate but one
        with the target policy, which is how coordinate ascent splits the objective.

        Args:
            batch: Minibatch whose states are evaluated.

        Returns:
            ``1 + dim`` statistic matrices.
        """
        states = jnp.asarray(batch.states, dtype=jnp.float32)
        current = self.policy.forward(states)
        target = jax.lax.stop_gradient(self.mc.get("policy", True).forward(states))
        return _mixed_stats(current, target)

    def main_obj(self, batch: Batch, l_suff_stats: list[Float[Array, "batch stats"]]) -> Float[Array, ""]:
        """Weighted objective, summed over coordinates when coordinate ascent is on.

        Args:
            batch: Minibatch of actions and E-step weights.
            l_suff_stats: Statistics from ``shuffle_up_stats``.

        Returns:
            Scalar objective, including the L2 penalty.
        """
        match self.split_objective:
            case True:
                total = jnp.zeros(())
                for index in range(l_suff_stats[0].shape[1]):
                    total = total + self.__lagrange_obj(batch, l_suff_stats[index + 1])
                return total
            case False:
                return self.__lagrange_obj(batch, l_suff_stats[0])

    def main_constraints(self, batch: Batch, l_suff_stats: list[Float[Array, "batch stats"]]) -> list[Float[Array, ""]]:
        """Trust-region slack ``epsilon - KL`` for each active constraint.

        Non-parametric mode uses the m-projection. Parametric mode uses the i-projection.

        Args:
            batch: Minibatch the divergence is averaged over.
            l_suff_stats: Statistics from ``shuffle_up_stats``.

        Returns:
            One slack per Lagrange multiplier.
        """
        if self.decoupled:
            states = jnp.asarray(batch.states, dtype=jnp.float32)
            current = self.policy.forward(states)
            target = jax.lax.stop_gradient(self.mc.get("policy", True).forward(states))
            kl_mu, kl_sigma = kl_parts(current, target, self.policy.cov_type, self.policy.mc.env.action_dim)
            return [
                jnp.asarray(self.v[0], dtype=jnp.float32) - kl_mu.mean(),
                jnp.asarray(self.v[1], dtype=jnp.float32) - kl_sigma.mean(),
            ]
        match self.mode:
            case "parametric":
                projection = "i-projection"
            case "non-parametric":
                projection = "m-projection"
            case _:
                raise NotImplementedError(self.mode)
        match self.split_constraint:
            case True:
                return [
                    jnp.asarray(self.v[index], dtype=jnp.float32)
                    - self.policy.kl_divergence(batch, l_suff_stats[index + 1], projection, "mean")
                    for index in range(l_suff_stats[0].shape[1])
                ]
            case False:
                return [
                    jnp.asarray(self.v[0], dtype=jnp.float32)
                    - self.policy.kl_divergence(batch, l_suff_stats[0], projection, "mean")
                ]

    def obj(self, batch: Batch) -> Float[Array, ""]:
        """Full Lagrangian, ``objective + exp(lambda) * constraint``.

        Args:
            batch: Minibatch used for the objective and the constraints.

        Returns:
            Scalar Lagrangian.
        """
        stats = self.shuffle_up_stats(batch)
        value = self.main_obj(batch, stats)
        multipliers = jnp.reshape(self.lm, (-1,))
        for multiplier, slack in zip(multipliers, self.main_constraints(batch, stats), strict=True):
            value = value + jnp.exp(multiplier) * slack
        return value

    def perform_mult_steps(self, num_steps: int, tc: TrajectoryCollection, batch_size: int) -> None:
        """M-step over ``num_steps`` minibatches from ``tc``.

        The non-parametric update without coordinate ascent is one ``lax.scan`` on the on-device
        copy. Every other setting runs ``perform_step`` on samples from the trajectories.

        Args:
            num_steps: Outer iterations. Each one takes one max step and one min step.
            tc: Trajectory collection. MPO attaches an on-device copy before training.
            batch_size: Transitions per step.
        """
        if num_steps <= 0:
            return
        self.sample_v()
        labels: tuple[str, ...]
        if self.decoupled:
            labels = ("alpha_mu x KL_mu", "alpha_sigma x KL_sigma")
        else:
            labels = tuple(f"lm_{index} x constr_{index}" for index in range(self.num_params))
        self.cli.empty().print(self.titlestr % ("k", "objective", *labels))
        device = tc.device
        match (self.mode, self.split_objective, self.split_constraint, device is not None and device.size > 0):
            case ("non-parametric", False, False, True):
                self._scan_default(device, num_steps, min(int(batch_size), int(device.size)))
            case _:
                for step in range(num_steps):
                    batch = tc.sample(batch_size, ["add_act", "grad_ranking", "add_pi"])
                    self.perform_step(step, batch)

    def perform_step(self, step: int, batch: Batch) -> None:
        """One max step on the policy and one min step on the multiplier.

        Args:
            step: Step index, printed every tenth update.
            batch: Minibatch with the E-step fields.
        """
        self.__max_step(batch)
        self.__min_step(batch)
        if step % 10 == 0:
            stats = self.shuffle_up_stats(batch)
            objective = float(self.main_obj(batch, stats))
            constraints = self.main_constraints(batch, stats)
            multipliers = np.asarray(self.lm).reshape(-1)
            values: list[float] = []
            for multiplier, slack in zip(multipliers, constraints, strict=True):
                values.extend((float(multiplier), float(slack)))
            self.cli.print(self.fmtstr % (step, objective, *values))

    def __lagrange_obj(self, batch: Batch, suff_stats: Float[Array, "batch stats"]) -> Float[Array, ""]:
        """Reweighted objective at an explicit sufficient statistic, before the constraints.

        Args:
            batch: Minibatch. Parametric mode reads ``a``; non-parametric mode reads the E-step.
            suff_stats: Sufficient statistics the log density is evaluated at.

        Returns:
            Scalar objective minus the L2 penalty.
        """
        return _weighted(self.policy.params, suff_stats, *_objective_inputs(self.mode, self.l2_reg, batch))

    def __max_step(self, batch: Batch) -> None:
        """Adam step that ascends the Lagrangian with respect to the policy."""
        states, actions, advantages, add_act, ranking = _batch_arrays(batch, self.mode)
        target = self.mc.get("policy", True).params
        epsilon = jnp.asarray(self.v, dtype=jnp.float32)
        multiplier = jax.lax.stop_gradient(self.lm)
        mode = self.mode
        split_objective = self.split_objective
        split_constraint = self.split_constraint
        l2_reg = self.l2_reg
        diff, static = eqx.partition(self.policy.params, weight_spec(self.policy.params))

        def loss(weights: Any) -> Float[Array, ""]:
            params = eqx.combine(weights, static)
            total, _main, _slack = _full_lagrangian(
                params,
                multiplier,
                target,
                states,
                mode,
                split_objective,
                split_constraint,
                epsilon,
                l2_reg,
                actions,
                advantages,
                add_act,
                ranking,
                self.decoupled,
            )
            return -total

        _value, grads = eqx.filter_value_and_grad(loss)(diff)
        updates, self.opt_outer = self.tx_outer.update(grads, self.opt_outer, diff)
        self.policy.params = eqx.combine(eqx.apply_updates(diff, updates), static)

    def __min_step(self, batch: Batch) -> None:
        """Adam step that descends the Lagrangian with respect to the multiplier."""
        states, actions, advantages, add_act, ranking = _batch_arrays(batch, self.mode)
        params = self.policy.params
        target = self.mc.get("policy", True).params
        epsilon = jnp.asarray(self.v, dtype=jnp.float32)
        mode = self.mode
        split_objective = self.split_objective
        split_constraint = self.split_constraint
        l2_reg = self.l2_reg

        def loss(multiplier: Float[Array, "..."]) -> Float[Array, ""]:
            total, _main, _slack = _full_lagrangian(
                params,
                multiplier,
                target,
                states,
                mode,
                split_objective,
                split_constraint,
                epsilon,
                l2_reg,
                actions,
                advantages,
                add_act,
                ranking,
                self.decoupled,
            )
            return total

        _value, grads = jax.value_and_grad(loss)(self.lm)
        updates, self.opt_inner = self.tx_inner.update(grads, self.opt_inner, self.lm)
        self.lm = eqx.apply_updates(self.lm, updates)

    def _scan_default(self, replay: ReplayBuffer, num_steps: int, batch_size: int) -> None:
        """Jitted non-parametric M-step. Logs the main objective, the raw multiplier and the slack.

        Replay states stay raw. The scan and the cached target statistics read a normalised copy
        of those rows, using the statistics frozen for this update.
        """
        states = normalize_obs(self.policy.observation_norm(), replay.states)
        _fill_target_stats(replay, self.mc.get("policy", target=True).params, states)
        diff, static = eqx.partition(self.policy.params, weight_spec(self.policy.params))
        if self.decoupled or str(self.conf.to_dict().get("eta_solver", "newton")) == "sgd":
            self._scan_options(replay, states, num_steps, batch_size, diff, static)
            return
        diff, self.opt_outer, self.lm, self.opt_inner, main, multipliers, constraints, static = _mpo_scan(
            (
                states,
                replay.add_act,
                replay.ranking,
                replay.target_stats,
                self.mc.rng.split(),
                jnp.int32(replay.size),
            ),
            diff,
            static,
            self.opt_outer,
            self.lm,
            self.opt_inner,
            self.tx_outer,
            self.tx_inner,
            num_steps,
            batch_size,
            jnp.float32(self.v[0]),
            self.l2_reg,
        )
        self.policy.params = eqx.combine(diff, static)
        main_np = np.asarray(main)
        lm_np = np.asarray(multipliers)
        constraint_np = np.asarray(constraints)
        for step in range(num_steps):
            if step % 10 == 0:
                self.cli.print(self.fmtstr % (step, main_np[step], lm_np[step], constraint_np[step]))

    def _scan_options(
        self,
        replay: ReplayBuffer,
        states: Float[Array, "cap obs"],
        num_steps: int,
        batch_size: int,
        diff: Any,
        static: Any,
    ) -> None:
        """M-step with a decoupled KL, a learned temperature, or both.

        The joint Newton path stays in ``_mpo_scan``. This scan is the one that carries two
        multipliers or an Adam state for η.
        """
        builder = self.mc.get("rnk").b
        eta_sgd = builder.eta_solver == "sgd"
        if eta_sgd:
            eta_raw = builder.eta_raw
            eta_opt = builder.eta_opt
            tx_eta = builder.tx_eta
            eta_eps = float(builder.non_parametric_variational_eps)
        else:
            eta_raw = jnp.zeros((), dtype=jnp.float32)
            tx_eta = optax.adam(1.0)
            eta_opt = tx_eta.init(eta_raw)
            eta_eps = 1.0
        epsilon = jnp.asarray(self.v, dtype=jnp.float32) if self.decoupled else jnp.float32(self.v[0])
        diff, self.opt_outer, self.lm, self.opt_inner, eta_raw, eta_opt, main, multipliers, constraints, static = (
            _mpo_scan_options(
                (
                    states,
                    replay.add_act,
                    replay.ranking,
                    replay.target_stats,
                    replay.add_q,
                    self.mc.rng.split(),
                    jnp.int32(replay.size),
                ),
                diff,
                static,
                self.opt_outer,
                self.lm,
                self.opt_inner,
                eta_raw,
                eta_opt,
                self.tx_outer,
                self.tx_inner,
                tx_eta,
                num_steps,
                batch_size,
                epsilon,
                self.l2_reg,
                eta_eps,
                self.decoupled,
                eta_sgd,
            )
        )
        self.policy.params = eqx.combine(diff, static)
        if eta_sgd:
            builder.eta_raw = eta_raw
            builder.eta_opt = eta_opt
            builder.eta = np.asarray([float(positive_temperature(eta_raw))], dtype=np.float32)
        main_np = np.asarray(main)
        lm_np = np.asarray(multipliers)
        constraint_np = np.asarray(constraints)
        for step in range(num_steps):
            if step % 10 != 0:
                continue
            if lm_np.ndim == 1:
                columns: tuple[float, ...] = (float(lm_np[step]), float(constraint_np[step]))
            else:
                columns = ()
                for index in range(lm_np.shape[1]):
                    # The stored multiplier is unconstrained. The constraint weight is its exponential.
                    columns = (
                        *columns,
                        float(np.exp(lm_np[step, index])),
                        float(constraint_np[step, index]),
                    )
            self.cli.print(self.fmtstr % (step, float(main_np[step]), *columns))


def _mixed_stats(
    current: Float[Array, "batch stats"], target: Float[Array, "batch stats"]
) -> list[Float[Array, "batch stats"]]:
    """Live statistics, then one mixed vector per coordinate."""
    frozen = jax.lax.stop_gradient(target)
    mixed = [current]
    width = current.shape[1]
    for index in range(width):
        parts: list[Float[Array, "batch ?"]] = []
        if index > 0:
            parts.append(frozen[:, :index])
        parts.append(current[:, index : index + 1])
        if index < width - 1:
            parts.append(frozen[:, index + 1 :])
        mixed.append(jnp.concatenate(parts, axis=1))
    return mixed


def _objective_inputs(
    mode: str, l2_reg: float, batch: Batch
) -> tuple[str, float, Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Arguments of ``_weighted`` after the sufficient statistics."""
    states_n = len(batch)
    match mode:
        case "parametric":
            actions = jnp.asarray(batch.actions, dtype=jnp.float32)
            advantages = jnp.asarray(batch["a"], dtype=jnp.float32)
            add_act = jnp.zeros((states_n, 1, actions.shape[-1]), dtype=jnp.float32)
            ranking = jnp.zeros((states_n, 1), dtype=jnp.float32)
        case "non-parametric":
            actions = jnp.asarray(batch.actions, dtype=jnp.float32)
            advantages = jnp.zeros((states_n,), dtype=jnp.float32)
            add_act = jnp.asarray(batch["add_act"], dtype=jnp.float32)
            ranking = jnp.asarray(batch["grad_ranking"], dtype=jnp.float32)
        case _:
            raise NotImplementedError(mode)
    return mode, l2_reg, actions, advantages, add_act, ranking


def _batch_arrays(
    batch: Batch, mode: str
) -> tuple[
    Float[Array, "batch obs"],
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
]:
    """States plus the action tensors ``_full_lagrangian`` reads."""
    _mode, _l2, actions, advantages, add_act, ranking = _objective_inputs(mode, 0.0, batch)
    states = jnp.asarray(batch.states, dtype=jnp.float32)
    return states, actions, advantages, add_act, ranking


def _weighted(
    params: GaussianParams,
    suff_stats: Float[Array, "batch stats"],
    mode: str,
    l2_reg: float,
    actions: Float[Array, "..."],
    advantages: Float[Array, "..."],
    add_act: Float[Array, "..."],
    ranking: Float[Array, "..."],
) -> Float[Array, ""]:
    """Parametric likelihood ratio, or the non-parametric weighted log density, minus L2."""
    match mode:
        case "parametric":
            density = log_prob(params, actions, suff_stats=suff_stats).reshape(-1)
            objective = jnp.mean(jnp.exp(density) * advantages.reshape(-1))
        case "non-parametric":
            density = log_prob(params, add_act, suff_stats=suff_stats)
            objective = jnp.mean(jnp.sum(jnp.exp(ranking) * density, axis=1))
        case _:
            raise NotImplementedError(mode)
    return objective - l2_reg * l2_norm(params)


def _full_lagrangian(
    params: GaussianParams,
    multiplier: Float[Array, "..."],
    target: GaussianParams,
    states: Float[Array, "batch obs"],
    mode: str,
    split_objective: bool,
    split_constraint: bool,
    epsilon: Float[Array, " constraints"],
    l2_reg: float,
    actions: Float[Array, "..."],
    advantages: Float[Array, "..."],
    add_act: Float[Array, "..."],
    ranking: Float[Array, "..."],
    decoupled: bool = False,
) -> tuple[Float[Array, ""], Float[Array, ""], list[Float[Array, ""]]]:
    """Lagrangian used by the Python M-step, including coordinate ascent.

    Returns:
        The full Lagrangian, the main objective printed in the log, and each constraint slack.
    """
    current = forward(params, states)
    target_stats = forward(target, states)
    if decoupled:
        main = _weighted(params, current, mode, l2_reg, actions, advantages, add_act, ranking)
        kl_mu, kl_sigma = kl_parts(
            current,
            jax.lax.stop_gradient(target_stats),
            params.cov_type,
            len(params.action_center),
        )
        dual_slacks = [epsilon[0] - kl_mu.mean(), epsilon[1] - kl_sigma.mean()]
        multipliers = jnp.reshape(multiplier, (-1,))
        total = main + jnp.exp(multipliers[0]) * dual_slacks[0] + jnp.exp(multipliers[1]) * dual_slacks[1]
        return total, main, dual_slacks
    mixed = _mixed_stats(current, target_stats)
    width = current.shape[1]
    match split_objective:
        case True:
            main = jnp.zeros(())
            for index in range(width):
                main = main + _weighted(params, mixed[index + 1], mode, l2_reg, actions, advantages, add_act, ranking)
        case False:
            main = _weighted(params, mixed[0], mode, l2_reg, actions, advantages, add_act, ranking)
    match mode:
        case "parametric":
            projection = "i-projection"
        case "non-parametric":
            projection = "m-projection"
        case _:
            raise NotImplementedError(mode)
    multipliers = jnp.reshape(multiplier, (-1,))
    match split_constraint:
        case True:
            total = main
            slacks: list[Float[Array, ""]] = []
            for index in range(width):
                slack = epsilon[index] - kl_divergence(params, target, states, mixed[index + 1], projection, "mean")
                slacks.append(slack)
                total = total + jnp.exp(multipliers[index]) * slack
        case False:
            slack = epsilon[0] - kl_divergence(params, target, states, mixed[0], projection, "mean")
            slacks = [slack]
            total = main + jnp.exp(multipliers[0]) * slack
    return total, main, slacks


def _fill_target_stats(replay: ReplayBuffer, target: GaussianParams, states: Float[Array, "cap obs"]) -> None:
    """Cache the target policy's sufficient statistics for the KL constraint.

    ``states`` is the normalised view of the raw replay. The scan covers a power-of-two prefix of
    the allocation (the whole allocation once the buffer is full). There is no randomness, so those
    rows match a full scan. ``n_chunks`` is static.
    """
    if replay.size == 0:
        return
    n_chunks = buckets.chunk_count(replay.size, replay.capacity, _STATS_CHUNK)
    replay.target_stats = _target_stats_all(target, states, jnp.int32(replay.size), n_chunks)


@eqx.filter_jit
def _target_stats_all(
    target: GaussianParams, states: Float[Array, "cap obs"], size: Array, n_chunks: int
) -> Float[Array, "cap stats"]:
    """Sufficient statistics of a frozen policy for the first ``n_chunks`` blocks. The tail is zero.

    There is no randomness. Block ``i`` reads the same rows a full-allocation scan would.
    """
    capacity = states.shape[0]

    def body(carry: Array, index: Array) -> tuple[Array, Array]:
        start = index * _STATS_CHUNK
        rows = jnp.minimum(jnp.arange(_STATS_CHUNK) + start, capacity - 1)
        stats = jax.lax.stop_gradient(forward(target, states[rows]))
        valid = (jnp.arange(_STATS_CHUNK) + start) < size
        return carry, jnp.where(valid[:, None], stats, 0)

    _carry, stats = jax.lax.scan(body, jnp.int32(0), jnp.arange(n_chunks))
    flat = stats.reshape((n_chunks * _STATS_CHUNK, stats.shape[-1]))
    prefix = min(n_chunks * _STATS_CHUNK, capacity)
    return jnp.zeros((capacity, flat.shape[-1]), dtype=flat.dtype).at[:prefix].set(flat[:prefix])


def _lagrangian(
    params: GaussianParams,
    multiplier: Float[Array, ""],
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch k act"],
    log_weights: Float[Array, "batch k"],
    target_stats: Float[Array, "batch stats"],
    epsilon: Float[Array, ""],
    l2_reg: float,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Non-parametric Lagrangian with cached target statistics.

    This is the default, non-split case of ``_full_lagrangian``. The target network is not
    evaluated again: ``target_stats`` was filled before the scan.

    Args:
        params: Current policy.
        multiplier: Raw Lagrange multiplier. The constraint weight is its exponential.
        states: Minibatch of states.
        actions: Extra actions from the E-step, shape ``[batch, k, act]``.
        log_weights: Per-state log categorical weights, shape ``[batch, k]``.
        target_stats: Sufficient statistics of the target policy.
        epsilon: Mean KL bound.
        l2_reg: Coefficient of the L2 penalty on the policy weights.

    Returns:
        The full Lagrangian, the main objective (what the log prints), and the constraint.
    """
    current = forward(params, states)
    main = _weighted(
        params,
        current,
        "non-parametric",
        l2_reg,
        actions,
        log_weights,
        actions,
        log_weights,
    )
    if params.cov_type == "full":
        # Same m-projection as the diagonal path: KL(target || current), summed over the two parts.
        kl_mu, kl_sigma = kl_parts(target_stats, current, "full", len(params.action_center))
        distance = (kl_mu + kl_sigma).mean()
    else:
        act = current.shape[-1] // 2
        mu, sigma = current[:, :act], current[:, act:]
        old_mu, old_sigma = target_stats[:, :act], target_stats[:, act:]
        # non-parametric M-step uses the forward KL, i.e. the m-projection onto the variational q
        distance = kl_normal(old_mu, old_sigma, mu, sigma).mean()
    constraint = epsilon - distance
    return main + jnp.exp(multiplier) * constraint, main, constraint


def _option_lagrangian(
    params: GaussianParams,
    multiplier: Float[Array, "..."],
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch k act"],
    log_weights: Float[Array, "batch k"],
    target_stats: Float[Array, "batch stats"],
    epsilon: Float[Array, "..."],
    l2_reg: float,
    decoupled: bool,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, "..."]]:
    """Joint Lagrangian, or the paper's mean and covariance constraints.

    Decoupled mode evaluates ``KL(current || target)`` with the covariance fixed at the target
    and again with the mean fixed at the target. Each slack keeps its own multiplier.
    """
    if not decoupled:
        return _lagrangian(params, multiplier, states, actions, log_weights, target_stats, epsilon, l2_reg)
    current = forward(params, states)
    main = _weighted(
        params,
        current,
        "non-parametric",
        l2_reg,
        actions,
        log_weights,
        actions,
        log_weights,
    )
    kl_mu, kl_sigma = kl_parts(current, target_stats, params.cov_type, len(params.action_center))
    slack = epsilon - jnp.stack([kl_mu.mean(), kl_sigma.mean()])
    total = main + jnp.sum(jnp.exp(multiplier) * slack)
    return total, main, slack


@eqx.filter_jit(donate="all-except-first")
def _mpo_scan(
    data: tuple[
        Float[Array, "cap obs"],
        Float[Array, "cap k act"],
        Float[Array, "cap k"],
        Float[Array, "cap stats"],
        PRNGKeyArray,
        Array,
    ],
    diff: Any,
    static: Any,
    opt_outer: Any,
    multiplier: Float[Array, ""],
    opt_inner: Any,
    tx_outer: Any,
    tx_inner: Any,
    num_steps: int,
    batch_size: int,
    epsilon: Float[Array, ""],
    l2_reg: float,
) -> tuple[
    Any,
    Any,
    Float[Array, ""],
    Any,
    Float[Array, " steps"],
    Float[Array, " steps"],
    Float[Array, " steps"],
    Any,
]:
    """Scan the max step (policy) and the min step (Lagrange multiplier).

    Replay tensors are the first argument and are not donated. The policy leaves, both Adam
    states, the multiplier and the static policy leaves are donated into the returned values.

    Returns:
        Updated policy leaves, policy optimiser state, multiplier, dual optimiser state,
        the per-step main objective, raw multiplier and constraint, and the static leaves.
    """
    states, actions, log_weights, target_stats, key, size = data

    def body(
        carry: tuple[Any, Any, Float[Array, ""], Any, PRNGKeyArray],
        _step: Array,
    ) -> tuple[tuple[Any, Any, Float[Array, ""], Any, PRNGKeyArray], tuple[Array, Array, Array]]:
        weights, policy_state, lm, dual_state, key_ = carry
        key_, sample_key = jax.random.split(key_)
        index = jax.random.randint(sample_key, (batch_size,), 0, size)
        batch_states = states[index]
        batch_actions = actions[index]
        batch_weights = log_weights[index]
        batch_target = jax.lax.stop_gradient(target_stats[index])

        def full(weights_: Any, lm_: Float[Array, ""]) -> Float[Array, ""]:
            params = eqx.combine(weights_, static)
            value, _main, _constraint = _lagrangian(
                params, lm_, batch_states, batch_actions, batch_weights, batch_target, epsilon, l2_reg
            )
            return value

        _loss, grads = eqx.filter_value_and_grad(lambda weights_: -full(weights_, jax.lax.stop_gradient(lm)))(weights)
        updates, policy_state = tx_outer.update(grads, policy_state, weights)
        weights = eqx.apply_updates(weights, updates)

        _dual, dual_grads = jax.value_and_grad(lambda lm_: full(weights, lm_))(lm)
        dual_updates, dual_state = tx_inner.update(dual_grads, dual_state, lm)
        lm = eqx.apply_updates(lm, dual_updates)

        _value, main, constraint = _lagrangian(
            eqx.combine(weights, static), lm, batch_states, batch_actions, batch_weights, batch_target, epsilon, l2_reg
        )
        return (weights, policy_state, lm, dual_state, key_), (main, lm, constraint)

    (diff, opt_outer, multiplier, opt_inner, _key), (main, multipliers, constraints) = jax.lax.scan(
        body, (diff, opt_outer, multiplier, opt_inner, key), jnp.arange(num_steps)
    )
    # ``static`` is donated with the other arguments, so hand the same buffer back. The std
    # floor lives there and the caller still needs it after the scan.
    return diff, opt_outer, multiplier, opt_inner, main, multipliers, constraints, static


@eqx.filter_jit(donate="all-except-first")
def _mpo_scan_options(
    data: tuple[
        Float[Array, "cap obs"],
        Float[Array, "cap k act"],
        Float[Array, "cap k"],
        Float[Array, "cap stats"],
        Float[Array, "cap k"],
        PRNGKeyArray,
        Array,
    ],
    diff: Any,
    static: Any,
    opt_outer: Any,
    multiplier: Float[Array, "..."],
    opt_inner: Any,
    eta_raw: Float[Array, ""],
    eta_opt: Any,
    tx_outer: Any,
    tx_inner: Any,
    tx_eta: Any,
    num_steps: int,
    batch_size: int,
    epsilon: Float[Array, "..."],
    l2_reg: float,
    eta_eps: float,
    decoupled: bool,
    eta_sgd: bool,
) -> tuple[
    Any,
    Any,
    Float[Array, "..."],
    Any,
    Float[Array, ""],
    Any,
    Float[Array, " steps"],
    Float[Array, "steps ..."],
    Float[Array, "steps ..."],
    Any,
]:
    """M-step scan with an optional split KL and an optional Adam step on η.

    ``decoupled`` and ``eta_sgd`` are static. When ``eta_sgd`` is set, each minibatch takes one
    Adam step on the temperature dual and the policy term uses ``log softmax(Q / η)`` at the
    updated temperature. The joint single-KL path is ``_mpo_scan``.
    """
    states, actions, log_weights, target_stats, q_values, key, size = data

    def body(
        carry: tuple[Any, Any, Float[Array, "..."], Any, Float[Array, ""], Any, PRNGKeyArray],
        _step: Array,
    ) -> tuple[
        tuple[Any, Any, Float[Array, "..."], Any, Float[Array, ""], Any, PRNGKeyArray],
        tuple[Array, Array, Array],
    ]:
        weights, policy_state, lm, dual_state, raw, eta_state, key_ = carry
        key_, sample_key = jax.random.split(key_)
        index = jax.random.randint(sample_key, (batch_size,), 0, size)
        batch_states = states[index]
        batch_actions = actions[index]
        batch_target = jax.lax.stop_gradient(target_stats[index])
        if eta_sgd:
            raw, eta_state, eta = descend_temperature(q_values[index], raw, eta_state, tx_eta, eta_eps)
            batch_weights = jax.nn.log_softmax(q_values[index] / eta, axis=-1)
        else:
            batch_weights = log_weights[index]

        def full(weights_: Any, lm_: Float[Array, "..."]) -> Float[Array, ""]:
            params = eqx.combine(weights_, static)
            value, _main, _constraint = _option_lagrangian(
                params, lm_, batch_states, batch_actions, batch_weights, batch_target, epsilon, l2_reg, decoupled
            )
            return value

        _loss, grads = eqx.filter_value_and_grad(lambda weights_: -full(weights_, jax.lax.stop_gradient(lm)))(weights)
        updates, policy_state = tx_outer.update(grads, policy_state, weights)
        weights = eqx.apply_updates(weights, updates)

        _dual, dual_grads = jax.value_and_grad(lambda lm_: full(weights, lm_))(lm)
        dual_updates, dual_state = tx_inner.update(dual_grads, dual_state, lm)
        lm = eqx.apply_updates(lm, dual_updates)

        _value, main, constraint = _option_lagrangian(
            eqx.combine(weights, static),
            lm,
            batch_states,
            batch_actions,
            batch_weights,
            batch_target,
            epsilon,
            l2_reg,
            decoupled,
        )
        return (weights, policy_state, lm, dual_state, raw, eta_state, key_), (main, lm, constraint)

    (diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, _key), (main, multipliers, constraints) = jax.lax.scan(
        body,
        (diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, key),
        jnp.arange(num_steps),
    )
    return diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, main, multipliers, constraints, static
