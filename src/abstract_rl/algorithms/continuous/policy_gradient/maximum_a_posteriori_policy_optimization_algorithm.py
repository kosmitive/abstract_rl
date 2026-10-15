"""MPO M-step: fit the parametric policy to the E-step under a decoupled KL trust region.

The policy Adam step, the Lagrange Adam step and one Adam step on the temperature run inside
one ``lax.scan``.
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
    kl_parts,
    log_prob,
)

_STATS_CHUNK = 1024


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """Maximum a posteriori policy optimisation.

    See https://arxiv.org/abs/1806.06920 and the decoupled mean and covariance constraints of
    arXiv 1812.02256. Retrace lives outside this class. One Adam step updates the policy (max
    step), one Adam step updates the two positive Lagrange multipliers (min step), and one Adam
    step updates the E-step temperature. The constraint weight is the exponential of the
    multiplier, so the weight stays positive.
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
        self.l2_reg = float(self.conf.get("l2_reg", 0.0))
        self.cli = CliPrinter().instance
        self.trust_regions = [float(self.conf["eps_mean"]), float(self.conf["eps_cov"])]
        self.num_params = 2
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

    def sample_v(self) -> list[Any]:
        """Draw the trust-region bounds for this epoch.

        A scalar bound is used as-is. A pair ``[low, high]`` is sampled uniformly.

        Returns:
            The mean bound and the covariance bound.
        """
        sampled: list[Any] = []
        for bound in self.trust_regions:
            if isinstance(bound, list):
                sampled.append(float(np.random.uniform(bound[0], bound[1])))
            else:
                sampled.append(float(bound))
        self.v = sampled
        return self.v

    def perform_mult_steps(self, num_steps: int, tc: TrajectoryCollection, batch_size: int) -> None:
        """M-step over ``num_steps`` minibatches from ``tc``.

        A device-backed collection is one ``lax.scan``. Without that copy each step samples
        the trajectories in Python.

        Args:
            num_steps: Outer iterations. Each one takes one max step and one min step.
            tc: Trajectory collection. MPO attaches an on-device copy before training.
            batch_size: Transitions per step.
        """
        if num_steps <= 0:
            return
        self.sample_v()
        labels = ("alpha_mu x KL_mu", "alpha_sigma x KL_sigma")
        self.cli.empty().print(self.titlestr % ("k", "objective", *labels))
        device = tc.device
        if device is not None and device.size > 0:
            self._scan_default(device, num_steps, min(int(batch_size), int(device.size)))
            return
        for step in range(num_steps):
            batch = tc.sample(batch_size, ["add_act", "grad_ranking", "add_pi"])
            self.perform_step(step, batch)

    def perform_step(self, step: int, batch: Batch) -> None:
        """One max step on the policy and one min step on the multipliers.

        Args:
            step: Step index, printed every tenth update.
            batch: Minibatch with the E-step fields.
        """
        prepared = _prepare(self, batch)
        self.__max_step(prepared)
        self.__min_step(prepared)
        if step % 10 == 0:
            _value, main, slack = _decoupled_lagrangian(
                self.policy.params,
                self.lm,
                prepared[0],
                prepared[1],
                prepared[2],
                prepared[3],
                prepared[4],
                self.l2_reg,
            )
            multipliers = np.exp(np.asarray(self.lm).reshape(-1))
            slacks = np.asarray(slack).reshape(-1)
            columns: list[float] = []
            for multiplier, gap in zip(multipliers, slacks, strict=True):
                columns.extend((float(multiplier), float(gap)))
            self.cli.print(self.fmtstr % (step, float(main), *columns))

    def __max_step(self, prepared: tuple[Any, ...]) -> None:
        """Adam step that ascends the Lagrangian with respect to the policy."""
        states, add_act, ranking, target, epsilon = prepared
        multiplier = jax.lax.stop_gradient(self.lm)
        l2_reg = self.l2_reg
        diff, static = eqx.partition(self.policy.params, weight_spec(self.policy.params))

        def loss(weights: Any) -> Float[Array, ""]:
            params = eqx.combine(weights, static)
            total, _main, _slack = _decoupled_lagrangian(
                params, multiplier, states, add_act, ranking, target, epsilon, l2_reg
            )
            return -total

        _value, grads = eqx.filter_value_and_grad(loss)(diff)
        updates, self.opt_outer = self.tx_outer.update(grads, self.opt_outer, diff)
        self.policy.params = eqx.combine(eqx.apply_updates(diff, updates), static)

    def __min_step(self, prepared: tuple[Any, ...]) -> None:
        """Adam step that descends the Lagrangian with respect to the multipliers."""
        states, add_act, ranking, target, epsilon = prepared
        params = self.policy.params
        l2_reg = self.l2_reg

        def loss(multiplier: Float[Array, " 2"]) -> Float[Array, ""]:
            total, _main, _slack = _decoupled_lagrangian(
                params, multiplier, states, add_act, ranking, target, epsilon, l2_reg
            )
            return total

        _value, grads = jax.value_and_grad(loss)(self.lm)
        updates, self.opt_inner = self.tx_inner.update(grads, self.opt_inner, self.lm)
        self.lm = eqx.apply_updates(self.lm, updates)

    def _scan_default(self, replay: ReplayBuffer, num_steps: int, batch_size: int) -> None:
        """Jitted M-step. Logs the main objective, the positive multipliers and the slacks.

        Replay states stay raw. The scan and the cached target statistics read a normalised copy
        of those rows, using the statistics frozen for this update.
        """
        states = normalize_obs(self.policy.observation_norm(), replay.states)
        _fill_target_stats(replay, self.mc.get("policy", target=True).params, states)
        diff, static = eqx.partition(self.policy.params, weight_spec(self.policy.params))
        builder = self.mc.get("rnk").b
        diff, self.opt_outer, self.lm, self.opt_inner, eta_raw, eta_opt, main, multipliers, constraints, static = (
            _mpo_scan(
                (
                    states,
                    replay.add_act,
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
                builder.eta_raw,
                builder.eta_opt,
                self.tx_outer,
                self.tx_inner,
                builder.tx_eta,
                num_steps,
                batch_size,
                jnp.asarray(self.v, dtype=jnp.float32),
                self.l2_reg,
                float(builder.non_parametric_variational_eps),
            )
        )
        self.policy.params = eqx.combine(diff, static)
        builder.eta_raw = eta_raw
        builder.eta_opt = eta_opt
        builder.eta = np.asarray([float(positive_temperature(eta_raw))], dtype=np.float32)
        main_np = np.asarray(main)
        lm_np = np.asarray(multipliers)
        constraint_np = np.asarray(constraints)
        for step in range(num_steps):
            if step % 10 != 0:
                continue
            columns: tuple[float, ...] = ()
            for index in range(lm_np.shape[1]):
                # The stored multiplier is unconstrained. The constraint weight is its exponential.
                columns = (*columns, float(np.exp(lm_np[step, index])), float(constraint_np[step, index]))
            self.cli.print(self.fmtstr % (step, float(main_np[step]), *columns))


def _prepare(
    algorithm: MPOAlgorithm, batch: Batch
) -> tuple[
    Float[Array, "batch obs"],
    Float[Array, "batch k act"],
    Float[Array, "batch k"],
    Float[Array, "batch stats"],
    Float[Array, " 2"],
]:
    """Normalised states, E-step actions and weights, frozen target statistics, and the bounds."""
    states = normalize_obs(algorithm.policy.observation_norm(), jnp.asarray(batch.states, dtype=jnp.float32))
    add_act = jnp.asarray(batch["add_act"], dtype=jnp.float32)
    ranking = jnp.asarray(batch["grad_ranking"], dtype=jnp.float32)
    target = jax.lax.stop_gradient(forward(algorithm.mc.get("policy", True).params, states))
    epsilon = jnp.asarray(algorithm.v, dtype=jnp.float32)
    return states, add_act, ranking, target, epsilon


def _weighted(
    params: GaussianParams,
    suff_stats: Float[Array, "batch stats"],
    l2_reg: float,
    add_act: Float[Array, "batch k act"],
    ranking: Float[Array, "batch k"],
) -> Float[Array, ""]:
    """Non-parametric weighted log density, minus the L2 penalty."""
    density = log_prob(params, add_act, suff_stats=suff_stats)
    objective = jnp.mean(jnp.sum(jnp.exp(ranking) * density, axis=1))
    return objective - l2_reg * l2_norm(params)


def _decoupled_lagrangian(
    params: GaussianParams,
    multiplier: Float[Array, " 2"],
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch k act"],
    log_weights: Float[Array, "batch k"],
    target_stats: Float[Array, "batch stats"],
    epsilon: Float[Array, " 2"],
    l2_reg: float,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " 2"]]:
    """Lagrangian with a mean constraint and a covariance constraint.

    ``KL(current || target)`` is evaluated once with the covariance fixed at the target and once
    with the mean fixed at the target. Each slack keeps its own multiplier.

    Returns:
        The full Lagrangian, the main objective printed in the log, and the two slacks.
    """
    current = forward(params, states)
    main = _weighted(params, current, l2_reg, actions, log_weights)
    kl_mu, kl_sigma = kl_parts(current, target_stats, params.cov_type, len(params.action_center))
    slack = epsilon - jnp.stack([kl_mu.mean(), kl_sigma.mean()])
    total = main + jnp.sum(jnp.exp(multiplier) * slack)
    return total, main, slack


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


@eqx.filter_jit(donate="all-except-first")
def _mpo_scan(
    data: tuple[
        Float[Array, "cap obs"],
        Float[Array, "cap k act"],
        Float[Array, "cap stats"],
        Float[Array, "cap k"],
        PRNGKeyArray,
        Array,
    ],
    diff: Any,
    static: Any,
    opt_outer: Any,
    multiplier: Float[Array, " 2"],
    opt_inner: Any,
    eta_raw: Float[Array, ""],
    eta_opt: Any,
    tx_outer: Any,
    tx_inner: Any,
    tx_eta: Any,
    num_steps: int,
    batch_size: int,
    epsilon: Float[Array, " 2"],
    l2_reg: float,
    eta_eps: float,
) -> tuple[
    Any,
    Any,
    Float[Array, " 2"],
    Any,
    Float[Array, ""],
    Any,
    Float[Array, " steps"],
    Float[Array, "steps 2"],
    Float[Array, "steps 2"],
    Any,
]:
    """Scan the policy step, the multiplier step and one Adam step on η.

    Each minibatch takes one Adam step on the temperature dual and the policy term uses
    ``log softmax(Q / η)`` at the updated temperature. Replay tensors are the first argument
    and are not donated. The policy leaves, both Adam states, the multipliers, the temperature
    and the static policy leaves are donated into the returned values.

    Returns:
        Updated policy leaves, policy optimiser state, multipliers, dual optimiser state,
        temperature parameter and its Adam state, the per-step main objective, positive-scale
        inputs (the raw multipliers) and slacks, and the static leaves.
    """
    states, actions, target_stats, q_values, key, size = data

    def body(
        carry: tuple[Any, Any, Float[Array, " 2"], Any, Float[Array, ""], Any, PRNGKeyArray],
        _step: Array,
    ) -> tuple[
        tuple[Any, Any, Float[Array, " 2"], Any, Float[Array, ""], Any, PRNGKeyArray],
        tuple[Array, Array, Array],
    ]:
        weights, policy_state, lm, dual_state, raw, eta_state, key_ = carry
        key_, sample_key = jax.random.split(key_)
        index = jax.random.randint(sample_key, (batch_size,), 0, size)
        batch_states = states[index]
        batch_actions = actions[index]
        batch_target = jax.lax.stop_gradient(target_stats[index])
        raw, eta_state, eta = descend_temperature(q_values[index], raw, eta_state, tx_eta, eta_eps)
        batch_weights = jax.nn.log_softmax(q_values[index] / eta, axis=-1)

        def full(weights_: Any, lm_: Float[Array, " 2"]) -> Float[Array, ""]:
            params = eqx.combine(weights_, static)
            value, _main, _constraint = _decoupled_lagrangian(
                params, lm_, batch_states, batch_actions, batch_weights, batch_target, epsilon, l2_reg
            )
            return value

        _loss, grads = eqx.filter_value_and_grad(lambda weights_: -full(weights_, jax.lax.stop_gradient(lm)))(weights)
        updates, policy_state = tx_outer.update(grads, policy_state, weights)
        weights = eqx.apply_updates(weights, updates)

        _dual, dual_grads = jax.value_and_grad(lambda lm_: full(weights, lm_))(lm)
        dual_updates, dual_state = tx_inner.update(dual_grads, dual_state, lm)
        lm = eqx.apply_updates(lm, dual_updates)

        _value, main, constraint = _decoupled_lagrangian(
            eqx.combine(weights, static),
            lm,
            batch_states,
            batch_actions,
            batch_weights,
            batch_target,
            epsilon,
            l2_reg,
        )
        return (weights, policy_state, lm, dual_state, raw, eta_state, key_), (main, lm, constraint)

    (diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, _key), (main, multipliers, constraints) = jax.lax.scan(
        body,
        (diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, key),
        jnp.arange(num_steps),
    )
    # ``static`` is donated with the other arguments, so hand the same buffer back. The std
    # floor lives there and the caller still needs it after the scan.
    return diff, opt_outer, multiplier, opt_inner, eta_raw, eta_opt, main, multipliers, constraints, static
