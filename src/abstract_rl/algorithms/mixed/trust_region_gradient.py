"""Natural gradient step with a KL line search (TRPO)."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.algorithms.mixed.conjugate_gradient import cg_solve
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import ravel_module
from abstract_rl.misc.obs_norm import normalize_obs
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.gaussian_math import GaussianParams

Objective = Callable[
    [
        GaussianParams,
        Float[Array, "batch obs"],
        Float[Array, "batch act"],
        Float[Array, " batch"],
        Float[Array, " batch"],
    ],
    Float[Array, ""],
]
Distance = Callable[[GaussianParams, GaussianParams, Float[Array, "batch obs"]], Float[Array, ""]]


class TrustRegionGradient:
    """Conjugate-gradient natural gradient and a backtracking line search.

    Subclasses implement ``tr_objective`` and ``tr_divergence``. The Fisher product and the line
    search call those functions, so the surrogate and the KL are written once. A live module is
    not a JAX pytree, which is why the functions take a parameter tree.
    """

    def __init__(
        self,
        model: ContinuousPolicy,
        tr_eps: float,
        cg_max_k: int = 10,
        cg_residual_tol: float = 1e-6,
        dvp_damping: float = 1e-5,
        mode: str = "min",
    ) -> None:
        """Attach the trust-region step to ``model``.

        Args:
            model: Policy whose parameters are updated.
            tr_eps: Mean KL bound.
            cg_max_k: Conjugate-gradient iteration cap.
            cg_residual_tol: Residual at which conjugate gradient stops.
            dvp_damping: Diagonal added to the Fisher-vector product.
            mode: Kept for the historical constructor. The surrogate is already a minimisation objective.
        """
        self._model = model
        self.tr_eps = tr_eps
        self.cg_max_k = cg_max_k
        self.cg_residual_tol = cg_residual_tol
        self.dvp_damping = dvp_damping
        self.mode = mode
        self.cli = CliPrinter().instance
        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def perform_tr_sgd(self, curst: int, batch: Batch, tr_batch: Batch) -> None:
        """One natural-gradient step on ``self._model``.

        Args:
            curst: Step index kept for the historical call site.
            batch: On-policy batch. The surrogate reads advantages from ``a``.
            tr_batch: Batch the Fisher product is measured on.
        """
        del curst
        policy = self._model
        norm = policy.observation_norm()
        states = normalize_obs(norm, jnp.asarray(batch.states, dtype=jnp.float32))
        actions = jnp.asarray(batch.actions, dtype=jnp.float32)
        old_log_prob = jnp.asarray(batch.log_likelihoods, dtype=jnp.float32)
        advantages = jnp.asarray(batch["a"], dtype=jnp.float32)
        tr_states = normalize_obs(norm, jnp.asarray(tr_batch.states, dtype=jnp.float32))
        target = policy.mc.get("policy", True).params
        # One compiled program for the gradient, the Fisher products and the conjugate gradient.
        # Building those closures inside this method compiled them again on every epoch.
        flat, full_step, expected = _trpo_natural_step(
            self.tr_objective,
            self.tr_divergence,
            policy.params,
            target,
            states,
            actions,
            old_log_prob,
            advantages,
            tr_states,
            jnp.float32(self.dvp_damping),
            jnp.int32(self.cg_max_k),
            jnp.float32(self.cg_residual_tol),
            jnp.float32(self.tr_eps),
        )
        self.line_search(batch, flat, full_step, expected)

    def line_search(
        self,
        batch: Batch,
        x: Float[Array, " params"],
        fullstep: Float[Array, " params"],
        exp_improv_rate: Float[Array, ""],
        max_steps: int = 10,
        accept_ratio: float = 0.1,
        max_kl_ratio: float = 1.5,
    ) -> tuple[bool, int, float, float]:
        """Backtracking line search on ``tr_objective``.

        Candidate parameters are written into the module before each probe, matching the
        historical search. The accepted vector is left in place.

        Args:
            batch: Batch passed to ``tr_obj`` and ``tr_distance``.
            x: Flat parameters at the start of the search.
            fullstep: Full natural-gradient step.
            exp_improv_rate: Expected improvement of a unit step.
            max_steps: How many times the step is halved.
            accept_ratio: Minimum ratio of actual to expected improvement.
            max_kl_ratio: Reject a step whose KL exceeds this multiple of ``tr_eps``.

        Returns:
            Whether a step was accepted, the probe index, the learning rate, and the objective.
        """
        fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %14.10f | %14.10f | %14.10f | %10.6f"
        titlestr = "%10s | %14s | %14s | %10s | %14s | %14s | %14s | %10s"
        self.cli.print(
            titlestr % ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv", "distance")
        ).line()

        # The probes are ``tr_objective`` and ``tr_divergence``. They run in one ``while_loop`` so a
        # rejected step does not sync the host, then the log is printed once.
        states = normalize_obs(self._model.observation_norm(), jnp.asarray(batch.states, dtype=jnp.float32))
        actions = jnp.asarray(batch.actions, dtype=jnp.float32)
        old_log_prob = jnp.asarray(batch.log_likelihoods, dtype=jnp.float32)
        advantages = jnp.asarray(batch["a"], dtype=jnp.float32)
        target = self._model.mc.get("policy", True).params
        accepted, step, lr, new_obj, old_obj, chosen, log = _backtrack(
            self.tr_objective,
            self.tr_divergence,
            self._model.params,
            target,
            x,
            fullstep,
            exp_improv_rate,
            states,
            actions,
            old_log_prob,
            advantages,
            jnp.float32(accept_ratio),
            jnp.float32(max_kl_ratio),
            jnp.float32(self.tr_eps),
            max_steps,
        )
        _current, unravel = ravel_module(self._model.params)
        self._model.params = unravel(chosen)
        accepted_py = bool(accepted)
        step_py = int(step)
        old_py = float(old_obj)
        n_lines = step_py + 1 if accepted_py else step_py
        rows = np.asarray(log)
        for index in range(n_lines):
            self.cli.print(
                fmtstr
                % (
                    index,
                    old_py,
                    rows[index, 0],
                    rows[index, 1],
                    rows[index, 2],
                    rows[index, 3],
                    rows[index, 4],
                    rows[index, 5],
                )
            )
        if not accepted_py:
            self.cli.line().print("FAILED")
        return accepted_py, step_py, float(lr), float(new_obj)

    def tr_distance(self, batch: Batch) -> Float[Array, ""]:
        """Trust-region distance of the subclass."""
        del batch
        raise NotImplementedError

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Training objective of the subclass."""
        del batch
        raise NotImplementedError

    @staticmethod
    def tr_objective(
        params: GaussianParams,
        states: Float[Array, "batch obs"],
        actions: Float[Array, "batch act"],
        old_log_prob: Float[Array, " batch"],
        advantages: Float[Array, " batch"],
    ) -> Float[Array, ""]:
        """Surrogate at ``params``. The natural step and ``tr_obj`` both call this."""
        del params, states, actions, old_log_prob, advantages
        raise NotImplementedError

    @staticmethod
    def tr_divergence(
        params: GaussianParams,
        target: GaussianParams,
        states: Float[Array, "batch obs"],
    ) -> Float[Array, ""]:
        """Trust-region distance at ``params``. The natural step and ``tr_distance`` both call this."""
        del params, target, states
        raise NotImplementedError


@eqx.filter_jit
def _trpo_natural_step(
    objective: Objective,
    divergence: Distance,
    params: GaussianParams,
    target: GaussianParams,
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch act"],
    old_log_prob: Float[Array, " batch"],
    advantages: Float[Array, " batch"],
    tr_states: Float[Array, "batch obs"],
    damping: Float[Array, ""],
    max_k: Array,
    tol: Float[Array, ""],
    tr_eps: Float[Array, ""],
) -> tuple[Float[Array, " params"], Float[Array, " params"], Float[Array, ""]]:
    """Flat parameters, the full natural step, and its expected improvement.

    ``objective`` and ``divergence`` are the subclass hooks. They are static.
    """
    flat, unravel = ravel_module(params)

    def mean_kl(vector: Float[Array, " params"]) -> Float[Array, ""]:
        return divergence(unravel(vector), target, tr_states)

    def hvp(direction: Float[Array, " params"]) -> Float[Array, " params"]:
        _value, product = jax.jvp(jax.grad(mean_kl), (flat,), (direction,))
        return product + damping * direction

    gradient = jax.grad(lambda vector: objective(unravel(vector), states, actions, old_log_prob, advantages))(flat)
    step = cg_solve(hvp, -gradient, max_k, tol)
    quadratic = 0.5 * jnp.dot(step, hvp(step))
    lagrange = jnp.sqrt(quadratic / tr_eps)
    return flat, step / lagrange, jnp.dot(-gradient, step) / lagrange


@eqx.filter_jit
def _backtrack(
    objective: Objective,
    divergence: Distance,
    params: GaussianParams,
    target: GaussianParams,
    x: Float[Array, " params"],
    fullstep: Float[Array, " params"],
    rate: Float[Array, ""],
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch act"],
    old_log_prob: Float[Array, " batch"],
    advantages: Float[Array, " batch"],
    accept_ratio: Float[Array, ""],
    max_kl_ratio: Float[Array, ""],
    tr_eps: Float[Array, ""],
    max_steps: int,
) -> tuple[
    Array,
    Array,
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " params"],
    Float[Array, "steps cols"],
]:
    """Backtracking search. Stops at the first step that satisfies the TRPO acceptance rule.

    Each probe calls ``objective`` and ``divergence``. The log columns are objective, learning
    rate, ratio, actual improvement, expected improvement and distance.
    """
    _flat, unravel = ravel_module(params)
    old_obj = objective(unravel(x), states, actions, old_log_prob, advantages)
    blank = jnp.zeros((max_steps, 6), dtype=jnp.float32)

    def cond(state: tuple[Any, ...]) -> Array:
        step, accepted, _lr, _current, _obj, _log = state
        return (step < max_steps) & (~accepted)

    def body(state: tuple[Any, ...]) -> tuple[Any, ...]:
        step, _accepted, lr, current, new_obj, log = state
        trial = x + lr * fullstep
        value = objective(unravel(trial), states, actions, old_log_prob, advantages)
        distance = divergence(unravel(trial), target, states)
        actual = old_obj - value
        expected = lr * rate
        ratio = jnp.where(expected != 0.0, actual / expected, 0.0)
        ok = (ratio > accept_ratio) & (actual > 0.0) & (distance <= max_kl_ratio * tr_eps)
        row = jnp.stack([value, lr, ratio, actual, expected, distance]).astype(jnp.float32)
        return (
            jnp.where(ok, step, step + jnp.int32(1)),
            ok,
            jnp.where(ok, lr, lr / 2.0),
            jnp.where(ok, trial, current),
            jnp.where(ok, value, new_obj),
            log.at[step].set(row),
        )

    step, accepted, lr, chosen, new_obj, log = jax.lax.while_loop(
        cond,
        body,
        (jnp.int32(0), jnp.bool_(False), jnp.float32(1.0), x, old_obj, blank),
    )
    return accepted, step, lr, new_obj, old_obj, chosen, log
