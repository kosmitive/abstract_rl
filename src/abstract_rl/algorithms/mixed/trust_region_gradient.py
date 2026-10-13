"""Natural gradient step with a KL line search (TRPO)."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import ravel_module
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.gaussian_math import kl_divergence, likelihood_ratio


class TrustRegionGradient:
    """Conjugate-gradient natural gradient and a backtracking line search.

    Subclasses implement ``tr_obj`` and ``tr_distance``. The Fisher product differentiates
    those same expressions through a flat parameter vector: a live module is not a JAX pytree,
    so the gradient cannot be taken with ``autograd.grad`` on ``self``.
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
        states = jnp.asarray(batch.states, dtype=jnp.float32)
        actions = jnp.asarray(batch.actions, dtype=jnp.float32)
        old_log_prob = jnp.asarray(batch.log_likelihoods, dtype=jnp.float32)
        advantages = jnp.asarray(batch["a"], dtype=jnp.float32)
        tr_states = jnp.asarray(tr_batch.states, dtype=jnp.float32)
        target = policy.mc.get("policy", True).params
        flat, unravel = ravel_module(policy.params)

        def surrogate(vector: Float[Array, " params"]) -> Float[Array, ""]:
            # Same expression as ``tr_obj``: the negative standardised likelihood ratio.
            return likelihood_ratio(unravel(vector), states, actions, old_log_prob, advantages)

        def mean_kl(vector: Float[Array, " params"]) -> Float[Array, ""]:
            # Same expression as ``tr_distance``: mean forward KL of the underlying Gaussians.
            return kl_divergence(unravel(vector), target, tr_states, mode_name="m-projection", est="mean")

        surrogate_jit = jax.jit(surrogate)
        gradient = jax.grad(surrogate_jit)(flat)
        kl_grad = jax.grad(mean_kl)

        def fisher_vector(direction: Float[Array, " params"]) -> Float[Array, " params"]:
            _unused, product = jax.jvp(kl_grad, (flat,), (direction,))
            return product + self.dvp_damping * direction

        fisher_vector = jax.jit(fisher_vector)
        step = conjugate_gradient(fisher_vector, -gradient, self.cg_max_k, self.cg_residual_tol)
        quadratic = 0.5 * jnp.dot(step, fisher_vector(step))
        lagrange = jnp.sqrt(quadratic / self.tr_eps)
        full_step = step / lagrange
        expected = jnp.dot(-gradient, step) / lagrange
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
        """Backtracking line search on ``tr_obj``.

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

        _current, unravel = ravel_module(self._model.params)
        old_obj = float(self.tr_obj(batch))
        rate = float(exp_improv_rate)
        step = 0
        lr = 1.0
        while step < max_steps:
            candidate = x + lr * fullstep
            self._model.params = unravel(candidate)
            new_obj = float(self.tr_obj(batch))
            actual = old_obj - new_obj
            expected = lr * rate
            ratio = actual / expected if expected != 0.0 else 0.0
            distance = float(self.tr_distance(batch))
            self.cli.print(fmtstr % (step, old_obj, new_obj, lr, ratio, actual, expected, distance))
            if ratio > accept_ratio and actual > 0.0 and distance <= max_kl_ratio * self.tr_eps:
                return True, step, lr, new_obj
            step += 1
            lr = lr / 2

        self.cli.line().print("FAILED")
        self._model.params = unravel(x)
        return False, step, lr, old_obj

    def tr_distance(self, batch: Batch) -> Float[Array, ""]:
        """Trust-region distance of the subclass."""
        del batch
        raise NotImplementedError

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Training objective of the subclass."""
        del batch
        raise NotImplementedError
