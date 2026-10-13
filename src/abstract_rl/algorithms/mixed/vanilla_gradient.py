"""Adam and L-BFGS steps shared by the value networks.

The public methods are the ones the networks inherited before the JAX port. Adam over a
device-backed trajectory collection is one ``lax.scan``; that scan is ``_scan_adam`` on the
network, because the loss (Q or V) is the subclass's ``tr_obj``.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize
from jaxtyping import Array, Float

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import ravel_module


class VanillaGradient:
    """Full-batch L-BFGS and mini-batch Adam. Subclasses implement ``tr_obj``."""

    params: Any

    def __init__(self, optimizer: str = "lbfgs", lr: float = 1e-3) -> None:
        """Attach an optimiser to ``self.params``.

        Args:
            optimizer: ``adam`` or ``lbfgs``.
            lr: Adam step size. Ignored by L-BFGS.
        """
        self.cli = CliPrinter().instance
        self.optimizer = optimizer
        self.lr = float(lr)
        self.fmtstr = "%10i | %14.10f | %14.10f"
        self.titlestr = "%10s | %14s | %14s"
        self.tx: Any = None
        self.opt_state: Any = None
        if optimizer == "adam":
            self.tx = optax.adam(self.lr)
            diff, _static = eqx.partition(self.params, eqx.is_inexact_array)
            self.opt_state = self.tx.init(diff)

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Training objective of the subclass."""
        raise NotImplementedError

    def objective_on(self, params: Any, batch: Batch) -> Float[Array, ""]:
        """``tr_obj`` evaluated at ``params`` instead of ``self.params``.

        L-BFGS calls this so the line search does not write into the live module on every probe.
        """
        del params, batch
        raise NotImplementedError

    def perform_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """Several optimiser steps. Adam on a device-backed collection is one scan.

        Args:
            num_steps: Number of updates.
            tc: Trajectory collection to sample. MPO attaches an on-device copy.
            batch_size: Transitions per update.
            fields: Annotation fields the Python sampler must copy. ``tq`` selects the critic loss.
        """
        device = getattr(tc, "device", None)
        if self.optimizer == "adam" and device is not None and device.size > 0 and "tq" in fields:
            self._scan_adam(device, num_steps, min(int(batch_size), int(device.size)))
            return
        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        for step in range(num_steps):
            batch = tc.sample(batch_size, fields)
            if self.optimizer == "adam":
                self.perform_adam_step(step, batch)
            else:
                self.perform_sgd(step, batch, 1)

    def _scan_adam(self, replay: Any, num_steps: int, batch_size: int) -> None:
        """Device Adam loop. Overridden by the Q network, which owns the critic loss."""
        raise NotImplementedError

    def perform_adam_step(self, curst: int, batch: Batch) -> None:
        """One Adam step on ``batch``, using ``tr_obj``."""
        diff, static = eqx.partition(self.params, eqx.is_inexact_array)

        def loss(weights: Any) -> Float[Array, ""]:
            return self.objective_on(eqx.combine(weights, static), batch)

        pre, grads = eqx.filter_value_and_grad(loss)(diff)
        updates, self.opt_state = self.tx.update(grads, self.opt_state, diff)
        self.params = eqx.combine(eqx.apply_updates(diff, updates), static)
        if curst % 10 == 0:
            post = self.tr_obj(batch)
            self.cli.print(self.fmtstr % (curst, float(pre), float(post)))

    def perform_sgd(self, curst: int, batch: Batch, maxiter: int = 25) -> None:
        """Fit ``tr_obj`` with L-BFGS-B.

        Args:
            curst: Step index, printed in the log line.
            batch: Batch the objective reads.
            maxiter: Iteration cap.
        """
        flat0, unravel = ravel_module(self.params)
        objective = eqx.filter_jit(lambda vector: self.objective_on(unravel(vector), batch))
        value_and_grad = jax.jit(jax.value_and_grad(objective))
        old = float(objective(flat0))

        def closure(x: np.ndarray) -> tuple[float, np.ndarray]:
            vector = jnp.asarray(x, dtype=jnp.float32)
            loss, grad = value_and_grad(vector)
            return float(loss), np.asarray(grad, dtype=np.float64)

        updated, _, _ = scipy.optimize.fmin_l_bfgs_b(closure, np.asarray(flat0, dtype=np.float64), maxiter=maxiter)
        self.params = unravel(jnp.asarray(updated, dtype=jnp.float32))
        new = float(objective(jnp.asarray(updated, dtype=jnp.float32)))
        self.cli.print(self.fmtstr % (curst, old, new))
