"""Adam and L-BFGS steps shared by the value networks.

The public methods are the ones the networks inherited before the JAX port. Adam over a
device-backed trajectory collection is one ``lax.scan``; that scan is ``_scan_adam`` on the
network, because the loss (Q or V) is the subclass's ``tr_obj``. L-BFGS is Optax with the
zoom line search, also one ``lax.scan``, so the value fit never leaves the jit.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import optax
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
        """Fit ``tr_obj`` with L-BFGS.

        Args:
            curst: Step index, printed in the log line.
            batch: Batch the objective reads.
            maxiter: Iteration cap. Static for the scan, so a new cap compiles once.
        """
        flat0, unravel = ravel_module(self.params)
        steps = int(maxiter)

        def loss(vector: Float[Array, " params"]) -> Float[Array, ""]:
            return self.objective_on(unravel(vector), batch)

        updated, old, new = eqx.filter_jit(lambda vector: lbfgs_minimize(loss, vector, steps))(flat0)
        self.params = unravel(updated)
        self.cli.print(self.fmtstr % (curst, float(old), float(new)))


def lbfgs_minimize(
    loss: Callable[[Float[Array, " params"]], Float[Array, ""]],
    vector: Float[Array, " params"],
    steps: int,
) -> tuple[Float[Array, " params"], Float[Array, ""], Float[Array, ""]]:
    """``steps`` Optax L-BFGS updates with the default zoom line search.

    The first returned scalar is the loss before the first step. The second is the loss after
    the last step. ``steps`` has to be a concrete integer because it is the length of the scan.
    """
    solver = optax.lbfgs()
    opt_state = solver.init(vector)

    def body(
        carry: tuple[Float[Array, " params"], Any], _: None
    ) -> tuple[tuple[Float[Array, " params"], Any], Float[Array, ""]]:
        params, state = carry
        value, grad = optax.value_and_grad_from_state(loss)(params, state=state)
        updates, state = solver.update(grad, state, params, value=value, grad=grad, value_fn=loss)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    (params, _state), values = jax.lax.scan(body, (vector, opt_state), None, length=steps)
    return params, values[0], loss(params)
