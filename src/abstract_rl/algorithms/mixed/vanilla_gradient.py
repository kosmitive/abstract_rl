"""Adam steps shared by the value networks.

The public methods are the ones the networks inherited before the JAX port. Adam over a
device-backed trajectory collection is one ``lax.scan``; that scan is ``_scan_adam`` on the
network, because the loss is the subclass's ``tr_obj``.
"""

from typing import Any

import equinox as eqx
import optax
from jaxtyping import Array, Float

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter


class VanillaGradient:
    """Mini-batch Adam. Subclasses implement ``tr_obj``."""

    params: Any

    def __init__(self, lr: float = 1e-3) -> None:
        """Attach Adam to ``self.params``.

        Args:
            lr: Adam step size.
        """
        self.cli = CliPrinter().instance
        self.lr = float(lr)
        self.fmtstr = "%10i | %14.10f | %14.10f"
        self.titlestr = "%10s | %14s | %14s"
        self.tx = optax.adam(self.lr)
        diff, _static = eqx.partition(self.params, eqx.is_inexact_array)
        self.opt_state = self.tx.init(diff)

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Training objective of the subclass."""
        raise NotImplementedError

    def objective_on(self, params: Any, batch: Batch) -> Float[Array, ""]:
        """``tr_obj`` evaluated at ``params`` instead of ``self.params``."""
        del params, batch
        raise NotImplementedError

    def perform_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """Several Adam steps. A device-backed critic collection is one scan.

        Args:
            num_steps: Number of updates.
            tc: Trajectory collection to sample. MPO attaches an on-device copy.
            batch_size: Transitions per update.
            fields: Annotation fields the Python sampler must copy. ``tq`` selects the critic scan.
        """
        device = getattr(tc, "device", None)
        if device is not None and device.size > 0 and "tq" in fields:
            self._scan_adam(device, num_steps, min(int(batch_size), int(device.size)))
            return
        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        for step in range(num_steps):
            batch = tc.sample(batch_size, fields)
            self.perform_adam_step(step, batch)

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
