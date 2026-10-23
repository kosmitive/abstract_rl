from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to


class TrustRegionGradient:
    """Add trust region gradient based optimization to any class by inheriting this one.

    The whole step, i.e. the gradient, the conjugate gradient with Fisher vector products (a jvp of the gradient of
    the trust region distance), the step size and the line search, is one compiled MLX function. MLX has no while
    loop, so the conjugate gradient always runs ``cg_max_k`` iterations (converged ones are frozen) and the line
    search evaluates all of its candidates and selects the first accepted one afterwards.
    """

    def __init__(
        self,
        model: nn.Module,
        tr_eps: float,
        cg_max_k: int = 10,
        cg_residual_tol: float = 1e-6,
        dvp_damping: float = 1e-5,
        mode: str = "min",
    ) -> None:
        """Should be self explanatory

        :param model: The model to use
        :param tr_eps: The trust region to use.
        :param cg_max_k: The maximum number of iterations for the conjugate gradient iteration.
        :param cg_residual_tol: The residual tolerance when executing the conjugate gradient.
        :param dvp_damping: The damping to use for the FIM
        :param mode: The mode in which to run the optimization.
        """
        self._model = model
        self.tr_eps = tr_eps
        self.cg_max_k = cg_max_k
        self.cg_residual_tol = cg_residual_tol
        self.dvp_damping = dvp_damping
        self.mode = mode
        self.cli = CliPrinter().instance
        self._compiled_step: Callable | None = None

        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def _flat_fn(self, fn: Callable[[Batch], mx.array], batch: Batch) -> Callable[[mx.array], mx.array]:
        """Turn an objective of the model into a function of its flat parameter vector."""

        def flat_fn(flat: mx.array) -> mx.array:
            set_flat_params_to(self._model, flat)
            return fn(batch)

        return flat_fn

    def tr_step(
        self, x: mx.array, tree: dict[str, Any], tr_tree: dict[str, Any], max_steps: int = 10
    ) -> dict[str, mx.array]:
        """
        One trust region step as a pure function of the flat parameters, traced by mx.compile.
        :param x: The flat parameters before the step.
        :param tree: The batch for the objective, see Batch.to_tree.
        :param tr_tree: The batch for the trust region distance.
        :param max_steps: The number of line search candidates.
        :return: The new parameters and the line search diagnostics.
        """
        batch, tr_batch = Batch.from_tree(tree), Batch.from_tree(tr_tree)

        # get v objective and the gradient
        old_obj, g = mx.value_and_grad(self._flat_fn(self.tr_obj, batch))(x)
        grad_dist = mx.grad(self._flat_fn(self.tr_distance, tr_batch))

        # fisher vector products as forward-over-reverse product of the gradient of the distance
        def fvp(v: mx.array) -> mx.array:
            _, (r_hvp,) = mx.jvp(grad_dist, [x], [v])
            return mx.stop_gradient(r_hvp) + self.dvp_damping * v

        # calc natural gradient and set gradient for line search
        s = conjugate_gradient(fvp, -g, self.cg_max_k, self.cg_residual_tol)
        sdvps = (s * fvp(s)).sum()
        nat_metric = 0.5 * sdvps
        lm = mx.sqrt(nat_metric / self.tr_eps)
        fullstep = s / lm
        exp_improv_rate = (-g * s).sum() / lm

        # line search, all candidates lr = 1, 1/2, 1/4, ... are evaluated
        obj_fn = self._flat_fn(self.tr_obj, batch)
        dist_fn = self._flat_fn(self.tr_distance, batch)
        lrs = 0.5 ** mx.arange(max_steps, dtype=mx.float32)
        new_objs = mx.stack([obj_fn(x + lrs[i] * fullstep) for i in range(max_steps)])
        distances = mx.stack([dist_fn(x + lrs[i] * fullstep) for i in range(max_steps)])
        act_improv = old_obj - new_objs
        exp_improv = lrs * exp_improv_rate
        ratios = act_improv / exp_improv
        accepted = (ratios > 0.1) & (act_improv > 0) & (distances <= 1.5 * self.tr_eps)

        # first accepted candidate, or max_steps if the line search failed
        first = mx.min(mx.where(accepted, mx.arange(max_steps), max_steps))
        lr_sel = mx.where(first < max_steps, 0.5 ** first.astype(mx.float32), 0.0)
        x_new = x + lr_sel * fullstep
        return {
            "x": x_new,
            "first": first,
            "old_obj": old_obj,
            "new_objs": new_objs,
            "distances": distances,
            "ratios": ratios,
            "act_improv": act_improv,
            "exp_improv": exp_improv,
        }

    def perform_tr_sgd(self, curst: int, batch: Batch, tr_batch: Batch) -> None:

        if self._compiled_step is None:
            # the parameters of the model are written by the step and restored below
            self._compiled_step = mx.compile(self.tr_step, inputs=self.tr_inputs())
        x = get_flat_params_from(self._model)
        out = self._compiled_step(x, batch.to_tree(["a"]), tr_batch.to_tree(["a"]))
        mx.eval(out)
        set_flat_params_to(self._model, out["x"])
        self.log_line_search(out)

    def log_line_search(self, out: dict[str, mx.array], max_steps: int = 10) -> tuple[bool, int, float, Any]:
        """Print the evaluated candidates of the line search up to the accepted one.

        :param out: The diagnostics of tr_step.
        :param max_steps: The number of line search candidates.
        :return: Whether a step was accepted, its index, the learning rate and the new objective.
        """

        fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %14.10f | %14.10f | %14.10f | %10.6f"
        titlestr = "%10s | %14s | %14s | %10s | %14s | %14s | %14s | %10s"

        self.cli.print(
            titlestr % ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv", "distance")
        ).line()
        vals = {k: np.array(v) for k, v in out.items()}
        first = int(vals["first"])
        old_obj = float(vals["old_obj"])
        for step in range(min(first + 1, max_steps)):
            self.cli.print(
                fmtstr
                % (
                    step,
                    old_obj,
                    vals["new_objs"][step],
                    0.5**step,
                    vals["ratios"][step],
                    vals["act_improv"][step],
                    vals["exp_improv"][step],
                    vals["distances"][step],
                )
            )
        if first < max_steps:
            return True, first, 0.5**first, float(vals["new_objs"][first])

        # if line search was not successful
        self.cli.line().print("FAILED")
        return False, max_steps, 0.5**max_steps, old_obj

    def tr_inputs(self) -> list[Any]:
        """Module states, which the objective and the distance read besides the model, e.g. a target policy."""
        return []

    def tr_distance(self, batch: Batch) -> mx.array:
        """The training distance."""
        raise NotImplementedError

    def tr_obj(self, batch: Batch) -> mx.array:
        """The training objective."""
        raise NotImplementedError
