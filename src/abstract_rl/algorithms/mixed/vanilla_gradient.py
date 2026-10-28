from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from abstract_rl.algorithms.mixed.background_fit import BackgroundFit
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.metal_kernels import FusedAdam

# Adam steps per compiled call of perform_mult_sgd, also the interval of its log lines
STEPS_PER_CALL = 10


def compile_adam_fns(model: nn.Module, opt: optim.Optimizer, objective: Callable) -> dict[str, Callable]:
    """
    The compiled Adam functions of a model, the parameters and the moments of Adam are their implicit inputs and
    outputs. ``step`` does one step on a batch, ``obj`` evaluates the objective and ``steps`` does one step per row of
    indices, each on the rows of the replay at the indices.
    :param model: The model to fit.
    :param opt: Its optimizer.
    :param objective: The objective of a batch, given as a tree of arrays.
    """
    loss_and_grad = nn.value_and_grad(model, objective)

    def step(tree: dict[str, Any]) -> mx.array:
        obj, grads = loss_and_grad(tree)
        opt.update(model, grads)
        return obj

    state = [model.state, opt.state]

    def steps(tree: dict[str, Any], indices: mx.array) -> tuple[mx.array, mx.array]:
        # one adam step per row of indices, the objective before and after the first one for the log line. The rows
        # of all steps are gathered at once, each step slices its batch from them
        num, size = indices.shape
        rows = tree_map(lambda v: mx.take(v, indices.reshape(-1), axis=0), tree)
        for j in range(num):
            batch = tree_map(lambda v, j=j: v[j * size : (j + 1) * size], rows)
            step_obj = step(batch)
            if j == 0:
                pre_obj, post_obj = step_obj, objective(batch)
        return pre_obj, post_obj

    return {
        "step": mx.compile(step, inputs=state, outputs=state),
        "obj": mx.compile(objective, inputs=state),
        "steps": mx.compile(steps, inputs=state, outputs=state),
    }


def run_adam_steps(
    steps: Callable, model: nn.Module, opt: optim.Optimizer, tree: dict[str, Any], indices: mx.array
) -> list[tuple[int, mx.array, mx.array]]:
    """
    Queue one Adam step per row of indices, STEPS_PER_CALL of them per call of the compiled ``steps``.
    :return: The step and the objective before and after it, of every first step of a call.
    """
    logs = []
    for s in range(0, indices.shape[0], STEPS_PER_CALL):
        pre_obj, post_obj = steps(tree, indices[s : s + STEPS_PER_CALL])
        mx.async_eval(pre_obj, post_obj, model.parameters(), opt.state)
        logs.append((s, pre_obj, post_obj))
    return logs


class VanillaGradient:
    """Simple vanilla gradient. Simply inherit from this class to add support.

    Value networks are fit with Adam on minibatches. One Adam step (forward, backward and update) is one compiled MLX
    function, the parameters and the moments of Adam are its implicit inputs and outputs. perform_mult_sgd runs
    STEPS_PER_CALL steps, each gathering its batch from a flat replay on the device, in one compiled function.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        """
        Add vanilla gradient optimization.
        :param model: The mlx model to use.
        :param lr: The Adam learning rate.
        """
        self._model = model
        self.cli = CliPrinter().instance

        self.fmtstr = "%10i | %14.10f | %14.10f"
        self.titlestr = "%10s | %14s | %14s"
        # torch.optim.Adam corrects the bias of the moments, mlx only does that if asked to. FusedAdam is optim.Adam
        # in fewer GPU dispatches
        self.opt = FusedAdam(learning_rate=lr, bias_correction=True)
        self.opt.init(self._model.trainable_parameters())
        # a plain attribute, not an entry of the module tree of the inheriting network
        self._compiled: dict[str, Callable]
        object.__setattr__(self, "_compiled", {})
        # the worker of fit_in_background and the log values of the last fit, not yet printed
        self._worker: BackgroundFit | None = None
        self._logs: list[tuple[int, Any, Any]] = []

    def _adam_fns(self) -> tuple[Callable, Callable]:
        if not self._compiled:
            self._compiled.update(compile_adam_fns(self._model, self.opt, lambda t: self.tr_obj(Batch.from_tree(t))))
        return self._compiled["step"], self._compiled["obj"]

    def background_spec(self) -> tuple[str, dict[str, Any]]:
        """The builder (``module:function``) and its arguments of a fit worker, see BackgroundFit."""
        raise NotImplementedError

    def fit_in_background(self) -> None:
        """
        Run every later perform_mult_sgd in a worker process. start_mult_sgd returns once the fit is submitted and
        finish_mult_sgd waits for it and sets the new parameters. The parameters are only valid after
        finish_mult_sgd. The moments of Adam live in the worker, so opt is not used anymore.
        """
        if self._worker is None:
            self._worker = BackgroundFit(*self.background_spec())

    def close(self) -> None:
        """Stop the fit worker, if any. Pending results are collected first."""
        if self._worker is not None:
            self.finish_mult_sgd()
            self._worker.close()
            self._worker = None

    def perform_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """
        Performs multiple stochastic gradient descent steps.

        :param num_steps: The number of steps.
        :param tc: The trajectory collection to sample from.
        :param batch_size: The batch size to use for each batch.
        :param fields: The additional fields, which need to be sampled.
        """
        self.start_mult_sgd(num_steps, tc, batch_size, fields)
        self.finish_mult_sgd()

    def start_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """
        Draws the batches and queues the steps of perform_mult_sgd, see there. finish_mult_sgd prints the log lines.
        """

        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        if num_steps == 0:
            return

        # the indices of all batches, drawn in the order and with the calls of one tc.sample per step
        replay = tc.flatten(fields)
        indices = mx.array(replay.draw_batches(num_steps, batch_size), dtype=mx.int32)
        self._start_steps(replay.tree, indices)

    def start_mult_sgd_on(self, replay: dict[str, Any], indices: mx.array) -> None:
        """
        As start_mult_sgd, one step per row of indices into the rows of replay (a tree of Batch.to_tree with the
        annotations of the objective).
        """
        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        self._start_steps(replay, indices)

    def _start_steps(self, replay: dict[str, Any], indices: mx.array) -> None:
        if self._worker is not None:
            self.finish_mult_sgd()
            self._worker.submit(self._model.parameters(), replay, indices)
            return
        self._adam_fns()
        self._logs = run_adam_steps(self._compiled["steps"], self._model, self.opt, replay, indices)

    def finish_mult_sgd(self) -> None:
        """Waits for the steps of start_mult_sgd, sets the parameters of a background fit and prints the log lines."""
        if self._worker is not None and self._worker.pending:
            params, self._logs = self._worker.result()
            self._model.update(params)

        # the log lines of perform_adam_step, printed once all steps are queued
        logs, self._logs = self._logs, []
        for s, pre_obj, post_obj in logs:
            self.cli.print(self.fmtstr % (s, float(pre_obj), float(post_obj)))

    def perform_adam_step(self, curst: int, batch: Batch, fields: list[str] | None = None) -> None:
        """Performs a single adam step on the passed batch.

        :param curst: The current step.
        :param batch: The batch to use.
        :param fields: The annotations of the batch the objective needs, all of them if None.
        """
        step, obj_fn = self._adam_fns()
        tree = batch.to_tree(fields)
        obj = step(tree)
        if curst % 10 == 0:
            new_obj = obj_fn(tree)
            self.cli.print(self.fmtstr % (curst, obj.item(), new_obj.item()))
        else:
            # the next batch is drawn while this step runs
            mx.async_eval(obj, self._model.parameters(), self.opt.state)

    def tr_obj(self, batch: Batch) -> mx.array:
        """The training objective."""
        raise NotImplementedError


def background_adam_fit(model: nn.Module, lr: float, objective: Callable) -> Callable:
    """
    The fit of a worker process (see BackgroundFit) for a model fit like VanillaGradient.perform_mult_sgd. The moments
    of Adam live in the returned function.
    """
    opt = FusedAdam(learning_rate=lr, bias_correction=True)
    opt.init(model.trainable_parameters())
    fns = compile_adam_fns(model, opt, objective)

    def fit(params: dict[str, Any], tree: dict[str, Any], indices: mx.array) -> tuple[Any, list]:
        model.update(params)
        logs = run_adam_steps(fns["steps"], model, opt, tree, indices)
        mx.eval(logs, model.parameters(), opt.state)
        return model.parameters(), [(s, pre.item(), post.item()) for s, pre, post in logs]

    return fit
