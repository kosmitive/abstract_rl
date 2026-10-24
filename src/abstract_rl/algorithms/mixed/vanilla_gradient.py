from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter

# Adam steps per compiled call of perform_mult_sgd, also the interval of its log lines
STEPS_PER_CALL = 10


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
        # torch.optim.Adam corrects the bias of the moments, mlx only does that if asked to
        self.opt = optim.Adam(learning_rate=lr, bias_correction=True)
        self.opt.init(self._model.trainable_parameters())
        # a plain attribute, not an entry of the module tree of the inheriting network
        self._compiled: dict[str, Callable]
        object.__setattr__(self, "_compiled", {})

    def _adam_fns(self) -> tuple[Callable, Callable]:
        if not self._compiled:
            loss_and_grad = nn.value_and_grad(self._model, lambda tree: self.tr_obj(Batch.from_tree(tree)))

            def step(tree: dict[str, Any]) -> mx.array:
                obj, grads = loss_and_grad(tree)
                self.opt.update(self._model, grads)
                return obj

            state = [self._model.state, self.opt.state]
            self._compiled["step"] = mx.compile(step, inputs=state, outputs=state)

            def obj(tree: dict[str, Any]) -> mx.array:
                return self.tr_obj(Batch.from_tree(tree))

            self._compiled["obj"] = mx.compile(obj, inputs=state)

            def steps(tree: dict[str, Any], indices: mx.array) -> tuple[mx.array, mx.array]:
                # one adam step per row of indices, the objective before and after the first one for the log line
                for j in range(indices.shape[0]):
                    batch = tree_map(lambda v, j=j: mx.take(v, indices[j], axis=0), tree)
                    step_obj = step(batch)
                    if j == 0:
                        pre_obj, post_obj = step_obj, obj(batch)
                return pre_obj, post_obj

            self._compiled["steps"] = mx.compile(steps, inputs=state, outputs=state)
        return self._compiled["step"], self._compiled["obj"]

    def perform_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """
        Performs multiple stochastic gradient descent steps.

        :param num_steps: The number of steps.
        :param tc: The trajectory collection to sample from.
        :param batch_size: The batch size to use for each batch.
        :param fields: The additional fields, which need to be sampled.
        """

        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        if num_steps == 0:
            return

        # the indices of all batches, drawn in the order and with the calls of one tc.sample per step
        replay = tc.flatten(fields)
        indices = mx.array(np.stack([replay.draw(batch_size) for _ in range(num_steps)]), dtype=mx.int32)
        self._adam_fns()
        logs = []
        for s in range(0, num_steps, STEPS_PER_CALL):
            pre_obj, post_obj = self._compiled["steps"](replay.tree, indices[s : s + STEPS_PER_CALL])
            mx.async_eval(pre_obj, post_obj, self._model.parameters(), self.opt.state)
            logs.append((s, pre_obj, post_obj))

        # the log lines of perform_adam_step, printed once all steps are queued
        for s, pre_obj, post_obj in logs:
            self.cli.print(self.fmtstr % (s, pre_obj.item(), post_obj.item()))

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
