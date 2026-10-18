import numpy as np
import scipy.optimize
import torch
from torch import nn

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_grad_from, get_flat_params_from, set_flat_params_to


class VanillaGradient:
    """Simple vanilla gradient. Simply inherit from this class to add support.

    Two optimizers are supported: full-batch L-BFGS (scipy) and Adam on mini batches.
    """

    def __init__(
        self, model: nn.Module, optimizer: str = "lbfgs", lr: float = 1e-3, grad_clip: float | None = None
    ) -> None:
        """
        Add vanilla gradient optimization.
        :param model: The pytorch model to use.
        :param optimizer: Either 'lbfgs' or 'adam'.
        :param lr: The learning rate used by adam.
        :param grad_clip: Global-norm clip applied after an Adam backward. ``None`` leaves the gradient unchanged.
        """
        self._model = model
        self.cli = CliPrinter().instance
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        self.fmtstr = "%10i | %14.10f | %14.10f"
        self.titlestr = "%10s | %14s | %14s"
        self.opt = torch.optim.Adam(self._model.parameters(), lr=lr)

    def perform_mult_sgd(self, num_steps: int, tc: TrajectoryCollection, batch_size: int, fields: list[str]) -> None:
        """
        Performs multiple stochastic gradient descent steps.

        :param num_steps: The number of steps.
        :param tc: The trajectory collection to sample from.
        :param batch_size: The batch size to use for each batch.
        :param fields: The additional fields, which need to be sampled.
        """

        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()

        for s in range(num_steps):
            batch = tc.sample(batch_size, fields)
            if self.optimizer == "adam":
                self.perform_adam_step(s, batch)
            else:
                self.perform_sgd(s, batch, 1)

    def perform_adam_step(self, curst: int, batch: Batch) -> None:
        """Performs a single adam step on the passed batch.

        :param curst: The current step.
        :param batch: The batch to use.
        """
        self.opt.zero_grad()
        obj = self.tr_obj(batch)
        obj.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
        self.opt.step()
        if curst % 10 == 0:
            with torch.no_grad():
                new_obj = self.tr_obj(batch)
            self.cli.print(self.fmtstr % (curst, obj.item(), new_obj.item()))

    def perform_sgd(self, curst: int, batch: Batch, maxiter: int = 25) -> None:
        """Performs a stochastic gradient descent step by using lbfgs.

        :param curst: The current step.
        :param batch: The batch to use.
        :param maxiter: Maximum of iterations.
        """

        # get v objective and the gradient
        with torch.no_grad():
            old_obj = self.tr_obj(batch).item()

        def closure(x: np.ndarray) -> tuple[float, np.ndarray]:
            set_flat_params_to(self._model, torch.as_tensor(x, dtype=torch.float32))
            self._model.zero_grad()
            obj = self.tr_obj(batch)
            obj.backward()
            return obj.item(), get_flat_grad_from(self._model).double().numpy()

        # start optimizer
        params, _, _ = scipy.optimize.fmin_l_bfgs_b(
            closure, get_flat_params_from(self._model).double().numpy(), maxiter=maxiter
        )
        set_flat_params_to(self._model, torch.as_tensor(params, dtype=torch.float32))

        # calc new obj for comparison
        with torch.no_grad():
            new_obj = self.tr_obj(batch).item()
        self.cli.print(self.fmtstr % (curst, old_obj, new_obj))

    def tr_obj(self, batch: Batch) -> torch.Tensor:
        """The training objective."""
        raise NotImplementedError
