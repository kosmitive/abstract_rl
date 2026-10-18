import torch

from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.misc.cli_printer import CliPrinter


class VanillaGradient:
    """Simple vanilla gradient. Simply inherit from this class to add support.

    Value networks are fit with Adam on minibatches.
    """

    def __init__(self, model: torch.nn.Module, lr: float = 1e-3) -> None:
        """
        Add vanilla gradient optimization.
        :param model: The pytorch model to use.
        :param lr: The Adam learning rate.
        """
        self._model = model
        self.cli = CliPrinter().instance

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
            self.perform_adam_step(s, batch)

    def perform_adam_step(self, curst: int, batch: Batch) -> None:
        """Performs a single adam step on the passed batch.

        :param curst: The current step.
        :param batch: The batch to use.
        """
        self.opt.zero_grad()
        obj = self.tr_obj(batch)
        obj.backward()
        self.opt.step()
        if curst % 10 == 0:
            new_obj = self.tr_obj(batch)
            self.cli.print(self.fmtstr % (curst, obj.item(), new_obj.item()))

    def tr_obj(self, batch: Batch) -> torch.Tensor:
        """The training objective."""
        raise NotImplementedError
