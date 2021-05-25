import scipy as sp

import torch
import torch.optim as optim
from abstract_rl.src.misc.cli_printer import CliPrinter
from abstract_rl.src.misc.flat_params import set_flat_params_to, get_flat_grad_from, get_flat_params_from

import scipy.optimize
import scipy as sp

class VanillaGradient:
    """Simple vanilla gradient. Simply inherit from this class to add support"""

    def __init__(self, model):
        """
        Add vanilla gradient optimization.
        :param model: The pytorch model to use.
        """
        self._model = model
        self.cli = CliPrinter().instance

        self.fmtstr = "%10i | %14.10f | %14.10ff"
        self.titlestr = "%10s | %14s | %14s"
        self.opt = torch.optim.Adam(self._model.parameters())

    def perform_mult_sgd(self, num_steps, tc, batch_size, fields):
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
            self.perform_sgd(s, batch, 1)

    def perform_sgd(self, curst, batch, maxiter=25):
        """Performs a stochastic gradient descent step by using lbfgs.

        :param curst: The current step.
        :param batch: The batch to use.
        :param maxiter: Maximum of iterations.
        """

        # get v objective and the gradient
        old_obj = self.tr_obj(batch)

        def closure(x):
            set_flat_params_to(self._model, torch.Tensor(x))
            obj = self.tr_obj(batch)
            obj.backward()
            return (obj.data.double().numpy(), get_flat_grad_from(self._model).data.double().numpy())

        # start optimizer
        params, _, opt_info = sp.optimize.fmin_l_bfgs_b(closure, get_flat_params_from(self._model).double().numpy(), maxiter=maxiter)
        set_flat_params_to(self._model, torch.Tensor(params))

        # calc new obj for comparison
        new_obj = self.tr_obj(batch)
        self.cli.print(self.fmtstr % (curst, old_obj, new_obj))

    def tr_obj(self, batch): pass