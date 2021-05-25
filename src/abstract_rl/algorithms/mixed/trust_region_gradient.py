import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable

from abstract_rl.src.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.src.misc.cli_printer import CliPrinter
from abstract_rl.src.misc.flat_params import get_flat_params_from, set_flat_params_to


class TrustRegionGradient:
    """Add trust region gradient based optimization to any class by inheriting this one.
    """

    def __init__(self, model, tr_eps, cg_max_k=10, cg_residual_tol=1e-6, dvp_damping=1e-5, mode='min'):
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

        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def perform_mult_tr_sgd(self, num_steps, tc, batch_size, fields):

        for s in range(num_steps):
            batch = tc.sample(batch_size, fields)
            self.perform_tr_sgd(s, batch, batch)

    def perform_tr_sgd(self, curst, batch, tr_batch):

        # get v objective and the gradient
        old_obj = self.tr_obj(batch)
        x = get_flat_params_from(self._model)
        g = autograd.grad(old_obj, self._model.parameters())
        g = torch.cat([el.view(-1) for el in g]).data

        # fisher vector products using direct mehtod
        def fvp(v):

            # parameters
            v_obj = self.tr_distance(tr_batch)
            grad_f = torch.autograd.grad(v_obj, self._model.parameters(), create_graph=True)
            grad_f = torch.cat([el.view(-1) for el in grad_f])
            z = (grad_f * Variable(v)).sum()
            grad_grad = torch.autograd.grad(z, self._model.parameters())
            r_hvp = torch.cat([grad.contiguous().view(-1) for grad in grad_grad])
            r_vp = r_hvp.data + self.dvp_damping * v
            return r_vp

        # calc natural gradient and set gradient for line search
        s = conjugate_gradient(fvp, -g, self.cg_max_k, self.cg_residual_tol)
        sdvps = (s * fvp(s)).sum()
        nat_metric = 0.5 * sdvps
        real_eps = self.tr_eps
        lm = np.sqrt(nat_metric / real_eps)

        # finish by line search
        neggdotstepdir = (-g * s).sum()
        self.line_search(batch, x, s / lm, neggdotstepdir / lm)

    def line_search(self, batch, x, fullstep, exp_improv_rate, max_steps=10, accept_ratio=0.1):
        """Perform a line search.

        :param batch: The batch to use for the objective.
        :param fullstep: The fullstep, e.g the learing rate and the gradient
        :param exp_improv_rate: The expected improvement rate for the line search.
        :param max_steps: The maximum number of steps for the line search.
        :param accept_ratio: The acceptance ratio between actual and expected improvement rate.
        """

        fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %14.10f | %14.10f | %14.10f"
        titlestr = "%10s | %14s | %14s | %10s | %14s | %14s | %14s"

        self.cli.print(titlestr % ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv")).line()

        with torch.no_grad():

            old_obj = self.tr_obj(batch).data
            old_obj = old_obj.detach().numpy()

            # start with the learning rate continuously decay until the objective
            # got better
            step = 0
            lr = 1.0
            while step < max_steps:

                # calc difference and revert this difference
                x_new = x + lr * fullstep
                set_flat_params_to(self._model, x_new)

                # update objective and learnrate
                new_obj = self.tr_obj(batch).data
                new_obj = new_obj.detach().numpy()

                act_improv = old_obj - new_obj
                exp_improv = lr * exp_improv_rate
                ratio = act_improv / exp_improv

                self.cli.print(fmtstr % (step, old_obj, new_obj, lr, ratio, act_improv, exp_improv))

                if ratio > accept_ratio and act_improv > 0:
                    return True, step, lr, new_obj

                step += 1
                lr = lr / 2

            # if line search was not successful
            self.cli.line().print("FAILED")
            set_flat_params_to(self._model, x)
            return False, step, lr, new_obj

    def tr_distance(self, batch):
        """The training distance."""
        pass

    def tr_obj(self, batch):
        """The training objective."""
        pass