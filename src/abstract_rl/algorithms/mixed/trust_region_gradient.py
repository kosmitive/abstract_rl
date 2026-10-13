import torch
import torch.autograd as autograd
from torch import nn

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to


class TrustRegionGradient:
    """Add trust region gradient based optimization to any class by inheriting this one."""

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

        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def perform_tr_sgd(self, curst: int, batch: Batch, tr_batch: Batch) -> None:

        # get v objective and the gradient
        old_obj = self.tr_obj(batch)
        x = get_flat_params_from(self._model)
        g_parts = autograd.grad(old_obj, list(self._model.parameters()))
        g = torch.cat([el.reshape(-1) for el in g_parts]).detach()

        # fisher vector products using direct mehtod
        def fvp(v: torch.Tensor) -> torch.Tensor:

            # parameters
            v_obj = self.tr_distance(tr_batch)
            grad_parts = torch.autograd.grad(v_obj, list(self._model.parameters()), create_graph=True)
            grad_f = torch.cat([el.reshape(-1) for el in grad_parts])
            z = (grad_f * v).sum()
            grad_grad = torch.autograd.grad(z, list(self._model.parameters()))
            r_hvp = torch.cat([grad.contiguous().view(-1) for grad in grad_grad])
            r_vp = r_hvp.detach() + self.dvp_damping * v
            return r_vp

        # calc natural gradient and set gradient for line search
        s = conjugate_gradient(fvp, -g, self.cg_max_k, self.cg_residual_tol)
        sdvps = (s * fvp(s)).sum()
        nat_metric = 0.5 * sdvps
        real_eps = self.tr_eps
        lm = torch.sqrt(nat_metric / real_eps)

        # finish by line search
        neggdotstepdir = (-g * s).sum()
        self.line_search(batch, x, s / lm, neggdotstepdir / lm)

    def line_search(
        self,
        batch: Batch,
        x: torch.Tensor,
        fullstep: torch.Tensor,
        exp_improv_rate: torch.Tensor,
        max_steps: int = 10,
        accept_ratio: float = 0.1,
        max_kl_ratio: float = 1.5,
    ) -> tuple[bool, int, float, float]:
        """Perform a line search.

        :param batch: The batch to use for the objective.
        :param fullstep: The fullstep, e.g the learing rate and the gradient
        :param exp_improv_rate: The expected improvement rate for the line search.
        :param max_steps: The maximum number of steps for the line search.
        :param accept_ratio: The acceptance ratio between actual and expected improvement rate.
        :param max_kl_ratio: Steps whose trust region distance exceeds max_kl_ratio * tr_eps are rejected, the
            quadratic approximation of the distance can underestimate it.
        """

        fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %14.10f | %14.10f | %14.10f | %10.6f"
        titlestr = "%10s | %14s | %14s | %10s | %14s | %14s | %14s | %10s"

        self.cli.print(
            titlestr % ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv", "distance")
        ).line()

        with torch.no_grad():
            old_obj = self.tr_obj(batch).item()

            # start with the learning rate continuously decay until the objective
            # got better
            step = 0
            lr = 1.0
            while step < max_steps:
                # calc difference and revert this difference
                x_new = x + lr * fullstep
                set_flat_params_to(self._model, x_new)

                # update objective and learnrate
                new_obj = self.tr_obj(batch).item()

                act_improv = old_obj - new_obj
                exp_improv = lr * float(exp_improv_rate)
                ratio = act_improv / exp_improv
                distance = self.tr_distance(batch).item()

                self.cli.print(fmtstr % (step, old_obj, new_obj, lr, ratio, act_improv, exp_improv, distance))

                if ratio > accept_ratio and act_improv > 0 and distance <= max_kl_ratio * self.tr_eps:
                    return True, step, lr, new_obj

                step += 1
                lr = lr / 2

            # if line search was not successful
            self.cli.line().print("FAILED")
            set_flat_params_to(self._model, x)
            return False, step, lr, old_obj

    def tr_distance(self, batch: Batch) -> torch.Tensor:
        """The training distance."""
        raise NotImplementedError

    def tr_obj(self, batch: Batch) -> torch.Tensor:
        """The training objective."""
        raise NotImplementedError
