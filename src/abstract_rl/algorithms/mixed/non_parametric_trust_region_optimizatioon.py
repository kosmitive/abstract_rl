import math

import torch
import torch.optim as opt
from torch.autograd import Variable
import numpy as np

from abstract_rl.src.misc.cli_printer import CliPrinter


def find_optimal_temperature(q_tensor, eps, eta0=1.0, lr=0.005):
    """
    Uses a batch of data to find the optimal temperature eta for reweighting.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: The learn rate to use for the optimizer
    :return: The found optimal temperature.
    """
    cli = CliPrinter().instance
    cli.empty().print("solve for q(a|s)").empty()
    with cli.indent():
        cli.print("max	\t\t∫ u(s) ∫ q(a|s) Q(s,a) dads")
        cli.print("st.	\t\t∫ u(s) KL(q(a|s), π(a|s)) ds < ε")
        cli.print("    \t\t∫∫ u(s) q(a|s) dads = 1")

    cli.empty().print("statistics of problem").empty()
    with cli.indent():
        cli.print(f'Q(s,a) \t~ {torch.mean(q_tensor)} +- [{torch.std(q_tensor)}]')
        cli.print(f'ε \t\t= {torch.mean(q_tensor)} +- [{torch.std(q_tensor)}]')

    cli.empty().print("obtain optimal eta*").empty()
    with cli.indent():

        # obtain maximum q value
        max_q_val = torch.max(q_tensor)
        q_tensor = q_tensor - max_q_val

        # optimization specific
        eps = torch.Tensor([eps])
        log_eta = Variable(torch.Tensor([np.log(eta0)]), requires_grad=True)
        eta = torch.exp(log_eta)
        optimizer = torch.optim.Adam([log_eta], lr)
        n_eta = eta[0].detach().numpy()
        c_eta = n_eta + 1
        diff = np.abs(n_eta - c_eta)

        fmtstr = "%10i | %10.5f %10.5f | %10.5f"
        titlestr = "%10s | %10s %10s | %10s"
        cli.print(titlestr % ("k", "eta", "diff", "objective")).line()

        k = 0
        while diff > 1e-5:

            # set grad to zero
            optimizer.zero_grad()

            # build both integrals
            K, N = q_tensor.shape
            inner_integral = torch.mean(torch.exp(q_tensor / eta), dim=1)
            outer_integral = torch.mean(torch.log(inner_integral))

            # constrain objective
            obj = eta * eps + eta * outer_integral + K * max_q_val * math.log(N)
            obj.backward()

            if k % 20 == 0: cli.print(fmtstr % (k, n_eta, diff, obj))
            k += 1

            optimizer.step()
            eta = torch.exp(log_eta)
            c_eta = n_eta
            n_eta = eta[0].detach().numpy()
            diff = np.abs(n_eta - c_eta)

        # print final eta
        eta = torch.exp(log_eta)
        n_eta = eta[0].detach().numpy()
        cli.print(fmtstr % (k, n_eta, diff, obj))

    return eta