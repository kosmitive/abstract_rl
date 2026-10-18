import numpy as np
import torch

from abstract_rl.misc.cli_printer import CliPrinter


def temperature_dual(q_tensor: torch.Tensor, eta: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Dual function of the non parametric E-step for a fixed temperature, see https://arxiv.org/abs/1806.06920.

        g(eta) = eta * eps + eta * mean_s log mean_a exp(Q(s, a) / eta)

    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eta: The temperature.
    :param eps: The KL bound.
    :return: The value of the dual function.
    """
    max_q = torch.max(q_tensor, dim=1, keepdim=True).values
    log_mean_exp = torch.logsumexp((q_tensor - max_q) / eta, dim=1) - np.log(q_tensor.shape[1])
    return eta * eps + torch.mean(max_q) + eta * torch.mean(log_mean_exp)


def find_optimal_temperature(
    q_tensor: torch.Tensor, eps: float, eta0: float = 1.0, lr: float = 0.05, max_steps: int = 200
) -> torch.Tensor:
    """
    Uses a batch of data to find the optimal temperature eta for reweighting by minimizing the convex dual with Adam.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: Adam learning rate on the log temperature. Defaults to 0.05.
    :param max_steps: Maximum number of Adam steps.
    :return: The found optimal temperature as a tensor of shape [1].
    """
    cli = CliPrinter().instance
    cli.empty().print("solve for q(a|s)").empty()
    with cli.indent():
        cli.print("max	\t\t∫ u(s) ∫ q(a|s) Q(s,a) dads")
        cli.print("st.	\t\t∫ u(s) KL(q(a|s), π(a|s)) ds < ε")
        cli.print("    \t\t∫∫ u(s) q(a|s) dads = 1")

    cli.empty().print("statistics of problem").empty()
    with cli.indent():
        cli.print(f"Q(s,a) \t~ {torch.mean(q_tensor):.5f} +- [{torch.std(q_tensor):.5f}]")
        cli.print(f"ε \t\t= {eps}")

    q_tensor = q_tensor.double()
    eta0_f = float(np.clip(eta0, 1e-6, None))
    log_eta = torch.tensor([np.log(eta0_f)], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_eta], lr=lr)
    obj = log_eta * 0
    for _ in range(max_steps):
        optimizer.zero_grad()
        eta = torch.exp(log_eta)
        obj = temperature_dual(q_tensor, eta, eps)
        obj.backward()
        optimizer.step()
    eta_v = float(torch.exp(log_eta).detach())

    with cli.indent():
        cli.empty().print(f"eta* = {eta_v:.6f} (dual = {float(obj.detach()):.6f}, iterations = {max_steps})")

    return torch.tensor([eta_v], dtype=torch.float32)
