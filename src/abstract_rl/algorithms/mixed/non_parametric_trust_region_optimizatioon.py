import numpy as np
import scipy.optimize
import torch
from numpy import float32

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
    return eta * eps + torch.mean(max_q.view(-1)) + eta * torch.mean(log_mean_exp)


def find_optimal_temperature(
    q_tensor: torch.Tensor, eps: float, eta0: float32 | float = 1.0, lr: float | None = None, max_steps: int = 200
) -> torch.Tensor:
    """
    Uses a batch of data to find the optimal temperature eta for reweighting by minimizing the convex dual.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: Unused, kept for backwards compatibility (the dual is solved with L-BFGS-B).
    :param max_steps: Maximum number of iterations of the solver.
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

    def fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        eta = torch.tensor(x[0], dtype=torch.float64, requires_grad=True)
        obj = temperature_dual(q_tensor, eta, eps)
        obj.backward()
        assert eta.grad is not None
        return obj.item(), np.array([eta.grad.item()])

    eta0 = float(np.clip(np.asarray(eta0).reshape(-1)[0], 1e-6, None))
    res = scipy.optimize.minimize(
        fn, np.array([eta0]), jac=True, method="L-BFGS-B", bounds=[(1e-6, None)], options={"maxiter": max_steps}
    )
    eta = float(res.x[0])

    with cli.indent():
        cli.empty().print(f"eta* = {eta:.6f} (dual = {res.fun:.6f}, iterations = {res.nit})")

    return torch.tensor([eta], dtype=torch.float32)
