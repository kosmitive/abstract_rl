import math
from typing import cast

import mlx.core as mx
import numpy as np

from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.mixed import on_gpu


def temperature_dual(q_tensor: mx.array, eta: mx.array, eps: float | mx.array) -> mx.array:
    """
    Dual function of the non parametric E-step for a fixed temperature, see https://arxiv.org/abs/1806.06920.

        g(eta) = eta * eps + eta * mean_s log mean_a exp(Q(s, a) / eta)

    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eta: The temperature.
    :param eps: The KL bound.
    :return: The value of the dual function.
    """
    max_q = mx.max(q_tensor, axis=1, keepdims=True)
    log_mean_exp = mx.logsumexp((q_tensor - max_q) / eta, axis=1) - math.log(q_tensor.shape[1])
    return eta * eps + mx.mean(max_q) + eta * mx.mean(log_mean_exp)


def _adam_dual_step(
    q_tensor: mx.array,
    eps: mx.array,
    lr: mx.array,
    log_eta: mx.array,
    m: mx.array,
    v: mx.array,
    step: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One Adam step on the log temperature, with the update rule of torch.optim.Adam (betas 0.9, 0.999, eps 1e-8)."""

    def dual(le: mx.array) -> mx.array:
        return temperature_dual(q_tensor, mx.exp(le), eps).sum()

    obj, g = mx.value_and_grad(dual)(log_eta)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    bias_correction1 = 1 - 0.9**step
    bias_correction2 = 1 - 0.999**step
    denom = mx.sqrt(v) / mx.sqrt(bias_correction2) + 1e-8
    log_eta = log_eta - (lr / bias_correction1) * m / denom
    return log_eta, m, v, obj


_compiled_dual_step = mx.compile(_adam_dual_step)


def find_optimal_temperature(
    q_tensor: mx.array, eps: float, eta0: float = 1.0, lr: float = 0.05, max_steps: int = 200
) -> mx.array:
    """
    Uses a batch of data to find the optimal temperature eta for reweighting by minimizing the convex dual with Adam.
    The dual is solved in float64 on the CPU, Metal has no float64, so on the GPU it is solved in float32.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: Adam learning rate on the log temperature. Defaults to 0.05.
    :param max_steps: Maximum number of Adam steps.
    :return: The found optimal temperature as a float32 array of shape [1].
    """
    cli = CliPrinter().instance
    cli.empty().print("solve for q(a|s)").empty()
    with cli.indent():
        cli.print("max	\t\t∫ u(s) ∫ q(a|s) Q(s,a) dads")
        cli.print("st.	\t\t∫ u(s) KL(q(a|s), π(a|s)) ds < ε")
        cli.print("    \t\t∫∫ u(s) q(a|s) dads = 1")

    cli.empty().print("statistics of problem").empty()
    with cli.indent():
        cli.print(f"Q(s,a) \t~ {mx.mean(q_tensor).item():.5f} +- [{mx.std(q_tensor, ddof=1).item():.5f}]")
        cli.print(f"ε \t\t= {eps}")

    dtype = mx.float32 if on_gpu() else mx.float64
    q_tensor = q_tensor.astype(dtype)
    eta0_f = float(np.clip(eta0, 1e-6, None))
    log_eta = mx.array([math.log(eta0_f)], dtype=dtype)
    m = mx.zeros_like(log_eta)
    v = mx.zeros_like(log_eta)
    eps_a, lr_a = mx.array(eps, dtype=dtype), mx.array(lr, dtype=dtype)
    obj = log_eta * 0

    # the steps are built lazily and evaluated at once
    for step in range(1, max_steps + 1):
        log_eta, m, v, obj = _compiled_dual_step(q_tensor, eps_a, lr_a, log_eta, m, v, mx.array(step, dtype=dtype))
    mx.eval(log_eta, obj)
    eta_v = math.exp(cast(float, log_eta.item()))

    with cli.indent():
        cli.empty().print(f"eta* = {eta_v:.6f} (dual = {obj.item():.6f}, iterations = {max_steps})")

    return mx.array([eta_v], dtype=mx.float32)
