import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int

from abstract_rl.misc.typing import DTYPE


def _mean(x: Float[Array, "N"], mask: Bool[Array, "N"] | None) -> Float[Array, ""]:
    return jnp.mean(x) if mask is None else jnp.sum(jnp.where(mask, x, 0.0)) / jnp.sum(mask)


def temperature_dual(
    q_tensor: Float[Array, "N K"], eta: Float[Array, "1"], eps: float, mask: Bool[Array, "N"] | None = None
) -> Float[Array, "1"]:
    """
    Dual function of the non parametric E-step for a fixed temperature, see https://arxiv.org/abs/1806.06920.

        g(eta) = eta * eps + eta * mean_s log mean_a exp(Q(s, a) / eta)

    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eta: The temperature.
    :param eps: The KL bound.
    :param mask: The valid rows of a padded memory, all rows if None.
    :return: The value of the dual function.
    """
    max_q = jnp.max(q_tensor, axis=1, keepdims=True)
    log_mean_exp = jax.scipy.special.logsumexp((q_tensor - max_q) / eta, axis=1) - np.log(q_tensor.shape[1])
    return eta * eps + _mean(max_q[:, 0], mask) + eta * _mean(log_mean_exp, mask)


def find_optimal_temperature(
    q_tensor: Float[Array, "N K"],
    eps: float,
    eta0: float | Float[Array, ""] = 1.0,
    lr: float = 0.05,
    max_steps: int = 200,
    mask: Bool[Array, "N"] | None = None,
) -> Float[Array, "1"]:
    """
    Uses a batch of data to find the optimal temperature eta for reweighting by minimizing the convex dual with Adam.
    The statistics of the problem, eta* and the dual value are printed on the host from the returned metrics
    (NonParametricVariationalPolicyBuilder.calc_eta), not inside the traced solve.
    :param q_tensor: A tensor of [NxA] with N states and A actions.
    :param eps: The eps to use.
    :param eta0: The initial eta
    :param lr: Adam learning rate on the log temperature. Defaults to 0.05.
    :param max_steps: Maximum number of Adam steps.
    :param mask: The valid rows of a padded memory, all rows if None.
    :return: The found optimal temperature as a tensor of shape [1].
    """
    # D4: the dual runs in float32 (the torch version used float64)
    q_tensor = jnp.where(mask[:, None], q_tensor, 0.0) if mask is not None else q_tensor
    q_tensor = q_tensor.astype(DTYPE)
    eta0_f = jnp.clip(jnp.asarray(eta0, DTYPE), 1e-6, None)
    log_eta = jnp.log(eta0_f).reshape(1)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_eta)

    def step(
        _: Int[Array, ""], carry: tuple[Float[Array, "1"], optax.OptState]
    ) -> tuple[Float[Array, "1"], optax.OptState]:
        log_eta, opt_state = carry
        grad = jax.grad(lambda le: temperature_dual(q_tensor, jnp.exp(le), eps, mask)[0])(log_eta)
        updates, opt_state = optimizer.update(grad, opt_state)
        return optax.apply_updates(log_eta, updates), opt_state

    log_eta, _ = jax.lax.fori_loop(0, max_steps, step, (log_eta, opt_state))
    return jnp.exp(log_eta).astype(DTYPE)
