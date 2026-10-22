from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to


class TrustRegionGradient:
    """Add trust region gradient based optimization to any class by inheriting this one."""

    def __init__(
        self,
        model: str,
        tr_eps: float,
        cg_max_k: int = 10,
        cg_residual_tol: float = 1e-6,
        dvp_damping: float = 1e-5,
        mode: str = "min",
    ) -> None:
        """Should be self explanatory

        :param model: The model state key of the model to optimize
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

        self.fmtstr = "%10i | %14.10f | %14.10f | %10.5f | %10.5f | %10.5f x %10.5f | %10.5f | %10.5f | %10.5f"
        self.titlestr = "%10s | %14s | %14s | %10s | %10s | %23s | %10s | %10s | %10s "

    def _at(self, state: ModelState, model: Any, flat: Float[Array, " n"]) -> ModelState:
        return state.set(self._model, set_flat_params_to(model, flat))

    def perform_tr_sgd(
        self, state: ModelState, curst: int, batch: Any, tr_batch: Any
    ) -> tuple[ModelState, dict[str, Array]]:

        # get v objective and the gradient
        model = state.get(self._model)
        x = get_flat_params_from(model)
        _, g = jax.value_and_grad(lambda flat: self.tr_obj(self._at(state, model, flat), batch))(x)

        # fisher vector products using direct mehtod (forward over reverse hessian vector product)
        def fvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
            grad_f = jax.grad(lambda flat: self.tr_distance(self._at(state, model, flat), tr_batch))
            r_hvp = jax.jvp(grad_f, (x,), (v,))[1]
            r_vp = r_hvp + self.dvp_damping * v
            return r_vp

        # calc natural gradient and set gradient for line search
        s = conjugate_gradient(fvp, -g, self.cg_max_k, self.cg_residual_tol)
        sdvps = (s * fvp(s)).sum()
        nat_metric = 0.5 * sdvps
        real_eps = self.tr_eps
        lm = jnp.sqrt(nat_metric / real_eps)

        # finish by line search
        neggdotstepdir = (-g * s).sum()
        accepted, step, lr, _, x_final, rows = self.line_search(state, batch, x, s / lm, neggdotstepdir / lm)
        state = self._at(state, model, x_final)
        return state, {"accepted": accepted, "step": step, "lr": lr, "rows": rows}

    def line_search(
        self,
        state: ModelState,
        batch: Any,
        x: Float[Array, " n"],
        fullstep: Float[Array, " n"],
        exp_improv_rate: Float[Array, ""],
        max_steps: int = 10,
        accept_ratio: float = 0.1,
        max_kl_ratio: float = 1.5,
    ) -> tuple[
        Bool[Array, ""], Int[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, " n"], Float[Array, "s 8"]
    ]:
        """Perform a line search.

        :param batch: The batch to use for the objective.
        :param fullstep: The fullstep, e.g the learing rate and the gradient
        :param exp_improv_rate: The expected improvement rate for the line search.
        :param max_steps: The maximum number of steps for the line search.
        :param accept_ratio: The acceptance ratio between actual and expected improvement rate.
        :param max_kl_ratio: Steps whose trust region distance exceeds max_kl_ratio * tr_eps are rejected, the
            quadratic approximation of the distance can underestimate it.
        :return: accepted, step, lr, objective, the final flat parameters (x if no step was accepted) and the printed
            table rows ("step", "old_obj", "obj", "lr", "ratio", "act_improv", "exp_improv", "distance").
        """
        model = state.get(self._model)
        old_obj = jax.lax.stop_gradient(self.tr_obj(self._at(state, model, x), batch))
        rows = jnp.full((max_steps, 8), jnp.nan, old_obj.dtype)

        # start with the learning rate continuously decay until the objective got better
        def cond(c: Any) -> Array:
            step, _, accepted, _, _ = c
            return (step < max_steps) & ~accepted

        def body(c: Any) -> Any:
            step, lr, _, _, rows = c
            # calc difference and revert this difference
            x_new = x + lr * fullstep
            new_state = self._at(state, model, x_new)

            # update objective and learnrate
            new_obj = self.tr_obj(new_state, batch)
            act_improv = old_obj - new_obj
            exp_improv = lr * exp_improv_rate
            ratio = act_improv / exp_improv
            distance = self.tr_distance(new_state, batch)
            rows = rows.at[step].set(
                jnp.stack([step.astype(old_obj.dtype), old_obj, new_obj, lr, ratio, act_improv, exp_improv, distance])
            )

            accepted = (ratio > accept_ratio) & (act_improv > 0) & (distance <= max_kl_ratio * self.tr_eps)
            next_step = jnp.where(accepted, step, step + 1)
            next_lr = jnp.where(accepted, lr, lr / 2)
            return next_step, next_lr, accepted, jnp.where(accepted, new_obj, old_obj), rows

        init = (jnp.asarray(0), jnp.asarray(1.0, old_obj.dtype), jnp.asarray(False), old_obj, rows)
        step, lr, accepted, obj, rows = jax.lax.while_loop(cond, body, init)

        # if line search was not successful the parameters stay at x
        x_final = jnp.where(accepted, x + lr * fullstep, x)
        return accepted, step, lr, obj, x_final, rows

    def tr_distance(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """The training distance."""
        raise NotImplementedError

    def tr_obj(self, state: ModelState, batch: Any) -> Float[Array, ""]:
        """The training objective."""
        raise NotImplementedError
