from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.batch import Batch


class VanillaGradient(eqx.Module):
    """Simple vanilla gradient. Simply inherit from this class to add support.

    Value networks are fit with Adam on minibatches. The Adam state lives in the model state under
    ``<mc_key>_opt``; the printed table of the torch version is returned as metrics.
    """

    fmtstr: ClassVar[str] = "%10i | %14.10f | %14.10f"
    titlestr: ClassVar[str] = "%10s | %14s | %14s"
    lr: float = eqx.field(static=True)
    mc_key: str = eqx.field(static=True)

    def init_opt(self) -> optax.OptState:
        """The initial Adam state of this model."""
        return optax.adam(self.lr).init(eqx.filter(self, eqx.is_inexact_array))

    def perform_mult_sgd(
        self, state: ModelState, key: PRNGKeyArray, num_steps: int, tc: Any, batch_size: int, fields: list[str]
    ) -> tuple[ModelState, dict[str, Array]]:
        """
        Performs multiple stochastic gradient descent steps.

        :param state: The traced model state, holds this model under ``mc_key`` and its Adam state.
        :param key: The random key of the minibatch sampling.
        :param num_steps: The number of steps.
        :param tc: The trajectory collection to sample from.
        :param batch_size: The batch size to use for each batch.
        :param fields: The additional fields, which need to be sampled.
        :return: The new state and the pre / post objectives of every step (``titlestr`` columns).
        """
        opt = optax.adam(self.lr)

        def step(
            carry: tuple[Any, Any, PRNGKeyArray], s: Int[Array, ""]
        ) -> tuple[tuple[Any, Any, PRNGKeyArray], Array]:
            model, opt_state, key = carry
            key, sub = jax.random.split(key)
            batch = tc.sample(sub, batch_size, fields)
            model, opt_state, obj, new_obj = model.perform_adam_step(opt, opt_state, s, batch)
            return (model, opt_state, key), jnp.stack([obj, new_obj])

        init = (state.get(self.mc_key), state.get(f"{self.mc_key}_opt"), key)
        (model, opt_state, _), objs = jax.lax.scan(step, init, jnp.arange(num_steps))
        state = state.set(self.mc_key, model).set(f"{self.mc_key}_opt", opt_state)
        return state, {"k_pre_post": objs}

    def perform_adam_step(
        self, opt: optax.GradientTransformation, opt_state: optax.OptState, curst: Int[Array, ""] | int, batch: Batch
    ) -> tuple[Any, optax.OptState, Float[Array, ""], Float[Array, ""]]:
        """Performs a single adam step on the passed batch.

        :param opt: The Adam transformation.
        :param opt_state: Its state.
        :param curst: The current step.
        :param batch: The batch to use.
        :return: The updated model and Adam state, the objective before and (every 10th step) after the step.
        """
        obj, grads = eqx.filter_value_and_grad(lambda m: m.tr_obj(batch))(self)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(self, eqx.is_inexact_array))
        model = eqx.apply_updates(self, updates)
        new_obj = jax.lax.cond(curst % 10 == 0, lambda: model.tr_obj(batch), lambda: jnp.asarray(jnp.nan, obj.dtype))
        return model, opt_state, obj, new_obj

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """The training objective."""
        raise NotImplementedError
