"""Fully connected state-value network, fit with minibatch Adam."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.flat_params import copy_module, ravel_module
from abstract_rl.misc.obs_norm import normalize_mc
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


def _f32(value: Any) -> Float[Array, "..."]:
    return jnp.asarray(value, dtype=jnp.float32)


class FullyVNeuralNet(Syncable, VanillaGradient):
    """State-value function. ``params`` is the Equinox network. Adam state lives on ``VanillaGradient``."""

    def __init__(self, mc: ModelConfiguration) -> None:
        conf = mc.get("conf")
        structure = [mc.env.observation_dim, *list(conf["struct"]), 1]
        self.mc = mc
        self.env = mc.env
        self.l2 = float(conf["l2_reg"])
        self.epochs = int(conf["epochs"])
        self.batch_size = int(conf["batch_size"])
        self.params = MultilayerPerceptron(
            mc.rng.split(),
            structure,
            conf["act_fn"],
            bool(conf.get("layer_norm", False)),
            str(conf.get("init", "orthogonal")),
        )
        self.net = self.params
        VanillaGradient.__init__(self, float(conf["lr"]))

    def clone(self) -> "FullyVNeuralNet":
        other = FullyVNeuralNet.__new__(FullyVNeuralNet)
        other.mc = self.mc
        other.env = self.env
        other.l2 = self.l2
        other.epochs = self.epochs
        other.batch_size = self.batch_size
        other.params = copy_module(self.params)
        other.net = other.params
        other.cli = self.cli
        other.lr = self.lr
        other.tx = self.tx
        other.opt_state = jax.tree.map(_copy_leaf, self.opt_state)
        other.fmtstr = self.fmtstr
        other.titlestr = self.titlestr
        return other

    def sync(self, v: "FullyVNeuralNet") -> "FullyVNeuralNet":
        v.params = self.params
        v.net = v.params
        return v

    def save(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self.params)

    def load(self, path: str) -> None:
        self.params = eqx.tree_deserialise_leaves(path, self.params)
        self.net = self.params

    def forward(self, states: Float[Array, "batch obs"] | np.ndarray) -> Float[Array, "batch 1"]:
        """Value of each state, shape ``[batch, 1]``."""
        return self.params.forward(normalize_mc(self.mc, states))

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Mean squared error plus an L2 penalty on the weights."""
        return self.objective_on(self.params, batch)

    def objective_on(self, params: MultilayerPerceptron, batch: Batch) -> Float[Array, ""]:
        """Value loss at an explicit parameter tree. Same expression as ``tr_obj``."""
        return _v_loss(params, normalize_mc(self.mc, batch.states), _f32(batch["tv"]).reshape(-1), self.l2)

    def perform_sgd(self, curst: int, batch: Batch) -> None:
        """Minibatch Adam on this rollout.

        Each pass shuffles the batch and takes ``n // batch_size`` steps. A rollout shorter than one
        minibatch is a single step on every transition. A trailing partial minibatch is left out.
        Adam state is kept across passes and across later calls. The logged losses are the full-batch
        value loss before the first step and after the last.
        """
        states = normalize_mc(self.mc, batch.states)
        targets = _f32(batch["tv"]).reshape(-1)
        n = int(states.shape[0])
        if n == 0:
            return
        size = min(int(self.batch_size), n)
        diff, static = eqx.partition(self.params, eqx.is_inexact_array)
        diff, self.opt_state, old, new = _fit_v(
            states,
            targets,
            diff,
            static,
            self.opt_state,
            self.tx,
            self.mc.rng.split(),
            jnp.float32(self.l2),
            int(self.epochs),
            n // size,
            size,
        )
        self.params = eqx.combine(diff, static)
        self.net = self.params
        self.cli.print(self.fmtstr % (curst, float(old), float(new)))


def _copy_leaf(value: object) -> object:
    if eqx.is_array(value):
        return jnp.array(value)
    return value


@eqx.filter_jit
def _fit_v(
    states: Float[Array, "batch obs"],
    targets: Float[Array, " batch"],
    diff: Any,
    static: Any,
    opt_state: Any,
    tx: Any,
    key: Array,
    l2: Float[Array, ""],
    epochs: int,
    n_batches: int,
    batch_size: int,
) -> tuple[Any, Any, Float[Array, ""], Float[Array, ""]]:
    """Shuffle the rollout for ``epochs`` passes and take ``n_batches`` Adam steps each pass.

    ``epochs``, ``n_batches`` and ``batch_size`` are concrete, so a new rollout size compiles once.
    ``l2`` is an array so a new penalty does not. The returned losses are the full-batch value loss
    before the first step and after the last.
    """
    old = _v_loss(eqx.combine(diff, static), states, targets, l2)

    def epoch(carry: tuple[Any, Any, Array], _: None) -> tuple[tuple[Any, Any, Array], None]:
        weights, state, key_ = carry
        key_, sub = jax.random.split(key_)
        perm = jax.random.permutation(sub, states.shape[0])

        def step(carry2: tuple[Any, Any], index: Array) -> tuple[tuple[Any, Any], None]:
            weights, state = carry2
            start = index * batch_size
            chosen = jax.lax.dynamic_slice(perm, (start,), (batch_size,))

            def loss(weights_: Any) -> Float[Array, ""]:
                return _v_loss(eqx.combine(weights_, static), states[chosen], targets[chosen], l2)

            _value, grads = eqx.filter_value_and_grad(loss)(weights)
            updates, state = tx.update(grads, state, weights)
            weights = eqx.apply_updates(weights, updates)
            return (weights, state), None

        (weights, state), _ = jax.lax.scan(step, (weights, state), jnp.arange(n_batches))
        return (weights, state, key_), None

    (diff, opt_state, _key), _ = jax.lax.scan(epoch, (diff, opt_state, key), None, length=epochs)
    new = _v_loss(eqx.combine(diff, static), states, targets, l2)
    return diff, opt_state, old, new


def _v_loss(
    network: MultilayerPerceptron,
    states: Float[Array, "batch obs"],
    targets: Float[Array, " batch"],
    l2: float | Float[Array, ""],
) -> Float[Array, ""]:
    prediction = network.forward(states).reshape(-1)
    flat, _ = ravel_module(network)
    return jnp.mean(jnp.square(prediction - targets)) + l2 * jnp.linalg.norm(flat)
