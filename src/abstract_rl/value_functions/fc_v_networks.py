"""Fully connected state-value network, fit with L-BFGS."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient, lbfgs_minimize
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.flat_params import copy_module, ravel_module
from abstract_rl.misc.obs_norm import normalize_mc
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron


def _f32(value: Any) -> Float[Array, "..."]:
    return jnp.asarray(value, dtype=jnp.float32)


class FullyVNeuralNet(Syncable, VanillaGradient):
    """State-value function. ``params`` is the Equinox network. L-BFGS comes from ``VanillaGradient``."""

    def __init__(self, mc: ModelConfiguration) -> None:
        conf = mc.get("conf")
        structure = [mc.env.observation_dim, *list(conf["struct"]), 1]
        self.mc = mc
        self.env = mc.env
        self.l2 = float(conf["l2_reg"])
        self.params = MultilayerPerceptron(
            mc.rng.split(),
            structure,
            conf["act_fn"],
            bool(conf.get("layer_norm", False)),
            str(conf.get("init", "orthogonal")),
        )
        self.net = self.params
        VanillaGradient.__init__(self)

    def clone(self) -> "FullyVNeuralNet":
        other = FullyVNeuralNet.__new__(FullyVNeuralNet)
        other.mc = self.mc
        other.env = self.env
        other.l2 = self.l2
        other.params = copy_module(self.params)
        other.net = other.params
        other.cli = self.cli
        other.optimizer = self.optimizer
        other.lr = self.lr
        other.tx = self.tx
        other.opt_state = self.opt_state
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

    def v_val(self, states: Float[Array, "batch obs"] | np.ndarray) -> Float[Array, "batch 1"]:
        return self.forward(states)

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Mean squared error plus an L2 penalty on the weights."""
        return self.objective_on(self.params, batch)

    def objective_on(self, params: MultilayerPerceptron, batch: Batch) -> Float[Array, ""]:
        """Value loss at an explicit parameter tree. Same expression as ``tr_obj``."""
        return _v_loss(params, normalize_mc(self.mc, batch.states), _f32(batch["tv"]).reshape(-1), self.l2)

    def perform_sgd(self, curst: int, batch: Batch, maxiter: int = 25) -> None:
        """L-BFGS on this batch.

        The whole fit is one jitted scan. States and targets are arguments, so a new epoch with the
        same shapes does not compile again.
        """
        states = normalize_mc(self.mc, batch.states)
        targets = _f32(batch["tv"]).reshape(-1)
        flat0, unravel = ravel_module(self.params)
        updated, old, new = _fit_v(self.params, states, targets, flat0, jnp.float32(self.l2), int(maxiter))
        self.params = unravel(updated)
        self.net = self.params
        self.cli.print(self.fmtstr % (curst, float(old), float(new)))


@eqx.filter_jit
def _fit_v(
    network: MultilayerPerceptron,
    states: Float[Array, "batch obs"],
    targets: Float[Array, " batch"],
    vector: Float[Array, " params"],
    l2: Float[Array, ""],
    steps: int,
) -> tuple[Float[Array, " params"], Float[Array, ""], Float[Array, ""]]:
    """Flat L-BFGS fit. ``steps`` is static. ``l2`` is an array so a new penalty does not recompile."""
    _flat, unravel = ravel_module(network)

    def loss(candidate: Float[Array, " params"]) -> Float[Array, ""]:
        return _v_loss(unravel(candidate), states, targets, l2)

    return lbfgs_minimize(loss, vector, steps)


def _v_loss(
    network: MultilayerPerceptron,
    states: Float[Array, "batch obs"],
    targets: Float[Array, " batch"],
    l2: float | Float[Array, ""],
) -> Float[Array, ""]:
    prediction = network.forward(states).reshape(-1)
    flat, _ = ravel_module(network)
    return jnp.mean(jnp.square(prediction - targets)) + l2 * jnp.linalg.norm(flat)
