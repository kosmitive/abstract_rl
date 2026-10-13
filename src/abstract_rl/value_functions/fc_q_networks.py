"""Fully connected Q network, fit with Adam."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.algorithms.mixed.vanilla_gradient import VanillaGradient
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.data_structures.temporal_difference_data.replay_buffer import ReplayBuffer
from abstract_rl.misc.flat_params import copy_module
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.gaussian_math import GaussianParams, sample


def _f32(value: Any) -> Float[Array, "..."]:
    return jnp.asarray(value, dtype=jnp.float32)


def q_forward(
    network: MultilayerPerceptron,
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch act"],
) -> Float[Array, "batch 1"]:
    """Q(s, a) for one action per state."""
    return network.forward(jnp.concatenate([states, actions], axis=-1))


def q_values(
    network: MultilayerPerceptron,
    states: Float[Array, "batch obs"],
    actions: Float[Array, "batch k act"],
) -> Float[Array, "batch k"]:
    """Q values of K actions per state.

    The action axis is folded into the batch so the network runs as one matmul. The values
    match a map over that axis.
    """
    batch, k, _act = actions.shape
    flat_states = jnp.broadcast_to(states[:, None, :], (batch, k, states.shape[-1])).reshape(
        (batch * k, states.shape[-1])
    )
    flat_actions = actions.reshape((batch * k, actions.shape[-1]))
    return q_forward(network, flat_states, flat_actions).reshape((batch, k))


def estimate_v(
    network: MultilayerPerceptron,
    policy: GaussianParams,
    states: Float[Array, "batch obs"],
    key: PRNGKeyArray,
    n_actions: int,
) -> Float[Array, " batch"]:
    """Monte Carlo estimate of V(s) under ``policy`` using the Q network."""
    actions = sample(policy, key, n_actions, states=states)
    return q_values(network, states, actions).mean(axis=1)


class FullyQNeuralNet(Syncable, VanillaGradient):
    """Action-value function. Adam state is kept across epochs, as in ``VanillaGradient``."""

    def __init__(self, mc: ModelConfiguration) -> None:
        conf = mc.get("conf")
        structure = [mc.env.observation_dim + mc.env.action_dim, *list(conf["struct"]), 1]
        self.mc = mc
        self.env = mc.env
        self.action_dim = mc.env.action_dim
        self.params = MultilayerPerceptron(
            mc.rng.split(),
            structure,
            conf["act_fn"],
            bool(conf.get("layer_norm", False)),
            str(conf.get("init", "orthogonal")),
        )
        # ``fc_net`` is the historical name of this module. ``params`` is the Equinox leaf the optimiser writes.
        self.fc_net = self.params
        VanillaGradient.__init__(self, str(conf.get("optimizer", "adam")), float(conf.get("lr", 1e-3)))

    def clone(self) -> "FullyQNeuralNet":
        other = FullyQNeuralNet.__new__(FullyQNeuralNet)
        other.mc = self.mc
        other.params = copy_module(self.params)
        other.fc_net = other.params
        other.env = self.env
        other.action_dim = self.action_dim
        other.lr = self.lr
        other.tx = self.tx
        other.opt_state = None if self.opt_state is None else jax.tree.map(_copy_leaf, self.opt_state)
        other.optimizer = self.optimizer
        other.cli = self.cli
        other.fmtstr = self.fmtstr
        other.titlestr = self.titlestr
        return other

    def sync(self, v: "FullyQNeuralNet") -> "FullyQNeuralNet":
        v.params = self.params
        v.fc_net = v.params
        return v

    def save(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self.params)

    def load(self, path: str) -> None:
        self.params = eqx.tree_deserialise_leaves(path, self.params)
        self.fc_net = self.params

    def forward(
        self, states: Float[Array, "batch obs"] | np.ndarray, actions: Float[Array, "batch act"] | np.ndarray
    ) -> Float[Array, "batch 1"]:
        """Q(s, a), the historical ``forward`` of this module."""
        return self.q_val(states, actions)

    def tr_obj(self, batch: Batch) -> Float[Array, ""]:
        """Mean squared error between Q(s, a) and the Retrace target ``tq``."""
        return self.objective_on(self.params, batch)

    def objective_on(self, params: MultilayerPerceptron, batch: Batch) -> Float[Array, ""]:
        """Critic loss at an explicit parameter tree. Same expression as ``tr_obj``."""
        prediction = q_forward(params, _f32(batch.states), _actions(batch.actions, self.action_dim)).reshape(-1)
        target = _f32(batch["tq"]).reshape(-1)
        return jnp.mean(jnp.square(prediction - target))

    def _scan_adam(self, replay: ReplayBuffer, num_steps: int, batch_size: int) -> None:
        """Adam on minibatches drawn on device, the whole loop in one ``lax.scan``."""
        if replay.size == 0:
            return
        diff, static = eqx.partition(self.params, eqx.is_inexact_array)
        diff, self.opt_state, pre, post = _critic_scan(
            (replay.states, replay.actions, replay.tq, self.mc.rng.split(), jnp.int32(replay.size)),
            diff,
            static,
            self.opt_state,
            self.tx,
            num_steps,
            batch_size,
        )
        self.params = eqx.combine(diff, static)
        self.fc_net = self.params
        self.cli.print(self.titlestr % ("k", "pre_objective", "post_objective")).line()
        pre_np, post_np = np.asarray(pre), np.asarray(post)
        for step in range(num_steps):
            if step % 10 == 0:
                self.cli.print(self.fmtstr % (step, pre_np[step], post_np[step]))

    def q_val(
        self, states: Float[Array, "batch obs"] | np.ndarray, actions: Float[Array, "batch act"] | np.ndarray
    ) -> Float[Array, "batch 1"]:
        return q_forward(self.params, _f32(states), _actions(actions, self.action_dim))

    def q_val_multi(
        self, states: Float[Array, "batch obs"] | np.ndarray, actions: Float[Array, "batch k ..."] | np.ndarray
    ) -> Float[Array, "batch k"]:
        """Evaluate K actions per state. Univariate actions may be ``[batch, k]``."""
        obs = _f32(states)
        act = jnp.asarray(actions, dtype=jnp.float32)
        if act.ndim == 2:
            act = act[:, :, None]
        return q_values(self.params, obs, act)

    def v_val(
        self,
        states: Float[Array, "batch obs"] | np.ndarray,
        exp_actions: int = 10,
        policy: Any = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch 1"]:
        """Average Q over actions sampled from ``policy`` (the target policy by default)."""
        behaviour = self.mc.get("policy", True) if policy is None else policy
        if key is None:
            key = self.mc.rng.split()
        values = estimate_v(self.params, behaviour.params, _f32(states), key, exp_actions)
        return values[:, None]


def _copy_leaf(value: object) -> object:
    if eqx.is_array(value):
        return jnp.array(value)
    return value


@eqx.filter_jit(donate="all-except-first")
def _critic_scan(
    data: tuple[Float[Array, "cap obs"], Float[Array, "cap act"], Float[Array, " cap"], PRNGKeyArray, Array],
    diff: Any,
    static: Any,
    opt_state: Any,
    tx: Any,
    num_steps: int,
    batch_size: int,
) -> tuple[Any, Any, Float[Array, " steps"], Float[Array, " steps"]]:
    """Scan Adam updates. Returns the new weights, optimiser state, and pre/post losses.

    Replay arrays are the first argument and are not donated. The weight leaves and the Adam
    state are donated into the values this function returns.
    """
    states, actions, targets, key, size = data

    def loss_on(weights: Any, idx: Array) -> Float[Array, ""]:
        network = eqx.combine(weights, static)
        prediction = q_forward(network, states[idx], actions[idx]).reshape(-1)
        return jnp.mean(jnp.square(prediction - targets[idx]))

    def body(
        carry: tuple[Any, Any, PRNGKeyArray], step: Array
    ) -> tuple[tuple[Any, Any, PRNGKeyArray], tuple[Array, Array]]:
        weights, state, key_ = carry
        key_, sample_key = jax.random.split(key_)
        idx = jax.random.randint(sample_key, (batch_size,), 0, size)
        pre, grads = eqx.filter_value_and_grad(lambda w: loss_on(w, idx))(weights)
        updates, state = tx.update(grads, state, weights)
        weights = eqx.apply_updates(weights, updates)
        post = jax.lax.cond(step % 10 == 0, lambda w: loss_on(w, idx), lambda _w: pre, weights)
        return (weights, state, key_), (pre, post)

    (diff, opt_state, _), (pre, post) = jax.lax.scan(body, (diff, opt_state, key), jnp.arange(num_steps))
    return diff, opt_state, pre, post


def _actions(actions: Float[Array, "..."] | np.ndarray, action_dim: int) -> Float[Array, "batch act"]:
    act = jnp.asarray(actions, dtype=jnp.float32)
    if act.ndim == 1:
        act = act[:, None]
    if act.shape[-1] != action_dim and action_dim == 1:
        act = act[:, :1]
    return act
