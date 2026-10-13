"""Shared behaviour of the squashed Gaussian policies."""

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.abstract_conf.syncable import Syncable
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.misc.flat_params import copy_module
from abstract_rl.neural_modules.neural_network import MultilayerPerceptron
from abstract_rl.policy.continuous.gaussian_math import (
    GaussianParams,
    entropy_from_sigma,
    forward,
    head_activations,
    kl_divergence,
    likelihood_ratio,
    log_prob,
    mode,
    sample,
    sigma_is_softplus,
)
from abstract_rl.policy.continuous.numpy_act import NumpyPolicy, act_numpy, extract_numpy_policy, seed_generator
from abstract_rl.policy.policy import Policy

_LIKELIHOOD = eqx.filter_jit(likelihood_ratio)
_KL = eqx.filter_jit(kl_divergence)


def build_gaussian_params(mc: ModelConfiguration, key: PRNGKeyArray, univariate: bool) -> GaussianParams:
    """Build the Gaussian configured in the current namespace of ``mc``.

    Args:
        mc: Model configuration with the environment registered. The active namespace is the policy.
        key: Initialisation key.

    Returns:
        Equinox parameters of the policy.
    """
    run_conf = mc.get("conf")
    cov_type = str(run_conf["cov_type"])
    action_dim = mc.env.action_dim
    squash = bool(run_conf.get("squash", False))
    low = np.asarray(mc.env.action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(mc.env.action_space.high, dtype=np.float64).reshape(-1)
    if squash and not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("squashing needs a bounded action space")
    center: tuple[float, ...]
    scale: tuple[float, ...]
    if squash and univariate:
        # The univariate policy squashes with ``max |high| * tanh(u)`` and no centre shift.
        center = (0.0,)
        scale = (float(np.max(np.abs(high))),)
    elif squash:
        center = tuple(float(v) for v in (high + low) / 2)
        scale = tuple(float(v) for v in (high - low) / 2)
    else:
        center = tuple(0.0 for _ in range(action_dim))
        scale = tuple(1.0 for _ in range(action_dim))

    match cov_type:
        case "state":
            n_out = 2 * action_dim
            log_sigma = None
            fixed_log_sigma = None
        case "fixed":
            n_out = action_dim
            log_sigma = None
            fixed_log_sigma = tuple(float(np.log(run_conf["sigma"])) for _ in range(action_dim))
        case "single":
            n_out = action_dim
            init = float(np.log(run_conf.get("sigma", 1.0)))
            log_sigma = jnp.full((action_dim,), init, dtype=jnp.float32)
            fixed_log_sigma = None
        case _:
            raise NotImplementedError(f"unknown covariance type {cov_type}")

    act_fn, sigma_acts = head_activations(run_conf["act_fn"], action_dim)
    structure: Sequence[int] = [mc.env.observation_dim, *list(run_conf["structure"]), n_out]
    network = MultilayerPerceptron(
        key,
        structure,
        act_fn,
        bool(run_conf["layer_norm"]),
        str(run_conf["init"]),
    )
    return GaussianParams(
        network=network,
        log_sigma=log_sigma,
        cov_type=cov_type,
        squash=squash,
        sigma_is_softplus=sigma_is_softplus(sigma_acts),
        action_center=center,
        action_scale=scale,
        fixed_log_sigma=fixed_log_sigma,
    )


class ContinuousPolicy(Policy, Syncable):
    """Gaussian policy wrapper. ``params`` is the Equinox module the optimisers update."""

    params: GaussianParams
    flat_actions: bool

    def __init__(self, mc: ModelConfiguration, flat_actions: bool) -> None:
        super().__init__(mc)
        self.flat_actions = flat_actions
        self.params = build_gaussian_params(mc, mc.rng.split(), univariate=flat_actions)
        self._act_cache_id = -1
        self._act_cache: NumpyPolicy | None = None
        self._act_rng: np.random.Generator | None = None

    @property
    def network(self) -> MultilayerPerceptron:
        return self.params.network

    @network.setter
    def network(self, value: MultilayerPerceptron) -> None:
        self.params = eqx.tree_at(lambda params: params.network, self.params, value)

    @property
    def squash(self) -> bool:
        return self.params.squash

    @property
    def cov_type(self) -> str:
        return self.params.cov_type

    def get_suff_stats_dim(self) -> int:
        return 2 * self.mc.env.action_dim

    def forward(self, states: Float[Array, "batch obs"] | np.ndarray) -> Float[Array, "batch stats"]:
        """Sufficient statistics ``[mu, sigma]``."""
        return forward(self.params, _f32(states))

    def log_prob(
        self,
        actions: Float[Array, "..."] | np.ndarray,
        states: Float[Array, "batch obs"] | np.ndarray | None = None,
        suff_stats: Float[Array, "batch stats"] | np.ndarray | None = None,
    ) -> Float[Array, "batch k"]:
        """Log probability of actions.

        Univariate policies take actions of shape ``[batch, k]``. Diagonal policies take
        ``[batch, k, act]`` or ``[batch, act]``.
        """
        return log_prob(
            self.params,
            _f32(actions),
            None if states is None else _f32(states),
            None if suff_stats is None else _f32(suff_stats),
        )

    def mode(
        self,
        states: Float[Array, "batch obs"] | np.ndarray | None = None,
        suff_stats: Float[Array, "batch stats"] | np.ndarray | None = None,
    ) -> Float[Array, "batch act"]:
        """Most likely action, shape ``[batch, act]``."""
        acted = mode(
            self.params,
            None if states is None else _f32(states),
            None if suff_stats is None else _f32(suff_stats),
        )
        if self.flat_actions:
            return acted[:, :1]
        return acted

    def sample_actions(
        self,
        states: Float[Array, "batch obs"] | np.ndarray | None = None,
        suff_stats: Float[Array, "batch stats"] | np.ndarray | None = None,
        num_actions: int = 1,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "..."]:
        """Sample ``num_actions`` actions per state.

        Returns ``[batch, k]`` for the univariate policy and ``[batch, k, act]`` otherwise.
        """
        if key is None:
            key = self.mc.rng.split()
        drawn = sample(
            self.params,
            key,
            num_actions,
            None if states is None else _f32(states),
            None if suff_stats is None else _f32(suff_stats),
        )
        if self.flat_actions:
            return drawn[..., 0]
        return drawn

    def act(self, states: np.ndarray, deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
        """Action selection for environment interaction.

        The density matches ``log_prob``. Weights are copied to the host once per parameter
        object, then every step is a NumPy matmul. Stochastic steps draw noise from a generator
        seeded from the run key instead of splitting a JAX key per step.

        Args:
            states: Batch of observations, shape ``[batch, obs]``.
            deterministic: Use the mode instead of a sample.

        Returns:
            Actions ``[batch, act]`` and log probabilities ``[batch]``, both NumPy arrays.
        """
        obs = np.asarray(states, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        spec = self._numpy_policy()
        if deterministic:
            actions, log_density = act_numpy(spec, obs, None)
        else:
            noise = self._action_generator().standard_normal((obs.shape[0], spec.center.shape[0]), dtype=np.float32)
            actions, log_density = act_numpy(spec, obs, noise)
        return actions, log_density

    def _numpy_policy(self) -> NumpyPolicy:
        cache_id = id(self.params)
        if self._act_cache is not None and self._act_cache_id == cache_id:
            return self._act_cache
        spec = extract_numpy_policy(self.params)
        self._act_cache_id = cache_id
        self._act_cache = spec
        return spec

    def _action_generator(self) -> np.random.Generator:
        if self._act_rng is None:
            self._act_rng = seed_generator(self.mc.rng.split())
        return self._act_rng

    def clone(self) -> "ContinuousPolicy":
        other = self.__class__.__new__(self.__class__)
        other.mc = self.mc
        other.flat_actions = self.flat_actions
        other.params = copy_module(self.params)
        other._act_cache_id = -1
        other._act_cache = None
        other._act_rng = None
        return other

    def sync(self, v: Any) -> Any:
        v.params = self.params
        return v

    def save(self, path: str) -> None:
        """Write the parameter tree to ``path``."""
        eqx.tree_serialise_leaves(path, self.params)

    def load(self, path: str) -> None:
        """Load a parameter tree previously written by ``save``."""
        self.params = eqx.tree_deserialise_leaves(path, self.params)

    def kl_divergence(
        self,
        batch: Batch,
        suff_stats: Float[Array, "batch stats"] | np.ndarray | None = None,
        mode: str = "i-projection",
        est: str = "sum",
    ) -> Float[Array, ""]:
        """Analytic KL against the target policy registered in the model configuration."""
        target = self.mc.get("policy", True).params
        stats = None if suff_stats is None else _f32(suff_stats)
        return _KL(self.params, target, _f32(batch.states), stats, mode, est)

    def likelihood_ratio(self, batch: Batch) -> Float[Array, ""]:
        """Negative standardised likelihood-ratio surrogate of ``batch``."""
        return _LIKELIHOOD(
            self.params,
            _f32(batch.states),
            _f32(batch.actions),
            _f32(batch.log_likelihoods),
            _f32(batch["a"]),
        )

    def entropy(self, batch: Batch) -> Float[Array, ""]:
        """Closed-form Gaussian entropy. Squashed policies have none."""
        if self.squash:
            raise NotImplementedError("entropy of a squashed gaussian policy is not available in closed form")
        stats = forward(self.params, _f32(batch.states))
        act = stats.shape[-1] // 2
        return entropy_from_sigma(stats[:, act:])


def _f32(value: Float[Array, "..."] | np.ndarray) -> Float[Array, "..."]:
    return jnp.asarray(value, dtype=jnp.float32)
