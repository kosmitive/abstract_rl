"""A Gaussian policy whose only trainable storage is one shared buffer.

The module is what TRPO and MPO already call: ``forward``, ``log_prob``,
``sample_actions``, ``kl_divergence``, flat-vector updates. The arithmetic is
``interop.forward``, so the JAX rollout evaluates the same formulas on a view
of ``theta``. Checkpoints use the names ``select_policy`` writes
(``network.layers.i.weight``, ``bias``, ``cov``), sliced out of the buffer.
Nothing is constructed at save time, so saving does not draw from the MLX RNG.

MLX optimizers return new arrays. ``update`` copies those bytes back into the
buffer and rebinds ``theta`` to the view, except while ``mx.compile`` is
tracing: evaluating the tracer is not allowed, and the compiled step installs
its result afterwards without calling ``update``. TRPO calls ``update`` again
on the host after the step, which refreshes the buffer. MPO's Adam step does
not, so the rollout copies once per rollout if the pointer changed. That copy
is one parameter vector.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.batch import Batch
from abstract_rl.interop.forward import (
    entropy_from_stats,
    log_prob_from_stats,
    mode_from_stats,
    normal_kl,
    sample_from_stats,
    suff_stats,
)
from abstract_rl.interop.namespace import MLX
from abstract_rl.interop.shared import SharedParams, buffer_address
from abstract_rl.interop.spec import GaussianSpec, make_gaussian_spec, pack_module, unpack_vector
from abstract_rl.misc.mixed import to_mx
from abstract_rl.policy.continuous.continuous_policy import ContinuousPolicy
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


class SharedGaussianPolicy(ContinuousPolicy):
    """Univariate or diagonal Gaussian policy stored in one buffer."""

    def __init__(self, mc: ModelConfiguration) -> None:
        """Build the policy configured in the current namespace.

        A ``select_policy`` module is created first, so the MLX RNG and the
        state-covariance initialization match the Gymnasium path. Its values
        are copied by name into the buffer and the module is dropped.

        Args:
            mc: The model configuration, with the environment already registered.
                Call this inside the ``policy`` namespace, as ``run_trpo`` does.
        """
        super().__init__(mc)
        template = select_policy(mc)
        self.spec = _spec_from_template(mc, template)
        flat, frozen = pack_module(template, self.spec)
        self._shared = SharedParams(self.spec.n_params)
        self._shared.write_numpy(flat)
        self.theta = self._shared.as_mlx()
        self.cov_type = self.spec.cov_type
        self.squash = self.spec.squash
        self.action_dim = self.spec.action_dim
        self.kind = self.spec.kind
        if self.cov_type == "fixed":
            self.fixed_log_sigma = mx.array(frozen.astype(np.float32))
            self.freeze(recurse=False, keys=["fixed_log_sigma"])
        del template

    @property
    def shared(self) -> SharedParams:
        """The buffer JAX reads."""
        return self._shared

    def export_for_rollout(self) -> SharedParams:
        """Make the buffer match the module, then return it.

        Returns:
            The shared parameters. If training left ``theta`` on a fresh array,
            its bytes are copied back first. That is one vector, not a second
            network.
        """
        try:
            copied = self._shared.write_from_mlx(self.theta)
        except RuntimeError as exc:
            raise RuntimeError("export_for_rollout cannot run inside mx.compile") from exc
        if copied or buffer_address(self.theta) != buffer_address(self._shared.as_mlx()):
            nn.Module.update(self, {"theta": self._shared.as_mlx()})
        return self._shared

    def fixed_numpy(self) -> np.ndarray:
        """The frozen ``log_sigma``, or an empty vector when the covariance is trained."""
        if self.cov_type != "fixed":
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.fixed_log_sigma, dtype=np.float32).reshape(-1)

    def update(self, parameters: dict[Any, Any], strict: bool = True) -> SharedGaussianPolicy:  # type: ignore[override]
        """Install parameters, then copy a fresh ``theta`` back into the buffer.

        ``nn.Module`` is also a mapping, so this override cannot match both
        signatures. During ``mx.compile`` the copy is skipped: the tracer cannot
        be evaluated, and MLX installs the concrete arrays after the call.
        """
        nn.Module.update(self, parameters, strict)
        try:
            copied = self._shared.write_from_mlx(self.theta)
        except RuntimeError:
            return self
        if copied:
            nn.Module.update(self, {"theta": self._shared.as_mlx()})
        return self

    def _fixed(self) -> mx.array | None:
        if self.cov_type != "fixed":
            return None
        return self.fixed_log_sigma

    def get_suff_stats_dim(self) -> int:
        """2 for a univariate policy and ``2A`` for a diagonal one."""
        if self.kind == "univariate":
            return 2
        return 2 * self.action_dim

    def forward(self, states: mx.array) -> mx.array:
        """Sufficient statistics ``[mu, sigma]``."""
        return suff_stats(MLX, self.spec, self.theta, states, self._fixed())

    def _stats(self, states: mx.array | None, suff: mx.array | None) -> mx.array:
        if (states is None) == (suff is None):
            raise ValueError("pass either states or suff_stats")
        if suff is None:
            assert states is not None
            return self.forward(states)
        return suff

    def log_prob(
        self, actions: mx.array, states: mx.array | None = None, suff_stats: mx.array | None = None
    ) -> mx.array:
        """Log probability. Shapes match the univariate and diagonal policies."""
        return log_prob_from_stats(MLX, self.spec, self._stats(states, suff_stats), actions)

    def mode(self, states: mx.array | None = None, suff_stats: mx.array | None = None) -> mx.array:
        """The most likely action."""
        return mode_from_stats(MLX, self.spec, self._stats(states, suff_stats))

    def sample_noise(self, num_states: int, num_actions: int = 1) -> mx.array:
        """Standard normal noise. Univariate ``[D, K]``, diagonal ``[D, K, A]``."""
        if self.kind == "univariate":
            return mx.random.normal((num_states, num_actions))
        return mx.random.normal((num_states, num_actions, self.action_dim))

    def sample_actions(
        self,
        states: mx.array | None = None,
        suff_stats: mx.array | None = None,
        num_actions: int = 1,
        noise: mx.array | None = None,
    ) -> mx.array:
        """Reparameterised samples, optionally with noise from ``sample_noise``."""
        stats = self._stats(states, suff_stats)
        drawn = noise if noise is not None else self.sample_noise(int(stats.shape[0]), num_actions)
        return sample_from_stats(MLX, self.spec, stats, drawn)

    def entropy(self, batch: Batch) -> mx.array:
        """Mean Gaussian entropy. Squashed policies have none in closed form."""
        return entropy_from_stats(MLX, self.spec, self.forward(to_mx(batch.states)))

    def kl_divergence(
        self, batch: Batch, suff_stats: mx.array | None = None, mode: str = "i-projection", est: str = "sum"
    ) -> mx.array:
        """Analytic KL against the target policy. The squash cancels, as in the other policies."""
        states = to_mx(batch.states)
        if suff_stats is None:
            suff_stats = self.forward(states)
        target = self.mc.get("policy", True)
        t_stats = mx.stop_gradient(target.forward(states))
        mu, sigma = _columns(self.spec, suff_stats)
        t_mu, t_sigma = _columns(self.spec, t_stats)
        match mode:
            case "i-projection":
                kl = normal_kl(MLX, mu, sigma, t_mu, t_sigma)
            case "m-projection":
                kl = normal_kl(MLX, t_mu, t_sigma, mu, sigma)
            case _:
                raise NotImplementedError(mode)
        if self.kind == "diagonal":
            kl = MLX.sum(kl, axis=1)
        match est:
            case "sum":
                return MLX.sum(kl)
            case "mean":
                return MLX.mean(kl)
            case _:
                raise NotImplementedError(est)

    def clone(self) -> SharedGaussianPolicy:
        """A new policy with its own buffer, holding the same values."""
        cloned = SharedGaussianPolicy(self.mc)
        self.sync(cloned)
        return cloned

    def sync(self, v: SharedGaussianPolicy) -> SharedGaussianPolicy:
        """Copy values into ``v``'s buffer. The two policies do not share storage."""
        v._shared.write_numpy(np.array(self._shared.numpy_view(), copy=True))
        nn.Module.update(v, {"theta": v._shared.as_mlx()})
        if self.cov_type == "fixed":
            v.fixed_log_sigma = mx.array(np.array(self.fixed_log_sigma, copy=True))
            v.freeze(recurse=False, keys=["fixed_log_sigma"])
        return v

    def save_weights(self, file: str) -> None:
        """Save under the names ``select_policy`` uses, read from the buffer."""
        self.export_for_rollout()
        frozen = self.fixed_numpy() if self.cov_type == "fixed" else None
        named = unpack_vector(self._shared.numpy_view(), self.spec, frozen)
        arrays = {name: mx.array(np.asarray(value, dtype=np.float32)) for name, value in named.items()}
        if file.endswith(".npz"):
            mx.savez(file, **arrays)
            return
        if file.endswith(".safetensors"):
            mx.save_safetensors(file, arrays)
            return
        raise ValueError(f"unsupported checkpoint extension for {file}")

    def load_weights(self, file_or_weights: str | list[tuple[str, mx.array]], strict: bool = True) -> nn.Module:
        """Load a ``select_policy`` checkpoint into this buffer."""
        if isinstance(file_or_weights, str):
            loaded = mx.load(file_or_weights)
            if not isinstance(loaded, dict):
                raise TypeError(f"checkpoint {file_or_weights} did not load as a dict of arrays")
            weights = {str(name): value for name, value in loaded.items()}
        else:
            weights = {str(name): value for name, value in file_or_weights}
        expected = {tensor.name for tensor in self.spec.named}
        if strict and set(weights) != expected:
            raise ValueError(f"checkpoint names {sorted(weights)} do not match {sorted(expected)}")
        flat = np.zeros(self.spec.n_params, dtype=np.float32)
        for tensor in self.spec.named:
            if tensor.name not in weights:
                continue
            value = np.asarray(weights[tensor.name], dtype=np.float32).reshape(tensor.shape)
            if tensor.offset is None:
                self.fixed_log_sigma = mx.array(value.reshape(-1))
                self.freeze(recurse=False, keys=["fixed_log_sigma"])
                continue
            flat[tensor.offset : tensor.offset + value.size] = value.reshape(-1)
        self._shared.write_numpy(flat)
        nn.Module.update(self, {"theta": self._shared.as_mlx()})
        return self


def _spec_from_template(
    mc: ModelConfiguration, template: UnivariateGaussianPolicy | DiagonalGaussianPolicy
) -> GaussianSpec:
    """The flat layout corresponding to a module ``select_policy`` just built."""
    conf = mc.get("conf")
    kind = "univariate" if isinstance(template, UnivariateGaussianPolicy) else "diagonal"
    action_dim = 1 if kind == "univariate" else int(template.action_dim)
    low = np.asarray(mc.env.action_space.low, dtype=np.float64)
    high = np.asarray(mc.env.action_space.high, dtype=np.float64)
    return make_gaussian_spec(
        obs_dim=int(mc.env.observation_dim),
        action_dim=action_dim,
        kind=kind,
        cov_type=str(template.cov_type),
        structure=[int(v) for v in conf["structure"]],
        act_fn=list(conf["act_fn"]),
        layer_norm=bool(conf["layer_norm"]),
        squash=bool(template.squash),
        action_low=low,
        action_high=high,
    )


def _columns(spec: GaussianSpec, stats: mx.array) -> tuple[mx.array, mx.array]:
    """``mu`` and ``sigma`` with the same rank the existing KL uses."""
    width = 1 if spec.kind == "univariate" else spec.action_dim
    mu, sigma = stats[:, :width], stats[:, width:]
    if spec.kind == "univariate":
        return mu[:, 0], sigma[:, 0]
    return mu, sigma

