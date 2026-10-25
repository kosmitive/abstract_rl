"""Evaluate a small gaussian policy in the rollout with NumPy instead of one MLX call per environment step.

Every MLX evaluation on the CPU costs about 40 us of scheduling, several times the work of a 64 x 64 policy network.
The host actor reproduces the compiled ``ContinuousPolicy.act`` of the CPU stream bit by bit:

* the noise uses the same keys as ``mx.random.normal``: the key chain of ``mx.random.split`` and the counters of the
  random bits are the threefry hash of MLX, the conversion of the bits into normal samples is the compiled MLX
  function, evaluated once for a block of steps,
* the matrix products are the same BLAS calls (one row at a time), the tanh of the hidden layers is the vForce
  ``vvtanhf`` of MLX and the tanh of the squashing is the scalar ``tanhf`` MLX uses in compiled kernels for fewer
  than 8 elements,
* the log likelihoods are computed with the compiled MLX log density for the whole episode at its end.

At the end of the episode the global MLX key is set to the one after the last step, as if every step had called
``act``. Only configurations whose bit identity is covered by the tests are accepted, see ``host_actor``.
"""

import ctypes
import sys
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from abstract_rl.misc.mixed import Identity

_MASK = 0xFFFFFFFF
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
# number of steps whose noise is drawn at once
BLOCK = 256
# (center, scale, u_max) of a tanh squashed policy
Squash = tuple[np.ndarray, np.ndarray, float]


def _threefry(k0: int, k1: int, c0: int, c1: int) -> tuple[int, int]:
    """threefry2x32 hash of MLX (mlx/backend/cpu/threefry.cpp) on Python integers."""
    k2 = k0 ^ k1 ^ 0x1BD11BDA
    ks = (k0, k1, k2)
    c0 = (c0 + k0) & _MASK
    c1 = (c1 + k1) & _MASK
    for i in range(5):
        for r in _ROTATIONS[i % 2]:
            c0 = (c0 + c1) & _MASK
            c1 = (((c1 << r) & _MASK) | (c1 >> (32 - r))) ^ c0
        c0 = (c0 + ks[(i + 1) % 3]) & _MASK
        c1 = (c1 + ks[(i + 2) % 3] + i + 1) & _MASK
    return c0, c1


def _threefry_np(k0: np.ndarray, k1: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """threefry2x32 hash of MLX for arrays of keys and counters (uint32, wrapping arithmetic)."""
    ks = (k0, k1, k0 ^ k1 ^ np.uint32(0x1BD11BDA))
    c0 = c0 + k0
    c1 = c1 + k1
    for i in range(5):
        for r in _ROTATIONS[i % 2]:
            c0 = c0 + c1
            c1 = ((c1 << np.uint32(r)) | (c1 >> np.uint32(32 - r))) ^ c0
        c0 = c0 + ks[(i + 1) % 3]
        c1 = c1 + ks[(i + 2) % 3] + np.uint32(i + 1)
    return c0, c1


def split_chain(key: tuple[int, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """The next n keys of ``mx.random.split`` (as in ``KeySequence::next``) starting from key.
    :return: The sub keys used for sampling [n, 2] and the state after each step [n, 2].
    """
    subs = np.empty((n, 2), np.uint32)
    states = np.empty((n, 2), np.uint32)
    k0, k1 = key
    for i in range(n):
        # bits of shape [2, 2]: counters (0, 2) and (1, 3) give the new state (row 0) and the sub key (row 1)
        a0, a2 = _threefry(k0, k1, 0, 2)
        a1, a3 = _threefry(k0, k1, 1, 3)
        k0, k1 = a0, a1
        states[i] = (a0, a1)
        subs[i] = (a2, a3)
    return subs, states


def random_bits(keys: np.ndarray, d: int) -> np.ndarray:
    """d random uint32 per key, as ``mx.random.bits`` (mlx/backend/cpu/primitives.cpp RandomBits) for every row."""
    n = keys.shape[0]
    k0, k1 = keys[:, 0], keys[:, 1]
    out = np.empty((n, d), np.uint32)
    half = d // 2
    off = half + d % 2
    for j in range(half):
        out[:, j], out[:, off + j] = _threefry_np(k0, k1, np.full(n, j, np.uint32), np.full(n, off + j, np.uint32))
    if d % 2:
        out[:, half] = _threefry_np(k0, k1, np.full(n, half, np.uint32), np.zeros(n, np.uint32))[0]
    return out


def _bits_to_normal(bits: mx.array) -> mx.array:
    # mlx/random.cpp: uniform in [nextafter(-1, 0), 1) from the bits, then sqrt(2) * erfinv
    low = mx.array(float(np.nextafter(np.float32(-1.0), np.float32(0.0))), mx.float32)
    high = mx.array(1.0, mx.float32)
    u = mx.divide(bits, mx.array(float(np.iinfo(np.uint32).max), mx.float32))
    u = mx.minimum(u, mx.array(float(np.nextafter(np.float32(1.0), np.float32(0.0))), mx.float32))
    return mx.array(float(np.sqrt(2.0)), mx.float32) * mx.erfinv((high - low) * u + low)


_bits_to_normal_compiled = mx.compile(_bits_to_normal)


def normal_from_bits(bits: np.ndarray) -> np.ndarray:
    """Standard normal samples of ``mx.random.normal`` (in a compiled function) from their random bits."""
    with mx.stream(mx.cpu):
        out: np.ndarray = np.array(_bits_to_normal_compiled(mx.array(bits)))
    return out


class _Accelerate:
    """vForce tanh (used by MLX for the tanh of contiguous arrays) and the scalar tanhf of the system library."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        self.lib.vvtanhf.restype = None
        self.lib.vvtanhf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.libm = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.libm.tanhf.restype = ctypes.c_float
        self.libm.tanhf.argtypes = [ctypes.c_float]

    def vtanh(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(x)
        n = ctypes.c_int(x.size)
        self.lib.vvtanhf(out.ctypes.data, x.ctypes.data, ctypes.byref(n))
        return out

    def scalar_tanh(self, x: np.ndarray) -> np.ndarray:
        tanhf = self.libm.tanhf
        return np.array([tanhf(v) for v in x.tolist()], dtype=np.float32)


_ACCELERATE: _Accelerate | None = None


def _accelerate() -> _Accelerate:
    global _ACCELERATE
    if _ACCELERATE is None:
        _ACCELERATE = _Accelerate()
    return _ACCELERATE


class HostActor:
    """Samples the actions of one episode of a gaussian policy on the host, see the module documentation."""

    def __init__(
        self,
        weights: list[tuple[np.ndarray, np.ndarray]],
        hidden_tanh: list[bool],
        sigma: np.ndarray,
        squash: Squash | None,
        sample_shape: tuple[int, ...],
        log_prob: Any,
        key: tuple[int, int] | None = None,
        noise: np.ndarray | None = None,
        cursor: list[int] | None = None,
    ) -> None:
        """
        :param weights: The (weight [out, in], bias [out]) of every linear layer.
        :param hidden_tanh: For every hidden layer, whether a tanh follows it (identity otherwise).
        :param sigma: The standard deviation of every action dimension.
        :param squash: (center, scale, u_max) of the squashing center + scale * tanh(clip(u)), None for no
            squashing.
        :param sample_shape: The shape of the sample of ``act`` for one state, e.g. [1, 1, A].
        :param log_prob: Compiled MLX function (actions [T, ...], suff_stats [T, S]) -> log likelihoods [T, 1].
        """
        self._acc = _accelerate()
        # w.T is a view, the products are the same BLAS calls as the x @ w.T of MLX
        self._weights = weights
        self._hidden_tanh = hidden_tanh
        self._sigma = sigma
        self._squash = squash
        self._sample_shape = sample_shape
        self._log_prob = log_prob
        self._dim = int(np.prod(sample_shape))
        self._states = np.empty((0, 2), np.uint32)
        self._actions: list[np.ndarray] = []
        self._stats: list[np.ndarray] = []
        # A preset buffer is the noise of a frozen snapshot. It does not read or write the global key: the M-step
        # owns that stream while this actor runs on another thread.
        self._preset = noise is not None
        self._cursor = cursor
        if noise is not None:
            if cursor is None:
                raise ValueError("preset noise needs a shared cursor")
            self._noise = noise
            self._pos = cursor[0]
            self._commit_key = False
            self._key = key if key is not None else (0, 0)
        else:
            self._commit_key = key is None
            if key is None:
                k0, k1 = (int(k) for k in np.array(cast(Any, mx.random.state)[0]))
                self._key = (k0, k1)
            else:
                self._key = key
            self._noise = np.empty((0, self._dim), np.float32)
            self._pos = 0

    def _draw(self) -> None:
        subs, self._states = split_chain(self._key, BLOCK)
        self._key = (int(self._states[-1, 0]), int(self._states[-1, 1]))
        self._noise = normal_from_bits(random_bits(subs, self._dim))
        self._pos = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """The sampled action [A] (float32) for the state [1, Ds] (float32)."""
        if self._pos == self._noise.shape[0]:
            if self._preset:
                raise RuntimeError("frozen policy snapshot exhausted its action noise")
            self._draw()
        h = state
        last = len(self._weights) - 1
        for i, (w, b) in enumerate(self._weights):
            h = h @ w.T + b
            if i < last and self._hidden_tanh[i]:
                h = self._acc.vtanh(h)
        mu = h.reshape(-1)
        u = mu + self._sigma * self._noise[self._pos]
        self._pos += 1
        if self._cursor is not None:
            self._cursor[0] = self._pos
        if self._squash is not None:
            center, scale, u_max = self._squash
            u = center + scale * self._acc.scalar_tanh(np.clip(u, np.float32(-u_max), np.float32(u_max)))
        self._actions.append(u)
        self._stats.append(mu)
        return u

    def finish(self) -> np.ndarray:
        """Set the MLX key to the one after the last step and return the log likelihoods of the episode [T]."""
        if self._commit_key and self._pos > 0:
            k0, k1 = (int(k) for k in self._states[self._pos - 1])
            mx.random.seed((k0 << 32) | k1)
        if not self._actions:
            return np.zeros(0, np.float32)
        mu = np.stack(self._stats)
        stats = np.concatenate([mu, np.broadcast_to(self._sigma, mu.shape)], axis=1)
        actions = np.stack(self._actions).reshape((len(self._actions), *self._sample_shape[1:]))
        with mx.stream(mx.cpu):
            return np.array(self._log_prob(mx.array(actions), mx.array(stats))).reshape(-1)


class FrozenHost:
    """NumPy copy of a supported gaussian policy and the noise of one rollout, shared by every episode.

    The weights and the noise are taken before the M-step. Later updates of the live policy do not change them, so a
    rollout thread can sample while the learner writes the live parameters.
    """

    def __init__(
        self,
        weights: list[tuple[np.ndarray, np.ndarray]],
        hidden_tanh: list[bool],
        sigma: np.ndarray,
        squash: Squash | None,
        sample_shape: tuple[int, ...],
        log_prob: Any,
        noise: np.ndarray,
        key: tuple[int, int],
    ) -> None:
        self.weights = weights
        self._hidden_tanh = hidden_tanh
        self._sigma = sigma
        self._squash = squash
        self._sample_shape = sample_shape
        self._log_prob = log_prob
        self._noise = noise
        self.key = key
        self._cursor = [0]
        self._actors: list[HostActor] = []

    def spawn(self) -> HostActor:
        """An actor for the next episode. Episodes share the noise cursor and do not touch the global MLX key."""
        actor = HostActor(
            self.weights,
            self._hidden_tanh,
            self._sigma,
            self._squash,
            self._sample_shape,
            self._log_prob,
            key=self.key,
            noise=self._noise,
            cursor=self._cursor,
        )
        self._actors.append(actor)
        return actor

    def materialize(self, trajectories: list[Any]) -> None:
        """Write the log likelihoods of every episode. Call this on the main thread, after the rollout has joined.

        The density uses the stored means and the snapshotted standard deviation, not the live network, so it stays
        the likelihood of the frozen policy after the M-step has replaced the parameters.
        """
        if len(trajectories) != len(self._actors):
            raise RuntimeError(f"frozen rollout has {len(self._actors)} episodes but {len(trajectories)} trajectories")
        for traj, actor in zip(trajectories, self._actors, strict=True):
            lls = np.asarray(actor.finish(), dtype=traj.log_likelihoods.dtype).reshape(traj.log_likelihoods.shape)
            traj.log_likelihoods[:] = lls


def _host_parts(
    policy: Any, sample_shape: tuple[int, ...]
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[bool], np.ndarray, Any] | None:
    """Numpy weights, hidden tanh flags, sigma and the compiled log density, or None if the host actor cannot run."""
    net = policy.network
    acts = net.act_fn
    num_layers = len(net.layers)
    if sys.platform != "darwin" or mx.default_device() != mx.cpu or net.layer_norm:
        return None
    # a product with a single output column is not the same BLAS call in NumPy, so at least two action dimensions
    if policy.cov_type not in ("single", "fixed") or not 2 <= int(np.prod(sample_shape)) < 8:
        return None
    if len(acts) != num_layers or any(isinstance(a, list) for a in acts) or not isinstance(acts[-1], Identity):
        return None
    if not all(isinstance(a, nn.Tanh | Identity) for a in acts[:-1]):
        return None
    with mx.stream(mx.cpu):
        weights = [(np.array(layer.weight), np.array(layer.bias)) for layer in net.layers]
        sigma = np.array(mx.exp(policy.cov)).reshape(-1)
    fns: dict[str, Any] = policy.__dict__.setdefault("_host_fns", {})
    if "log_prob" not in fns:

        def log_prob(actions: mx.array, suff_stats: mx.array) -> mx.array:
            return policy.log_prob(actions, suff_stats=suff_stats)

        fns["log_prob"] = mx.compile(log_prob, inputs=[policy.state])
    hidden_tanh = [isinstance(a, nn.Tanh) for a in acts[:-1]]
    return weights, hidden_tanh, sigma, fns["log_prob"]


def host_actor(policy: Any, sample_shape: tuple[int, ...], squash: Squash | None) -> HostActor | None:
    """Host actor of a gaussian policy with a state independent standard deviation, None if it is not supported.

    Supported (and tested to be bit identical): macOS, the default stream on the CPU, a network without layer norm
    whose hidden activations are tanh or none and whose output has no activation, cov_type single or fixed, 2 to 7
    action dimensions.
    """
    parts = _host_parts(policy, sample_shape)
    if parts is None:
        return None
    weights, hidden_tanh, sigma, log_prob = parts
    return HostActor(weights, hidden_tanh, sigma, squash, sample_shape, log_prob)


def frozen_host(
    policy: Any, sample_shape: tuple[int, ...], squash: Squash | None, key: tuple[int, int], num_actions: int
) -> FrozenHost | None:
    """A frozen host actor with ``num_actions`` of noise drawn from ``key``, or None if the host actor is unsupported.

    ``key`` is a private subkey. The global MLX key is not read or written.
    """
    parts = _host_parts(policy, sample_shape)
    if parts is None:
        return None
    weights, hidden_tanh, sigma, log_prob = parts
    dim = int(np.prod(sample_shape))
    subs, _states = split_chain(key, num_actions)
    noise = normal_from_bits(random_bits(subs, dim))
    return FrozenHost(weights, hidden_tanh, sigma, squash, sample_shape, log_prob, noise, key)
