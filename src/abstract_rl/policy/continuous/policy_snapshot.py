"""Frozen behavioral policy for one pipelined MPO rollout.

The learner keeps the global MLX key and the live parameters. The rollout gets the other half of a split and a
snapshot taken before the M-step, so the two do not race. On a configuration the NumPy host actor supports, the
snapshot is a host copy and the rollout runs on a thread (MuJoCo and the host actor do not call MLX). Otherwise it
is a cloned module evaluated on the calling thread when the rollout is joined.
"""

import threading
from typing import Any, cast

import mlx.core as mx
import numpy as np

from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.policy.continuous.host_actor import FrozenHost


def read_key() -> tuple[int, int]:
    """The current global MLX key as ``(k0, k1)``."""
    k0, k1 = (int(k) for k in np.array(cast(Any, mx.random.state)[0]))
    return k0, k1


def write_key(key: tuple[int, int]) -> None:
    """Set the global MLX key. ``mx.random.seed`` of the packed key matches ``HostActor.finish``."""
    k0, k1 = key
    mx.random.seed((k0 << 32) | k1)


def fork_random_key() -> tuple[int, int]:
    """Split the global MLX key. The global key becomes the learner half; the subkey is returned for the rollout."""
    new, sub = mx.random.split(cast(Any, mx.random.state)[0])
    mx.eval(new, sub)
    new_np, sub_np = np.array(new), np.array(sub)
    write_key((int(new_np[0]), int(new_np[1])))
    return int(sub_np[0]), int(sub_np[1])


def _clone_keeping_rng(policy: Any) -> Any:
    """``policy.clone()`` without the random draws of the fresh module's initializer.

    Construction reads ``policy_*`` keys, so it runs in that namespace. The caller is the epoch loop, whose
    namespace stack is empty.
    """
    key = read_key()
    try:
        with policy.conf.ns("policy"):
            return policy.clone()
    finally:
        write_key(key)


class BehaviorSnapshot:
    """A policy and a private random key frozen for the rollout of one epoch."""

    def __init__(self, host: FrozenHost | None, clone: Any, rollout_key: tuple[int, int]) -> None:
        if (host is None) == (clone is None):
            raise RuntimeError("a behavior snapshot needs exactly one of a host actor or a cloned policy")
        self.host = host
        self.clone = clone
        self.rollout_key = rollout_key

    def overlaps_learner(self) -> bool:
        """Whether the rollout can run beside the M-step. Only the NumPy snapshot never touches MLX."""
        return self.host is not None


def behavior_snapshot(policy: Any, num_actions: int | None) -> BehaviorSnapshot:
    """Fork the MLX key and freeze ``policy`` for one rollout of at most ``num_actions`` steps.

    ``num_actions`` is None when the episode length is unbounded: the host actor cannot predraw its noise, so the
    snapshot is a cloned module. Call this on the rollout stream. The global key afterwards is the learner half
    of the split, including when building the clone consumes random numbers.

    Args:
        policy: The live policy, before the M-step.
        num_actions: Upper bound on the env steps of the rollout, or None.

    Returns:
        The frozen behavioral policy.
    """
    rollout_key = fork_random_key()
    host = None
    if num_actions is not None and hasattr(policy, "frozen_host"):
        host = policy.frozen_host(rollout_key, num_actions)
    clone = None if host is not None else _clone_keeping_rng(policy)
    return BehaviorSnapshot(host, clone, rollout_key)


class PendingRollout:
    """The next epoch's trajectories, collected from a behavior snapshot.

    A host snapshot starts immediately on a daemon thread. A cloned module is rolled out on ``join``, on the calling
    thread, with the snapshot's key installed for the duration.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        snapshot: BehaviorSnapshot,
        max_steps: int | None,
        num_steps: int,
        render: bool,
        rew_field_name: str,
    ) -> None:
        self._env = env
        self._policy = policy
        self._snapshot = snapshot
        self._max_steps = max_steps
        self._num_steps = num_steps
        self._render = render
        self._rew = rew_field_name
        self._tc: TrajectoryCollection | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if snapshot.host is not None:
            self._thread = threading.Thread(target=self._run, name="stale-policy-rollout", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self._tc = self._env.execute_policy(
                self._policy,
                self._max_steps,
                self._num_steps,
                exploration=True,
                render=self._render,
                rew_field_name=self._rew,
                frozen=self._snapshot.host,
                quiet=True,
            )
        except BaseException as exc:
            self._error = exc

    def join(self) -> TrajectoryCollection:
        """The collected trajectories. Log likelihoods of a host snapshot are filled here."""
        if self._thread is not None:
            self._thread.join()
            if self._error is not None:
                raise self._error
            if self._tc is None or self._snapshot.host is None:
                raise RuntimeError("the stale rollout finished without trajectories")
            self._snapshot.host.materialize(self._tc.trajectories())
            self._tc.print()
            return self._tc
        saved = read_key()
        write_key(self._snapshot.rollout_key)
        try:
            clone = self._snapshot.clone
            assert clone is not None
            return self._env.execute_policy(
                clone,
                self._max_steps,
                self._num_steps,
                exploration=True,
                render=self._render,
                rew_field_name=self._rew,
            )
        finally:
            write_key(saved)

    def abandon(self) -> None:
        """Join a host rollout and drop its result. Used when the epoch loop exits early."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
