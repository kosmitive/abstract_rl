from typing import Any

import equinox as eqx


class Trajectory(eqx.Module):
    """
    One lane of a rollout: L consecutive steps holding one or more whole episodes. ``ends`` marks the last step of every
    episode (terminated, truncated by the environment or cut at max_steps); ``dones`` marks true terminations only;
    ``mask`` marks the valid steps (the lane is padded after its last episode). A trajectory collection holds the lanes
    of one rollout with a leading lane axis.
    """

    _states: Any
    _next_states: Any
    _actions: Any
    _rewards: Any
    _raw_rewards: Any
    _dones: Any
    _log_likelihoods: Any
    _ends: Any
    _mask: Any
    _annotations: dict[str, Any]

    def __init__(
        self,
        states: Any,
        next_states: Any,
        actions: Any,
        rewards: Any,
        raw_rewards: Any,
        dones: Any,
        log_likelihoods: Any,
        ends: Any,
        mask: Any,
        annotations: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new trajectory from the (padded) step arrays of one lane."""
        self._states = states
        self._next_states = next_states
        self._actions = actions
        self._rewards = rewards
        self._raw_rewards = raw_rewards
        self._dones = dones
        self._log_likelihoods = log_likelihoods
        self._ends = ends
        self._mask = mask
        self._annotations = dict(annotations) if annotations is not None else {}

    def __len__(self) -> int:
        return int(self._mask.shape[-1])

    def set(self, key: str, item: Any) -> "Trajectory":
        """A new trajectory with the annotation ``key`` set (trajectories are immutable)."""
        return eqx.tree_at(lambda t: t._annotations, self, {**self._annotations, key: item})

    def replace(self, **fields: Any) -> "Trajectory":
        """A new trajectory with some step arrays replaced, e.g. ``replace(states=...)``."""
        names = tuple(f"_{k}" for k in fields)
        return eqx.tree_at(lambda t: tuple(getattr(t, n) for n in names), self, tuple(fields.values()))

    def __getitem__(self, key: str) -> Any:
        return self._annotations[key]

    def __contains__(self, value: str) -> bool:
        return value in self._annotations

    # --- standard reinforcement learning fields

    @property
    def states(self) -> Any:
        """
        States of trajectory.
        :return: All states of the trajectory.
        """
        return self._states

    @property
    def actions(self) -> Any:
        """
        Actions of trajectory.
        :return: All actions of the trajectory.
        """
        return self._actions

    @property
    def dones(self) -> Any:
        """
        True terminations of the trajectory (truncations are not done).
        :return: The done flags.
        """
        return self._dones

    @property
    def log_likelihoods(self) -> Any:
        """
        Log likelihoods of trajectory.
        :return: All log likelihoods of the trajectory.
        """
        return self._log_likelihoods

    @property
    def rewards(self) -> Any:
        """
        Rewards of trajectory.
        :return: All rewards of the trajectory.
        """
        return self._rewards

    @property
    def raw_rewards(self) -> Any:
        """Rewards of the environment, before the training transform (for the logged return)."""
        return self._raw_rewards

    @property
    def next_states(self) -> Any:
        """
        Next states of trajectory.
        :return: All next states of the trajectory.
        """
        return self._next_states

    @property
    def ends(self) -> Any:
        """Last step of every episode."""
        return self._ends

    @property
    def mask(self) -> Any:
        """Valid steps."""
        return self._mask
