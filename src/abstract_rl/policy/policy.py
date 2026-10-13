"""Policy interface."""

from jaxtyping import Array, Float

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration


class Policy:
    """A policy mapping states to a distribution over actions."""

    def __init__(self, mc: ModelConfiguration) -> None:
        self.mc = mc

    def get_suff_stats_dim(self) -> int:
        """Width of the sufficient-statistics vector."""
        raise NotImplementedError

    def forward(self, states: Float[Array, "batch obs"]) -> Float[Array, "batch stats"]:
        """Sufficient statistics for each state.

        Args:
            states: States, shape ``[batch, obs]``.

        Returns:
            Sufficient statistics.
        """
        raise NotImplementedError
