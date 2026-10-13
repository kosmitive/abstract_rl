"""Explicit JAX random keys shared by one training run."""

import jax
from jaxtyping import Array


class RNG:
    """Splits a single key so network initialisation and sampling stay deterministic."""

    def __init__(self, seed: int) -> None:
        self.key: Array = jax.random.PRNGKey(seed)

    def split(self) -> Array:
        """Return a fresh key and advance the internal one.

        Returns:
            A PRNG key that is not reused.
        """
        self.key, sub = jax.random.split(self.key)
        return sub
