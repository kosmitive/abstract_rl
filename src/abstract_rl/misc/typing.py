"""Shared jaxtyping aliases and the float dtype of the JAX port.

Dimension names: D states in a batch, Ds state dim, A action dim, K actions per state, S suff stats, T time, E episodes.
"""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

DTYPE = jnp.float32

Key = PRNGKeyArray
Scalar = Float[Array, ""]
States = Float[Array, "D Ds"]
SuffStats = Float[Array, "D S"]
Actions = Float[Array, "D A"]
MultiActions = Float[Array, "D K A"]
UniActions = Float[Array, "D K"]
LogProbs = Float[Array, "D K"]
Mask = Bool[Array, "D"]
Step = Int[Array, ""]

__all__ = ["DTYPE", "Actions", "Array", "Bool", "Float", "Int", "Key", "LogProbs", "Mask", "MultiActions",
           "Scalar", "States", "Step", "SuffStats", "UniActions"]
