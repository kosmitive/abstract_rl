"""One policy buffer shared by a JAX rollout and MLX training.

Importing this package does not import JAX. The rollout backend is opt-in
(``env_rollout_backend: jax``); the Gymnasium path is unchanged.
"""
