"""Closed-form checks for the pieces that were ported from the PyTorch implementation."""

import jax.numpy as jnp
import numpy as np

from abstract_rl.operator.gae_operator import _gae_jit
from abstract_rl.operator.retrace_operator import retrace_accumulate
from abstract_rl.policy.continuous.gaussian_math import kl_normal


def test_gae_matches_the_reverse_recursion():
    td = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    dones = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    advantages = _gae_jit(td, dones, jnp.array([3]), jnp.float32(0.5))
    # a_2 = 3, a_1 = 2 + 0.5 * 3, a_0 = 1 + 0.5 * a_1. The terminal mask kills the bootstrap.
    expected = np.array([[2.75, 3.5, 3.0]], dtype=np.float32)
    assert np.allclose(np.asarray(advantages), expected, atol=1e-6)


def test_retrace_skips_the_current_coefficient():
    q = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    td = jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32)
    # c_0 is huge on purpose: the product that multiplies later TD errors starts at c_1.
    coefficients = jnp.array([[9.0, 0.5, 0.25]], dtype=jnp.float32)
    targets = retrace_accumulate(q, td, coefficients, 0.5, bootstrap=3, lengths=jnp.array([3]))
    expected = np.array([[1.159375, 2.2375, 3.3]], dtype=np.float32)
    assert np.allclose(np.asarray(targets), expected, atol=1e-5)


def test_kl_matches_the_diagonal_gaussian_formula():
    mu = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    sigma = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    other_mu = jnp.array([[0.5, -1.0]], dtype=jnp.float32)
    other_sigma = jnp.array([[1.5, 0.5]], dtype=jnp.float32)
    got = float(kl_normal(mu, sigma, other_mu, other_sigma)[0])
    var_ratio = (sigma**2 + (mu - other_mu) ** 2) / (2.0 * other_sigma**2)
    expected = float((jnp.log(other_sigma / sigma) + var_ratio - 0.5).sum())
    assert np.isclose(got, expected, atol=1e-5)
