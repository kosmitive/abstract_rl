import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import TypeCheckError, jaxtyped

from abstract_rl.misc.typing import DTYPE, States, SuffStats


def test_dtype_is_float32():
    assert jnp.float32 == DTYPE


def test_aliases_check_shapes():
    @jaxtyped(typechecker=beartype)
    def f(s: States) -> SuffStats:
        return s[:, :2]

    assert f(jnp.zeros((4, 3))).shape == (4, 2)
    with pytest.raises(TypeCheckError):
        f(jnp.zeros((4,)))
