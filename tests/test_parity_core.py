import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.mixed.conjugate_gradient import conjugate_gradient
from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import find_optimal_temperature
from abstract_rl.misc.flat_params import get_flat_params_from, set_flat_params_to
from golden import golden


def test_cg_matches_torch():
    g = golden("cg")
    a = jnp.asarray(g["A"])
    x = conjugate_gradient(lambda v: a @ v, jnp.asarray(g["b"]), 10, 1e-10)
    np.testing.assert_allclose(x, g["x"], rtol=1e-4, atol=1e-5)


def test_flat_params_round_trip():
    mlp = eqx.nn.MLP(3, 2, 4, 1, key=jax.random.key(0))
    flat = get_flat_params_from(mlp)
    assert flat.ndim == 1 and flat.size == sum(x.size for x in jax.tree.leaves(eqx.filter(mlp, eqx.is_inexact_array)))
    back = set_flat_params_to(mlp, flat + 1.0)
    np.testing.assert_allclose(get_flat_params_from(back), flat + 1.0)
    assert mlp.layers[0].weight is not back.layers[0].weight  # the input is not mutated


def test_temperature_dual_matches_torch():
    for case in ("cold", "warm"):
        g = golden(f"dual_{case}")
        eta = find_optimal_temperature(jnp.asarray(g["q"]), 0.1, float(g["eta0"]), 0.05)
        # D4: float32 here, float64 in torch
        np.testing.assert_allclose(eta, g["eta"], rtol=1e-3)
