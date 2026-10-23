import mlx.core as mx
import numpy as np

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimization import find_optimal_temperature


def test_temperature_satisfies_kl_bound():
    q = np.random.default_rng(0).standard_normal((256, 20)).astype(np.float32) * 10
    eps = 0.1
    eta = find_optimal_temperature(mx.array(q), eps).item()

    # the KL between the reweighted and the uniform (sampling) distribution matches the bound
    z = q / eta
    w = np.exp(z - z.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    kl = float(np.mean(np.sum(w * np.log(w * q.shape[1] + 1e-12), axis=1)))
    assert eta > 0
    assert np.isclose(kl, eps, rtol=0.1, atol=0.01)
