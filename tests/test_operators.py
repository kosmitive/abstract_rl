import numpy as np
import torch

from abstract_rl.algorithms.mixed.temperature_dual import find_optimal_temperature


def test_temperature_satisfies_kl_bound():
    torch.manual_seed(0)
    q = torch.randn(256, 20) * 10
    eps = 0.1
    eta = find_optimal_temperature(q, eps).item()

    # the KL between the reweighted and the uniform (sampling) distribution matches the bound
    w = torch.softmax(q / eta, dim=1)
    kl = torch.mean(torch.sum(w * torch.log(w * q.shape[1] + 1e-12), dim=1)).item()
    assert eta > 0
    assert np.isclose(kl, eps, rtol=0.1, atol=0.01)
