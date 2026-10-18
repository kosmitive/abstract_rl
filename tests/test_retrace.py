import torch

from abstract_rl.operator.retrace_operator import (
    bounded_importance_coefficients,
    n_step_targets,
    retrace_targets,
)


def test_importance_coefficients_are_lambda_times_the_clipped_ratio():
    pi = torch.tensor([0.0, -1.0, 1.0])
    mu = torch.zeros(3)
    coefficients = bounded_importance_coefficients(pi, mu, trace_lambda=0.5)
    ratio = torch.exp(pi - mu)
    assert torch.allclose(coefficients, 0.5 * torch.minimum(ratio, torch.ones_like(ratio)))


def test_retrace_matches_the_truncated_product_of_coefficients():
    q = torch.tensor([1.0, 2.0, 3.0])
    rewards = torch.tensor([0.1, 0.2, 0.3])
    next_v = torch.tensor([4.0, 5.0, 6.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    coefficients = torch.tensor([0.5, 0.2, 0.8])
    discount = 0.5
    delta = rewards + discount * next_v * (1 - dones) - q
    targets = retrace_targets(q, rewards, next_v, dones, coefficients, discount, horizon=3)
    expected = torch.stack(
        [
            q[0]
            + delta[0]
            + discount * coefficients[1] * delta[1]
            + discount**2 * coefficients[1] * coefficients[2] * delta[2],
            q[1] + delta[1] + discount * coefficients[2] * delta[2],
            q[2] + delta[2],
        ]
    )
    assert torch.allclose(targets, expected, atol=1e-6)


def test_n_step_return_bootstraps_and_stops_at_termination():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    next_v = torch.tensor([10.0, 20.0, 30.0])
    dones = torch.tensor([0.0, 1.0, 0.0])
    targets = n_step_targets(rewards, next_v, dones, discount=0.5, n_step=2)
    # t = 0 reaches the terminal reward at t = 1 and does not bootstrap
    # t = 1 is itself terminal
    # t = 2 has only one stored reward, so it bootstraps from V(s_3)
    expected = torch.tensor([1.0 + 0.5 * 1.0, 1.0, 1.0 + 0.5 * 30.0])
    assert torch.allclose(targets, expected, atol=1e-6)
