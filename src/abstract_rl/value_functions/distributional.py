"""Optional distributional critics. Both are off unless the value config selects them.

C51 is the categorical projection of Bellemare et al., 2017. The mixture of Gaussians regresses the scalar target
with a diagonal mixture; it is a lighter alternative, not a distributional Bellman backup.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


def c51_projection(target: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Project a Dirac at ``target`` onto the categorical support.

    Args:
        target: Scalar returns of shape ``[batch]``.
        atoms: Increasing support of shape ``[num_atoms]``.

    Returns:
        A probability mass of shape ``[batch, num_atoms]``. Rows sum to one.
    """

    target = target.view(-1)
    num_atoms = int(atoms.numel())
    v_min, v_max = atoms[0], atoms[-1]
    delta = (v_max - v_min) / (num_atoms - 1)
    bounded = target.clamp(v_min, v_max)
    position = (bounded - v_min) / delta
    lower = position.floor().long()
    upper = position.ceil().long()
    mass_upper = (position - lower.float()).clamp(0, 1)
    mass_lower = (upper.float() - position).clamp(0, 1)
    same = lower == upper
    mass_lower = torch.where(same, torch.ones_like(mass_lower), mass_lower)
    mass_upper = torch.where(same, torch.zeros_like(mass_upper), mass_upper)
    mass = target.new_zeros(target.shape[0], num_atoms)
    mass.scatter_add_(1, lower.clamp(0, num_atoms - 1).unsqueeze(1), mass_lower.unsqueeze(1))
    mass.scatter_add_(1, upper.clamp(0, num_atoms - 1).unsqueeze(1), mass_upper.unsqueeze(1))
    return mass


def c51_loss(logits: torch.Tensor, target: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the projected categorical target. The target carries no gradient.

    Args:
        logits: Unnormalized atom scores of shape ``[batch, num_atoms]``.
        target: Scalar backup of shape ``[batch]``.
        atoms: Support of shape ``[num_atoms]``.

    Returns:
        The mean cross-entropy.
    """

    mass = c51_projection(target.detach(), atoms)
    return -(mass * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def c51_expectation(logits: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Expected value of a categorical distribution.

    Args:
        logits: Unnormalized atom scores of shape ``[batch, num_atoms]``.
        atoms: Support of shape ``[num_atoms]``.

    Returns:
        A column vector of shape ``[batch, 1]``.
    """

    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities * atoms).sum(dim=-1, keepdim=True)


def split_mixture(raw: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a mixture head into means, positive standard deviations and logits.

    Args:
        raw: Network output of shape ``[batch, 3 * n_components]``.
        n_components: Number of Gaussian components.

    Returns:
        Means, standard deviations and mixture logits, each ``[batch, n_components]``.
    """

    means, raw_std, logits = raw.split(n_components, dim=-1)
    return means, F.softplus(raw_std) + 1e-6, logits


def mixture_mean(raw: torch.Tensor, n_components: int) -> torch.Tensor:
    """Mean of a Gaussian mixture, shape ``[batch, 1]``."""

    means, _, logits = split_mixture(raw, n_components)
    return (torch.softmax(logits, dim=-1) * means).sum(dim=-1, keepdim=True)


def mog_nll(raw: torch.Tensor, target: torch.Tensor, n_components: int) -> torch.Tensor:
    """Negative log likelihood of a scalar target under a Gaussian mixture.

    Args:
        raw: Network output of shape ``[batch, 3 * n_components]``.
        target: Scalar backup of shape ``[batch]``. The gradient does not flow into it.
        n_components: Number of Gaussian components.

    Returns:
        The mean negative log likelihood.
    """

    means, std, logits = split_mixture(raw, n_components)
    point = target.detach().view(-1, 1)
    component_log = F.log_softmax(logits, dim=-1) + Normal(means, std).log_prob(point)
    return -torch.logsumexp(component_log, dim=-1).mean()
