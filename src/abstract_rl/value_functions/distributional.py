"""Optional mixture-of-Gaussians critic, off unless the value config selects it.

The head matches Acme's Gaussian mixture: unnormalized component logits, means, and scales passed through a softplus.
The backup draws samples from the target mixture and projects them with ``reward + discount * (1 - done) * sample``.
The online mixture is trained by the mean negative log likelihood of those samples. The E-step uses the mixture mean.
"""

import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Acme's GaussianMixture keeps every scale at least this far above zero.
MIN_SCALE = 1e-4
# softplus(0) = log(2); dividing by it makes a zero pre-activation reproduce init_scale.
_SOFTPLUS_AT_ZERO = math.log(2.0)


def split_mixture(
    raw: torch.Tensor, n_components: int, init_scale: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a mixture head into means, positive scales and logits.

    Args:
        raw: Network output of shape ``[batch, 3 * n_components]``, ordered means, raw scales, logits.
        n_components: Number of Gaussian components.
        init_scale: Scale reproduced by a zero pre-activation, before the floor is added.

    Returns:
        Means, scales and mixture logits, each ``[batch, n_components]``.
    """

    means, raw_scale, logits = raw.split(n_components, dim=-1)
    scale_factor = float(init_scale) / _SOFTPLUS_AT_ZERO
    scales = scale_factor * F.softplus(raw_scale) + MIN_SCALE
    return means, scales, logits


def mixture_mean(means: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean of a Gaussian mixture, shape ``[batch, 1]``.

    Args:
        means: Component locations of shape ``[batch, components]``.
        logits: Unnormalized mixture weights, same shape.

    Returns:
        The probability-weighted mean of the components.
    """

    return (torch.softmax(logits, dim=-1) * means).sum(dim=-1, keepdim=True)


def sample_mixture(means: torch.Tensor, scales: torch.Tensor, logits: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Draw samples from a univariate Gaussian mixture.

    Args:
        means: Component locations of shape ``[batch, components]``.
        scales: Positive component scales, same shape.
        logits: Unnormalized mixture weights, same shape.
        num_samples: Number of independent draws per batch row.

    Returns:
        Samples of shape ``[num_samples, batch]``.
    """

    index = Categorical(logits=logits).sample((num_samples,)).unsqueeze(-1)
    expanded_means = means.unsqueeze(0).expand(num_samples, -1, -1)
    expanded_scales = scales.unsqueeze(0).expand(num_samples, -1, -1)
    loc = expanded_means.gather(-1, index).squeeze(-1)
    scale = expanded_scales.gather(-1, index).squeeze(-1)
    return torch.normal(loc, scale)


def mixture_log_prob(
    means: torch.Tensor, scales: torch.Tensor, logits: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Log density of a univariate Gaussian mixture.

    Args:
        means: Component locations of shape ``[batch, components]``.
        scales: Positive component scales, same shape.
        logits: Unnormalized mixture weights, same shape.
        values: Points at which to evaluate the density. The last dimension is the batch.

    Returns:
        Log probabilities with the batch axis left in place and the component axis removed.
    """

    extra = values.ndim - 1
    view = (1,) * extra + means.shape
    point = values.unsqueeze(-1)
    component_log = F.log_softmax(logits.view(view), dim=-1) + Normal(means.view(view), scales.view(view)).log_prob(
        point
    )
    return torch.logsumexp(component_log, dim=-1)


def project_mixture_samples(
    samples: torch.Tensor, reward: torch.Tensor, discount: float, dones: torch.Tensor
) -> torch.Tensor:
    """Project target-critic samples by the one-step backup ``reward + discount * (1 - done) * sample``.

    Args:
        samples: Draws from the target mixture, shape ``[num_samples, batch]``.
        reward: Immediate rewards, shape ``[batch]``.
        discount: Discount factor.
        dones: One where the transition ended and the bootstrap must vanish, shape ``[batch]``.

    Returns:
        Projected samples of the same shape. The result is detached.
    """

    bootstrap = (1.0 - dones.reshape(-1)) * float(discount)
    projected = reward.reshape(-1) + bootstrap * samples
    return projected.detach()


def mog_sample_nll(
    means: torch.Tensor, scales: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Mean negative log likelihood of projected target samples under the online mixture.

    Args:
        means: Online component locations of shape ``[batch, components]``.
        scales: Online scales, same shape.
        logits: Online mixture logits, same shape.
        targets: Stopped samples whose last dimension is the batch.

    Returns:
        A scalar mean over every sample and every batch row.
    """

    return -mixture_log_prob(means, scales, logits, targets).mean()
