import torch
from torch import nn


def get_flat_grad_from(model: nn.Module) -> torch.Tensor:
    """Extract flat gradients from a pytorch model (parameters without gradient contribute zeros)."""
    grads = [torch.zeros_like(p).view(-1) if p.grad is None else p.grad.reshape(-1) for p in model.parameters()]
    return torch.cat(grads)


def get_flat_params_from(
    model: nn.Module,
) -> torch.Tensor:
    """Extract flat parameters from a pytorch model."""
    return torch.cat([param.data.reshape(-1) for param in model.parameters()])


def set_flat_params_to(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set flat parameters to a pytorch model."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
