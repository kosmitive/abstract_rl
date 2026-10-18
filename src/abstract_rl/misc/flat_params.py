import numpy as np
import torch


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """Extract flat parameters form pytorch model."""

    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model: torch.nn.Module, flat_params: torch.Tensor) -> None:
    """Set flat parameters to pytorch model."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
