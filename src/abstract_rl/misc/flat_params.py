import torch
import numpy as np

def get_flat_grad_from(model):
    """Extract flat gradients form pytorch model."""

    params = []
    for param in model.parameters():
        params.append(param.grad.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def get_flat_grad_from(model):
    params = []
    for param in model.parameters():
        params.append(param.grad.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def get_flat_params_from(model):
    """Extract flat parameters form pytorch model."""

    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """Set flat parameters to pytorch model."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
