import torch
import torch.nn as nn


epsilon = 1e-6

def tt(x): return torch.Tensor(x)
def tn(x): return x.detach().numpy()


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)
