import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sac.rl_util import weights_init_


class VNetwork(nn.Module):

    def __init__(self, env, hidden_dim):
        super(VNetwork, self).__init__()

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def clone(self):

        v_net = VNetwork(self.env, self.hidden_dim)
        sd = self.state_dict()
        v_net.load_state_dict(sd)
        return v_net

    def psi(self):
        return self.parameters()

    def set_psi(self, psi):
        l = 0
        r = l
        for p in self.parameters():
            shp = p.shape
            r += np.prod(p.shape)
            p.data = psi[l:r].reshape(shp)
            l = r

    def exp_avg(self, rho, other_params):
        for p, q in zip(self.parameters(), other_params):
            p.data = rho * p.data + (1 - rho) * q.data
