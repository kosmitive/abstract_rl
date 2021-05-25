import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sac.rl_util import weights_init_


class QNetwork(nn.Module):

    def __init__(self, env, hidden_dim):
        super(QNetwork, self).__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = hidden_dim

        # Q1 architecture
        self.linear1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def theta(self):
        return self.parameters()

    def params_q1(self):
        return list(self.linear1.parameters()) \
               + list(self.linear2.parameters()) \
               + list(self.linear3.parameters())

    def params_q2(self):
        return list(self.linear4.parameters()) \
               + list(self.linear5.parameters()) \
               + list(self.linear6.parameters())