import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from src.sac.rl_util import weights_init_, epsilon, tt


class GaussianPolicy(nn.Module):

    def __init__(self, env, hidden_dim):
        super(GaussianPolicy, self).__init__()

        action_space = env.action_space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = hidden_dim

        # the linear layers at the beginning
        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # two separate layers for the mean and log standard deviation
        self.mean_layer = nn.Linear(hidden_dim, self.action_dim)
        self.chol_layer = nn.Linear(hidden_dim, self.action_dim)

        # apply the initializer of the weights
        self.apply(weights_init_)

        # rescale the actions based on the environment
        self.action_scale = torch.FloatTensor(action_space.high)

    def forward(self, states):

        # check if the states have to be extended
        extend_states = len(states.shape) == 1

        # extend
        if extend_states: states = states.unsqueeze(0)

        # apply the network to the states
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))

        # obtain mean and cholesky diagonal factors
        mean = self.mean_layer(x)
        chol_diag = self.chol_layer(x)

        # shrink
        if extend_states:
            mean = mean.squeeze(0)
            chol_diag = chol_diag.squeeze(0)

        # build the covariance
        return mean, chol_diag

    def sample(self, states):

        # check if the states have to be extended
        extend_states = len(states.shape) == 1

        # extend
        if isinstance(states, np.ndarray): states = tt(states)
        if extend_states: states = states.unsqueeze(0)

        # get sufficient statistics
        mean, chol_diag = self.forward(states)
        chol_diag = F.softplus(chol_diag)
        chol = torch.stack([c * torch.eye(self.action_dim) for c in chol_diag])

        # obtain action in real space and bounded space
        action_distribution = dist.MultivariateNormal(mean, scale_tril=chol)
        unscaled_action = action_distribution.rsample()
        norm_action = torch.tanh(unscaled_action)
        action = self.action_scale * norm_action

        # calculate the log probability
        log_prob = action_distribution.log_prob(unscaled_action).unsqueeze(1)
        log_prob -= torch.log(self.action_scale * (1 - norm_action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = self.action_scale * torch.tanh(mean)

        if extend_states:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            mean = mean.squeeze(0)

        return action, log_prob, mean

    def phi(self):
        return torch.cat([p.view(-1) for p in self.parameters() if p.requires_grad])
