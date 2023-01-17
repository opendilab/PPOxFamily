import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.register_buffer('action_space', torch.tensor([10, np.pi / 2]))
        self.fc = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh()
        )
        self.sigma = nn.Parameter(torch.FloatTensor([1, 1]))

    def forward(self, x, return_p=False):
        mu = 0.5 * (self.fc(x) + 1) * self.action_space
        sigma = torch.exp(self.sigma) * self.action_space[None, :]
        
        epsilon = torch.randn_like(mu)
        action = mu + sigma * epsilon

        p = None
        if return_p:
            p = torch.exp(-0.5 * (action.detach() - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
            p = p.prod(dim=1)

        return action, mu, sigma, p
