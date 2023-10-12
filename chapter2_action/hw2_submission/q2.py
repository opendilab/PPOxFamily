# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2023/1/17 10:48 上午
# Description : 
"""
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import math
class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:

        super(PolicyNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, action_shape),
            nn.Tanh(),
        )
        self.sigma_param = nn.Parameter(torch.zeros(2,))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x = self.encoder(x)
        mu = (x+1)/2 * torch.tensor([math.pi/2, 10])
        sigma = torch.exp(self.sigma_param) * torch.tensor([math.pi/2, 10])

        return {'mu': mu, 'sigma': sigma}

    def update(self, obs):
        t = self.forward(obs)
        pi_a_s = Normal(t["mu"], t["sigma"])
        act = pi_a_s.sample()
        act = torch.clamp(act, torch.tensor([0, 0]), torch.tensor([math.pi / 2, 10]))
        log_prob = pi_a_s.log_prob(act)

        rew = get_reward(obs, act)
        # PG loss: $$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
        loss = -(log_prob * rew.unsqueeze(1)).mean()
        loss.backward()

        for name, param in self.encoder.named_parameters():
            print(f'{name} in linear, grad = {param.grad}')




def get_reward(obs, act):
    d1, d2 = obs[:, 0], obs[:, 1]
    theta, v = act[:, 0], act[:, 1]

    v_x = theta*torch.cos(v)
    v_y = theta*torch.sin(v)
    t = obs[:, 0] / (v_x + 1e-8)
    g = 9.8
    dy = v_y*t - g*np.power(t, 2)/2
    r = 100-np.power(v, 2)
    r[dy < d2] = 0
    return r



if __name__ == '__main__':
    policy = PolicyNetwork(2, 2)
    policy.update(torch.Tensor([[0.2,0.2]]))

