# -*- coding: utf-8 -*-
# @Time    : 2023-01-17 22:19
# @Author  : 吴佳杨

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, obs_shape: int):
        """
        策略函数（对pytorch不是很熟）
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(obs_shape, 2)
        self.param = nn.Parameter(torch.zeros(2))
        self.w = torch.tensor([torch.pi / 2, 10])

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.linear(x))
        x = (x + 1) / 2
        mu = x * self.w                         # shape = (batch_size, 2)
        sigma = torch.exp(self.param) * self.w  # shape = (2,)
        return mu, sigma


def get_reward(d1, d2, action):
    theta, v = action[0], action[1]
    v_x = v * torch.cos(theta)
    v_y = v * torch.sin(theta)
    t = d1 / v_x
    delta_y = v_y * t - 0.5 * 9.8 * t * t
    if delta_y >= d2:
        reward = 100 - v ** 2
    else:
        reward = 0
    return reward


def sample_action(mu, sigma):
    """
    根据mu,sigma采样动作

    :param mu: shape = (2,)
    :param sigma: shape = (2,)
    :return: action: shape = (2,)
    """
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    action[0] = torch.clip(action[0], min=0, max=0.5 * torch.pi)
    action[1] = torch.clip(action[1], min=0, max=10)
    return action


def loss_fn(mu, sigma, action, g_reward):
    dist = torch.distributions.Normal(mu, sigma)
    log_prob = dist.log_prob(action)
    loss = -(log_prob * g_reward).mean()
    return loss


def main():
    d1, d2 = 0.2, 0.2
    model = Model(obs_shape=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.tensor([[d1, d2]])  # batch_size = 1
    mu, sigma = model(x)
    mu = mu[0]
    print("mu: ", mu)
    print("sigma: ", sigma)
    action = sample_action(mu, sigma)
    print("action: ", action)
    reward = get_reward(d1, d2, action)
    print("reward", reward)
    loss = loss_fn(mu, sigma, action, reward)
    print("loss: ", loss)
    loss.backward()
    for name, param in model.named_parameters():
        print('name:', name)
        print('grad:', param.grad)
        print()


main()
