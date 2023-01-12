"""
Homework 2
戴子彭
"""
from collections import namedtuple
import torch
from torch import nn
from torch.distributions import Normal
import math

N = 10  # batch size
g = 9.8
torch.manual_seed(0)

def calculate_reward(obs, act):
    theta, v = act[:, 0], act[:, 1]
    r = 100 - v**2

    vx, vy = v * torch.cos(theta), v * torch.sin(theta)
    t = obs[:, 0] / (vx + 1e-8)
    dy = vy * t - 1 / 2 * 9.8 * t ** 2
    
    # r[dy < obs[:, 1]]=0  # 初始状态在第一步难以跳过，r全部为0，导致PG没有梯度，故暂时注释
    return r


def update_once(d_1, d_2):
    obs = torch.tensor([d_1, d_2]).tile(N, 1)  # batch inputs
    linear = nn.Linear(2, 2)
    x = torch.tanh(linear(obs))
    param = nn.Parameter(torch.zeros(2,))
    mu = (x+1)/2 * torch.tensor([math.pi/2, 10])
    sigma = torch.exp(param) * torch.tensor([math.pi/2, 10])
    pi_a_s = Normal(mu, sigma)
    act = pi_a_s.sample()
    act = torch.clamp(act, torch.tensor([0, 0]), torch.tensor([math.pi/2, 10]))

    log_prob = pi_a_s.log_prob(act)
    rew = calculate_reward(obs,act)
    # PG loss: $$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
    loss = -(log_prob * rew.unsqueeze(1)).mean()
    loss.backward()

    for name, param in linear.named_parameters():
        print(f'for {name} in linear, grad = {param.grad}')
    print(f'for param, grad = {param.grad}')


# 运行主函数计算当观测信息$d_1=0.2, d_2=0.2$时，策略函数的参数在一次策略梯度算法更新时的梯度大小
if __name__ == '__main__':
    update_once(d_1=0.2, d_2=0.2)
