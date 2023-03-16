import torch
from torch import nn
import numpy as np
from torch.distributions import Normal,Independent

import math
N = 8 # batch size


def env(d_1, d_2, theta, v):
    reward = 0
    v_x = v * math.cos(theta)
    v_y = v * math.sin(theta)
    t = d_1 / v_x
    delta_y = v_y * t - 0.5 * 9.8 * t*t
    if delta_y >= d_2:
        reward = 100 - v**2
    return reward

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork,self).__init__()
        self.mu = nn.Linear(2,2)
        self.sigma = nn.Parameter(torch.zeros(2))   # action_log_sigma [x,x]
    def forward(self,d):
        x = torch.tanh(self.mu(d))
        assert x.shape == torch.Size([2])
        logit_mu = x
        logit_sigma = self.sigma
        return logit_mu, logit_sigma   # policy 只输出最关键的几个元素。构建正态分布啦，概率分布函数求导啦都是其他函数的功能

def sample_action(logit_mu, logit_sigma):
    # mu_theta = 0.5 * (logit_mu[0] + 1) * 3.14 / 2
    # mu_v = 0.5 * (logit_sigma[1] + 1) * 10
    # mu = torch.tensor([mu_theta,mu_v])
    # sigma_theta = torch.exp(logit_sigma[0]) * 3.14 / 2
    # sigma_v = torch.exp(logit_sigma[1]) * 10
    # sigma = torch.tensor([sigma_theta,sigma_v])
    mu = 0.5 * (logit_mu + 1) * torch.tensor([3.14 / 2, 10.0])
    sigma = torch.exp(logit_sigma) * torch.tensor([3.14 / 2, 10.0])
    dist = Normal(mu, sigma)
    dist = Independent(dist, 1)
    return dist.sample()

def pg_error(logit_mu, logit_sigma, action, return_):
    # logit_mu 结构: [x0,x1]

    # mu_theta = 0.5 * (logit_mu[0] + 1) * 3.14 / 2
    # mu_v = 0.5 * (logit_mu[1] + 1) * 10
    # mu = torch.tensor([mu_theta,mu_v])                   # 这种写法是错误的，会导致梯度丢失
    mu = 0.5*(logit_mu+1)*torch.tensor([3.14/2,10.0])

    # sigma_theta = torch.exp(logit_sigma[0]) * 3.14 / 2
    # sigma_v = torch.exp(logit_sigma[1]) * 10
    # sigma = nn.Parameter(torch.zeros(2))
    # sigma.data[0] = sigma_theta
    # sigma.data[1] = sigma_v                             #  这种写法是错误的，会导致梯度丢失
    sigma = torch.exp(logit_sigma) * torch.tensor([3.14 / 2, 10.0])

    dist = Normal(mu, sigma)
    dist = Independent(dist, 1)
    log_prob = dist.log_prob(action)     # 把采样到的概率计算出来，就是 pi ,,并求log

    policy_loss = -(log_prob * return_)  # 这里就是 策略梯度更新公式 的计算值

    return policy_loss

def main():
    policy_net = PolicyNetwork()
    optimizer = torch.optim.Adam(policy_net.parameters(),lr=0.01)
    state = torch.tensor([0.2, 0.2])
    logit_mu, logit_sigma = policy_net(state)
    action = sample_action(logit_mu, logit_sigma)
    G = env(state[0],state[1],action[0],action[1])
    loss = pg_error(logit_mu, logit_sigma, action, return_=1)
    loss.backward()
    print("mu", policy_net.mu.weight)
    print("sigma",policy_net.sigma)
    print()
    print("mu_grad",policy_net.mu.weight.grad)
    print("sigma_grad",policy_net.sigma.grad)
    optimizer.step()
    optimizer.zero_grad()

if __name__ == "__main__":
    main()





