import math
import torch
import torch.nn as nn
from torch import exp, tanh
from torch.nn import Parameter
from torch.distributions import Normal

g = 9.8  # 重力加速度
d1, d2 = 0.2, 0.2
N = 10  # batch size

torch.manual_seed(1000)

def isJumpOver(act):
    theta, v = act[:, 0], act[:, 1]
    # 动力学公式
    vx, vy = v * torch.cos(theta), v * torch.sin(theta)
    t = d1 / (vx + 1e-8)  # 防止vx为0算得t为inf，会错误地导致在v=0时也能跳过去
    dy = vy * t - 1 / 2 * g * t ** 2
    # 奖励函数
    r = 100 - v**2
    # r[dy < d2] = 0  # 初始化的网络参数，没有一次能跳过去，r全部为0，没有梯度，因此注释掉这句
    return r

# 定义策略的可学习参数
linear = nn.Linear(2, 2)
param = Parameter(torch.tensor([0.0, 0.0]))

# 前向传播得到动作
obs = torch.tensor([d1, d2])
# obs = torch.tile(obs, (N, 1))  # 将obs复制N份作为batch
# 没必要复制N份，直接在同一个distribution中sampleN次动作即可
x = tanh(linear(obs))
mu = 1/2 * (x + 1) * torch.tensor([math.pi/2, 10])
sigma = exp(param) * torch.tensor([math.pi/2, 10])

# mu, sigma have grad, but act not have, so we need to use reparam trick!
# can log_prob of torch.distribution automatically conduct reparam trick?
dist = Normal(mu, sigma)
# dist2 = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
act = dist.sample([N])
# act2 = sigma * dist.sample([N]) + mu
act = torch.clamp(act, torch.tensor([0, 0]), torch.tensor([math.pi/2, 10]))

# 计算梯度
log_prob = dist.log_prob(act).sum(dim=-1)  # 直观上不同的动作维度是独立的，logprob可以相加，等价于各维度prob相乘
r = isJumpOver(act)
loss = torch.mean(r * -log_prob)
loss.backward()

for name, param in linear.named_parameters():
    print(f'for {name} in linear, grad = {param.grad}')
print(f'for param, grad = {param.grad}')







