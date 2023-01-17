from typing import Dict
from collections import namedtuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import math

g = 9.8
pg_data = namedtuple('pg_data', ['logit', 'action', 'return_'])
pg_loss = namedtuple('pg_loss', ['policy_loss', 'entropy_loss'])

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **ContinuousPolicyNetwork 定义概述**:
            定义 PPO 中所使用的连续动作策略网络，其主要包含三部分：编码器（encoder），均值（mu）和对数空间标准差（log_sigma）
        """
        # 继承 PyTorch 神经网络类所必需的操作，自定义的神经网络必须是 ``nn.Module`` 的子类
        super(ContinuousPolicyNetwork, self).__init__()
        # 定义编码器模块，将原始的状态映射为特征向量。对于不同的环境，可能状态信息的模态不同，需要根据情况选择适当的编码器神经网络，例如对于图片状态信息就常常使用卷积神经网络
        # 这里我们用一个简单的两层 MLP 作为例子，即:
        # $$ y = max(W_2 max(W_1x+b_1, 0) + b_2, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 2),
            nn.Tanh(),
        )
        # 定义输出均值 mu 的模块，一般使用一层全连接层即可，输出的 mu 将用于构建高斯分布
        # $$ \mu = Wx + b $$
        # self.mu = nn.Linear(32, action_shape)
        # 定义对数空间标准差 log_sigma 模块，它是一个与输入状态无关的可学习参数。
        # 这里定义成对数空间，取值和使用比较方便。你也可以根据自己的需要，调整它的初始化值
        # $$\sigma = e^w$$
        self.log_sigma = nn.Parameter(torch.zeros(1, action_shape))

    # delimiter
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        **forward 函数功能概述**:
            描述 PPO 中所使用的连续动作策略网络的前向计算图
            ``x -> encoder -> mu -> \mu`` .
            ``log_sigma -> exp -> sigma`` .
        """
        # 将原始的状态信息转换为特征向量，维度变化为: $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # 根据特征向量输出动作均值 mu，维度变化为: $$(B, N) -> (B, A)$$
        mu = torch.mul(x+1, torch.tensor([[0.25*math.pi, 5]]))
        # 借助”广播“机制让对数空间标准差的维度和均值一致（在 batch 维度上复制）
        # ``zeros_like`` 操作并不会传递梯度
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.log_sigma + torch.zeros_like(mu)
        # 通过取指数操作得到最终的标准差 sigma
        # $$\sigma = e^w$$
        sigma = torch.mul(torch.exp(log_sigma), torch.tensor([[0.5*math.pi, 10]]))
        return {'mu': mu, 'sigma': sigma}


# delimiter
def sample_continuous_action(logit: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    **sample_continuous_action 函数功能概述**:
        输入 logit（包含 mu 和 sigma）采样得到离散动作，输入是一个包含 mu 和 sigma 的字典，它们的维度都是 (B, action_shape)，输出的维度是 (B, action_shape)。
        在这个示例中，课程中提到的 distributions 工具库的三个维度分别为
        batch_shape = (B, ), event_shape = (action_shape, ), sample_shape = ()
    """
    # 根据 mu 和 sigma 构建高斯分布
    # $$X \sim \mathcal{N}(\mu,\,\sigma^{2})$$
    # 它的概率密度函数为: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)$$
    # <link https://en.wikipedia.org/wiki/Normal_distribution link>
    dist = Normal(logit['mu'], logit['sigma'])
    # 将 ``action_shape`` 个高斯分布转义为一个有着对角协方差矩阵的多维高斯分布。
    # 并保证高斯分布中，每一维之间都是互相独立的（因为协方差矩阵是对角矩阵）
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    dist = Independent(dist, 1)
    # 为一个 batch 里的每个样本采样一个维度为 ``action_shape`` 的连续动作，并返回它
    return dist.sample()


def get_return(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    d1 = state[:, 0]
    d2 = state[:, 1]
    v = action[:, 0]
    v = torch.clip(v, min=0, max=10)
    theta = action[:, 1]
    theta = torch.clip(theta, min=0, max=0.5*math.pi)
    vx = v * torch.cos(theta)
    vy = v * torch.sin(theta)
    t = d1 / vx
    dy = vy * t - 0.5 * 9.8 * t ** 2
    r = torch.where(dy>=d2, 100-v**2, 0)
    return r


def pg_error(data: namedtuple) -> namedtuple:
    """
    **Overview**:
        Implementation of PG (Policy Gradient)
    """
    # Unpack data: $$<\pi(a|s), a, G_t>$$
    logit, action, return_ = data
    # Prepare policy distribution from logit and get log propability.
    dist = Normal(logit['mu'], logit['sigma'])
    # 将 ``action_shape`` 个高斯分布转义为一个有着对角协方差矩阵的多维高斯分布。
    # 并保证高斯分布中，每一维之间都是互相独立的（因为协方差矩阵是对角矩阵）
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    dist = Independent(dist, 1)
    log_prob = dist.log_prob(action)
    
    # Policy loss: $$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
    policy_loss = -(log_prob * return_).mean()
    # Entropy bonus: $$\frac 1 N \sum_{n=1}^{N} \pi(a^n|s^n) log(\pi(a^n|s^n))$$
    entropy_loss = dist.entropy().mean()
    # Return final loss.
    return pg_loss(policy_loss, entropy_loss)


# delimiter
def test_sample_continuous_action():
    """
    **test_sample_continuous_action 函数功能概述**:
        连续动作空间的主函数，构建一个简单的连续动作策略网络，执行前向计算过程，并采样得到一组连续动作
    """
    # 设置相关参数 batch_size = 4, obs_shape = 10, action_shape = 6.
    # ``action_shape`` 在离散和连续动作空间中的语义不太一样，前者是表示可选择的离散选项的个数，但只从中选出某一离散动作，而后者是连续动作的数量（维度）
    B, obs_shape, action_shape = 4, 2, 2
    # 从0-1 的均匀分布中生成状态数据
    state = torch.ones(B, obs_shape) * 0.2
    # 定义策略网络（类似重参数化方法）
    policy_network = ContinuousPolicyNetwork(obs_shape, action_shape)
    # 策略网络执行前向计算，即输入状态输出字典类型的 logit
    # $$ \mu, \sigma = \pi(a|s)$$
    logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['mu'].shape == (B, action_shape)
    assert logit['sigma'].shape == (B, action_shape)
    # 根据 logit (mu, sigma) 采样得到最终的连续动作
    action = sample_continuous_action(logit)
    assert action.shape == (B, action_shape)
    return_ = get_return(state, action)
    data = pg_data(logit, action, return_)
    loss = pg_error(data)
    assert all([l.shape == tuple() for l in loss])
    # assert logit.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    # assert isinstance(logit.grad, torch.Tensor)
    for name, param in policy_network.named_parameters():
        print('name:', name)
        print('grad:', param.grad)


if __name__ == "__main__":
    test_sample_continuous_action()