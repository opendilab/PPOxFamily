"""
 ``Proximal Policy Optimization (PPO)``  算法建模连续动作空间的入门示例（PyTorch版）
<link https://arxiv.org/pdf/1707.06347.pdf link>
PPO 是最经典常用的强化学习算法之一（策略梯度类）。它结合了经典的 Actor-Critic 范式和信赖域策略优化方法，并把相关工作整合为一个简洁而有效的算法。和之前传统的策略梯度类强化学习算法相比（例如 REINFORCE 和 A2C），PPO 可以更稳定高效地提升智能体策略，通过如下所示的截断式优化目标不断强化智能体：
$$J(\theta) = \min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
这个截断式的优化目标是原始未截断版本的一个下界（即一种悲观的约束）。通过公式中的 ``min`` 操作，忽略掉一些对于策略提升较大的重要性采样系数（IS），但又在策略优化变得糟糕时保留足够的 IS，从而让整个优化过程更稳定。
详细的数学符号定义可以在符号表中找到 <link https://github.com/opendilab/PPOxFamily/blob/main/chapter1_overview/chapter1_notation.pdf link> 。
连续动作空间是最常见的动作空间之一，它常常用于机器人操纵，无人机控制这样的决策问题中。一般包含一系列可控制的连续动作，每次决策时需要让 RL 智能体输出合适且精确的连续值。连续动作空间可以被直接回归，也可以被建模成一个高斯分布（整体类似一个回归问题）。
本文档将主要分为三个部分，读者可以从这些样例代码中一步一步进行学习，也可以将其中一些代码片段用到自己的程序中：
  - 策略神经网络架构
  - 动作采样函数
  - 主函数（测试函数）
更多的可视化结果和实际应用样例，可以参考这个链接 <link https://github.com/opendilab/PPOxFamily/issues/4 link>
"""
from typing import Dict
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import numpy as np
from jump_env import Jump_Env

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
            nn.Linear(obs_shape, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
        )
        # 定义输出均值 mu 的模块，一般使用一层全连接层即可，输出的 mu 将用于构建高斯分布
        # $$ \mu = Wx + b $$
        self.mu = nn.Linear(32, action_shape)
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
        mu_init = self.mu(x)
        input_init = nn.Tanh()
        mu = input_init(mu_init)
        mu_trans = torch.ones((4, 2))
        mu_trans1 = [i[0] for i in mu]
        mu_trans2 = [i[1] for i in mu]
        for i in range(len(mu)):
            mu_trans[i][0] = (mu_trans1[i]+1)/2*(np.pi/2)
            mu_trans[i][1] = (mu_trans2[i]+1)/2*10
        print('mu_trans:', mu_trans)
        print('mu:', mu)

        # 借助”广播“机制让对数空间标准差的维度和均值一致（在 batch 维度上复制）
        # ``zeros_like`` 操作并不会传递梯度
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.log_sigma + torch.zeros_like(mu_trans)

        # 通过取指数操作得到最终的标准差 sigma
        # $$\sigma = e^w$$
        sigma = torch.exp(log_sigma)
        sigma_trans = torch.ones((4, 2))
        sigma_trans1 = [i[0] for i in sigma]
        sigma_trans2 = [i[1] for i in sigma]
        for i in range(len(sigma)):
            sigma_trans[i][0] = sigma_trans1[i]*(np.pi/2)
            sigma_trans[i][1] = sigma_trans2[i]*10
        print('sigma_trans:', sigma_trans)
        return {'mu': mu_trans, 'sigma': sigma_trans}


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


# delimiter
def test_sample_continuous_action(state):
    """
    **test_sample_continuous_action 函数功能概述**:
        连续动作空间的主函数，构建一个简单的连续动作策略网络，执行前向计算过程，并采样得到一组连续动作
    """
    # 设置相关参数 batch_size = 4, obs_shape = 10, action_shape = 6.
    # ``action_shape`` 在离散和连续动作空间中的语义不太一样，前者是表示可选择的离散选项的个数，但只从中选出某一离散动作，而后者是连续动作的数量（维度）
    B, obs_shape, action_shape = 4, 2, 2
    # Jump_Env_model = Jump_Env
    # # 从0-1 的均匀分布中生成状态数据
    # state = Jump_Env_model().reset()
    print('state:',state)
    # 定义策略网络（类似重参数化方法）
    # state = torch.rand(B, obs_shape)
    policy_network = ContinuousPolicyNetwork(obs_shape, action_shape)
    # 策略网络执行前向计算，即输入状态输出字典类型的 logit
    # $$ \mu, \sigma = \pi(a|s)$$
    logit = policy_network(torch.tensor(state).float())
    # logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['mu'].shape == (B, action_shape)
    assert logit['sigma'].shape == (B, action_shape)
    # 根据 logit (mu, sigma) 采样得到最终的连续动作
    action = sample_continuous_action(logit)
    print('action:', action)
    # Jump_Env_model().step(state, action)
    assert action.shape == (B, action_shape)
    return logit, action


if __name__ == "__main__":
    test_sample_continuous_action()
