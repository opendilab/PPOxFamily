"""
 ``Proximal Policy Optimization (PPO)``  算法建模连续动作空间的入门示例（PyTorch版）
<link https://arxiv.org/pdf/1707.06347.pdf link>

PPO 是最经典常用的强化学习算法之一（策略梯度类）。它结合了经典的 Actor-Critic 范式和信赖域策略优化方法，并把相关工作整合为一个简洁而有效的算法。和之前传统的策略梯度类强化学习算法相比（例如 REINFORCE 和 A2C），PPO 可以更稳定高效地提升智能体策略，通过如下所示的截断式优化目标不断强化智能体：
$$J(\theta) = \min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
这个截断式的优化目标是原始未截断版本的一个下界（即一种悲观的约束）。通过公式中的 ``min`` 操作，忽略掉一些对于策略提升较大的重要性采样系数（IS），但又在策略优化变得糟糕时保留足够的 IS，从而让整个优化过程更稳定。
详细的数学符号定义可以在符号表中找到 <link https://github.com/opendilab/PPOxFamily/blob/main/chapter1_overview/chapter1_notation.pdf link> 。

混合动作空间常常用于一些复杂的实践决策应用，例如《星际争霸2》和《王者荣耀》。它包含一系列可控制的决策变量，形式上可以用一棵树来表示，树的中间节点表示多级的离散选择，而叶节点则可以是任意的离散或连续动作空间。由于这种复杂的结构，混合动作空间需要更加复杂的算法设计和代码实现。

本文档将主要分为三个部分，其中包含使用 mask 和 treetensor 的相关代码逻辑，读者可以从这些样例代码中一步一步进行学习，也可以将其中一些代码片段用到自己的程序中：
  - 策略神经网络架构
  - 动作采样函数
  - 主函数（测试函数）
更多的可视化结果和实际应用样例，可以参考这个链接 <link https://github.com/opendilab/PPOxFamily/issues/4 link>

P.S, 如果需要安装 treetesor，你可以使用这样的命令
 ``pip install DI-treetensor`` .
"""
from typing import Dict
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from torch.distributions import Normal, Independent


class HybridPolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: Dict[str, int]) -> None:
        """
        **HybridPolicyNetwork 概述**:
            定义 PPO 中所使用的混合动作空间（这里特指参数化动作空间）策略网络，其主要包含三部分：编码器（encoder），动作类型预测器（离散）和动作参数预测器（continuous）
        """
        # 继承 PyTorch 神经网络类所必需的操作，自定义的神经网络必须是 ``nn.Module`` 的子类
        super(HybridPolicyNetwork, self).__init__()
        # 定义编码器模块，将原始的状态映射为特征向量。对于不同的环境，可能状态信息的模态不同，需要根据情况选择适当的编码器神经网络，例如对于图片状态信息就常常使用卷积神经网络
        # 这里我们用一个简单的两层 MLP 作为例子，即:
        # $$ y = max(W_2 max(W_1x+b_1, 0) + b_2, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        # 定义离散动作类型预测器，输出离散动作的 logit
        # $$ y = Wx + b $$
        self.action_type_shape = action_shape['action_type_shape']
        self.action_type_head = nn.Linear(32, self.action_type_shape)
        # 定义连续动作参数预测器，类似 PPO 在连续动作上的设计，输出相应的 mu 和 sigma
        # $$ \mu = Wx + b $$
        # $$\sigma = e^w$$
        self.action_args_shape = action_shape['action_args_shape']
        self.action_args_mu = nn.Linear(32, self.action_args_shape)
        self.action_args_log_sigma = nn.Parameter(torch.zeros(1, self.action_args_shape))

    # delimiter
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        **forward 函数功能概述**:
            描述 PPO 中所使用的混合动作策略网络的前向计算图
            ``x -> encoder -> action_type_head -> action_type_logit`` .
            ``x -> encoder -> action_args_mu -> \mu`` .
            ``action_args_log_sigma -> exp -> sigma`` .
        """
        # 将原始的状态信息转换为特征向量，维度变化为: $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # 输出离散动作类型的 logit
        logit = self.action_type_head(x)
        # 根据特征向量输出动作均值 mu
        mu = self.action_args_mu(x)
        # 借助”广播“机制让对数空间标准差的维度和均值一致（在 batch 维度上复制
        # ``zeros_like`` 操作并不会传递梯度
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.action_args_log_sigma + torch.zeros_like(mu)
        # 通过取指数操作得到最终的标准差 sigma
        # $$\sigma = e^w$$
        sigma = torch.exp(log_sigma)
        # 返回 treetensor 类型的输出
        return ttorch.as_tensor({'action_type': logit, 'action_args': {'mu': mu, 'sigma': sigma}})


# delimiter
def sample_hybrid_action(logit: ttorch.Tensor) -> torch.Tensor:
    """
    **sample_hybrid_action 函数功能概述**:
        输入 logit 采样得到混合动作，输入是一个 treetensor 包含 ``action_type`` 和 ``action_args`` 两部分
    """
    # 将 logit 转化为概率（logit 一般指神经网络的原始输出，不经过激活函数，例如最后一层全连接层的输出）
    # $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
    prob = torch.softmax(logit.action_type, dim=-1)
    # 构建广义伯努利分布。它的概率质量函数为: $$f(x=i|\boldsymbol{p})=p_i$$
    # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
    discrete_dist = torch.distributions.Categorical(probs=prob)
    # 为一个 batch 里的每个样本采样一个离散动作，并返回它
    action_type = discrete_dist.sample()

    # 根据 mu 和 sigma 构建高斯分布
    # $$X \sim \mathcal{N}(\mu,\,\sigma^{2})$$
    # 它的概率密度函数为: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)$$
    # <link https://en.wikipedia.org/wiki/Normal_distribution link>
    continuous_dist = Normal(logit.action_args.mu, logit.action_args.sigma)
    # 将 ``action_shape`` 个高斯分布转义为一个有着对角协方差矩阵的多维高斯分布。
    # 并保证高斯分布中，每一维之间都是互相独立的（因为协方差矩阵是对角矩阵）
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    continuous_dist = Independent(continuous_dist, 1)
    # 为一个 batch 里的每个样本采样一个维度为 ``action_shape`` 的连续参数
    action_args = continuous_dist.sample()
    # 返回最终的混合动作（参数化动作）
    return ttorch.as_tensor({
        'action_type': action_type,
        'action_args': action_args,
    })


# delimiter
def test_sample_hybrid_action():
    """
    **test_sample_hybrid_action 函数功能概述**:
        混合动作空间的主函数，构建一个混合动作空间的策略网络，执行前向计算过程，并采样得到一组混合动作
    """
    # 设置相关参数 batch_size = 4, obs_shape = 10, action_shape 是一个字典，包含 3 种可选择的离散动作类型选项和 3 种对应的连续参数。动作类型和参数之间的关系将用下方的 mask 变量来表示
    B, obs_shape, action_shape = 4, 10, {'action_type_shape': 3, 'action_args_shape': 3}
    mask = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # 从0-1 的均匀分布中生成状态数据
    state = torch.rand(B, obs_shape)
    # 定义策略网络，包含编码器，离散动作类型预测器和连续动作参数预测器
    policy_network = HybridPolicyNetwork(obs_shape, action_shape)
    # 策略网络执行前向计算，即输入状态输出 treetensor 类型的 logit
    logit = policy_network(state)
    assert isinstance(logit, ttorch.Tensor)
    assert logit.action_type.shape == (B, action_shape['action_type_shape'])
    assert logit.action_args.mu.shape == (B, action_shape['action_args_shape'])
    assert logit.action_args.sigma.shape == (B, action_shape['action_args_shape'])
    # 根据 logit 中的相关部分采样相应的动作部分
    action = sample_hybrid_action(logit)
    assert action.action_type.shape == (B, )
    assert action.action_args.shape == (B, action_shape['action_args_shape'])
    # 通过动作类型查找获得每个数据样本具体所对应的 mask
    data_mask = torch.as_tensor([mask[i] for i in action.action_type]).bool()
    # 用 mask 选择（过滤）出动作类型相对应的动作参数，并重新赋值
    filtered_action_args = ttorch.masked_select(action.action_args, data_mask)
    action.action_args = filtered_action_args
    assert action.action_args.shape == (B, )
    # （treetensor 使用举例）通过切片操作选择部分训练样本
    selected_action = action[1:3]
    assert selected_action.action_type.shape == (2, )


if __name__ == "__main__":
    test_sample_hybrid_action()
