"""
 ``Proximal Policy Optimization (PPO)``  算法建模离散动作空间的入门示例（PyTorch版）
<link https://arxiv.org/pdf/1707.06347.pdf link>

PPO 是最经典常用的强化学习算法之一（策略梯度类）。它结合了经典的 Actor-Critic 范式和信赖域策略优化方法，并把相关工作整合为一个简洁而有效的算法。和之前传统的策略梯度类强化学习算法相比（例如 REINFORCE 和 A2C），PPO 可以更稳定高效地提升智能体策略，通过如下所示的截断式优化目标不断强化智能体：
$$J(\theta) = \min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
这个截断式的优化目标是原始未截断版本的一个下界（即一种悲观的约束）。通过公式中的 ``min`` 操作，忽略掉一些对于策略提升较大的重要性采样系数（IS），但又在策略优化变得糟糕时保留足够的 IS，从而让整个优化过程更稳定。
详细的数学符号定义可以在符号表中找到 <link https://github.com/opendilab/PPOxFamily/blob/main/chapter1_overview/chapter1_notation.pdf link> 。

离散动作空间是最常见的动作空间之一，例如像超级马里奥，雅达利（Atari）这样的视频游戏就通过这种动作空间来决策。具体来说，它包含一组可选的离散动作选项，每次决策需要从其中选择一个。离散动作空间常常被建模成广义伯努利分布来优化（类似一个分类问题）。

本文档将主要分为三个部分，读者可以从这些样例代码中一步一步进行学习，也可以将其中一些代码片段用到自己的程序中：
  - 策略神经网络架构
  - 动作采样函数
  - 主函数（测试函数）
更多的可视化结果和实际应用样例，可以参考这个链接 <link https://github.com/opendilab/PPOxFamily/issues/4 link>
"""
from typing import List
import torch
import torch.nn as nn


class DiscretePolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **DiscretePolicyNetwork 定义概述**:
            定义 PPO 中所使用的离散动作策略网络，其主要包含两部分：编码器（encoder）和决策输出头（head）
        """
        # 继承 PyTorch 神经网络类所必需的操作，自定义的神经网络必须是 ``nn.Module`` 的子类
        super(DiscretePolicyNetwork, self).__init__()
        # 定义编码器模块，将原始的状态映射为特征向量。对于不同的环境，可能状态信息的模态不同，需要根据情况选择适当的编码器神经网络，例如对于图片状态信息就常常使用卷积神经网络
        # 这里我们用一个简单的单层 MLP 作为例子，即:
        # $$y = max(Wx+b, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
        )
        # 定义离散动作的决策输出头，一般仅仅一层全连接层即可，即:
        # $$y=Wx+b$$
        self.head = nn.Linear(32, action_shape)

    # delimiter
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        **forward 函数功能概述**:
            描述 PPO 中所使用的离散动作策略网络的前向计算图
            ``x -> encoder -> head -> logit`` .
        """
        # 将原始的状态信息转换为特征向量，维度变化为: $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # 为每个可能的离散动作选项，计算相应的 logit，维度变化为: $$(B, N) -> (B, A)$$
        logit = self.head(x)
        return logit


# delimiter
class MultiDiscretePolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: List[int]) -> None:
        """
        **MultiDiscretePolicyNetwork 定义概述**:
            定义 PPO 中所使用的多维离散动作策略网络，其主要包含两部分：编码器（encoder）和多维决策输出头（head）
        """
        # 继承 PyTorch 神经网络类所必需的操作，自定义的神经网络必须是 ``nn.Module`` 的子类
        super(MultiDiscretePolicyNetwork, self).__init__()
        # 定义编码器模块，将原始的状态映射为特征向量。对于不同的环境，可能状态信息的模态不同，需要根据情况选择适当的编码器神经网络，例如对于图片状态信息就常常使用卷积神经网络
        # 这里我们用一个简单的单层 MLP 作为例子，即:
        # $$y = max(Wx+b, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
        )
        # 定义多维离散动作的决策输出头，即创建多个离散动作的预测器，整体用 ``nn.ModuleList`` 进行管理
        self.head = nn.ModuleList()
        for size in action_shape:
            self.head.append(nn.Linear(32, size))

    # delimiter
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        **Overview**:
            The computation graph of discrete action policy network used in PPO.
            ``x -> encoder -> multiple head -> multiple logit`` .
        """
        # 将原始的状态信息转换为特征向量，维度变化为: $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # 为每个可能的离散动作选项，计算相应的 logit，维度变化为: $$(B, N) -> [(B, A_1), ..., (B, A_N)]$$
        logit = [h(x) for h in self.head]
        return logit


# delimiter
def sample_action(logit: torch.Tensor) -> torch.Tensor:
    """
    **sample_action 函数功能概述**:
        输入 logit 采样获得离散动作，输入维度为 (B, action_shape) 输出维度为 output shape = (B, )
        在这个示例中，课程中提到的 distributions 工具库的三个维度分别为
        batch_shape = (B, ), event_shape = (), sample_shape = ()
    """
    # 将 logit 转化为概率（logit 一般指神经网络的原始输出，不经过激活函数，例如最后一层全连接层的输出）
    # $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
    prob = torch.softmax(logit, dim=-1)
    # 构建广义伯努利分布。它的概率质量函数为: $$f(x=i|\boldsymbol{p})=p_i$$
    # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
    dist = torch.distributions.Categorical(probs=prob)
    # 为一个 batch 里的每个样本采样一个离散动作，并返回它
    return dist.sample()


# delimiter
def test_sample_discrete_action():
    """
    **test_sample_discrete_action 函数功能概述**:
        离散动作空间的主函数，构建一个简单的策略网络，执行前向计算过程，并采样得到一组离散动作
    """
    # 设置相关参数 batch_size = 4, obs_shape = 10, action_shape = 6.
    B, obs_shape, action_shape = 4, 10, 6
    # 从0-1 的均匀分布中生成状态数据
    state = torch.rand(B, obs_shape)
    # 定义策略网络
    policy_network = DiscretePolicyNetwork(obs_shape, action_shape)
    # 策略网络执行前向计算，即输入状态输出 logit
    # $$ logit = \pi(a|s)$$
    logit = policy_network(state)
    assert logit.shape == (B, action_shape)
    # 根据 logit 采样得到最终的离散动作
    action = sample_action(logit)
    assert action.shape == (B, )


# delimiter
def test_sample_multi_discrete_action():
    """
    **test_sample_multi_discrete_action 函数功能概述**:
        多维离散动作空间的主函数，构建一个简单的策略网络，执行前向计算过程，并采样得到一组多维离散动作
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = [4, 5, 6].
    B, obs_shape, action_shape = 4, 10, [4, 5, 6]
    # 从0-1 的均匀分布中生成状态数据
    state = torch.rand(B, obs_shape)
    # 定义策略网络
    policy_network = MultiDiscretePolicyNetwork(obs_shape, action_shape)
    # 策略网络执行前向计算，即输入状态输出多个 logit
    # $$ logit = \pi(a|s)$$
    logit = policy_network(state)
    for i in range(len(logit)):
        assert logit[i].shape == (B, action_shape[i])
    # 对于多个 logit，依次调用采样函数，根据相应的 logit 采样得到最终的离散动作
    for i in range(len(logit)):
        action_i = sample_action(logit[i])
        assert action_i.shape == (B, )


if __name__ == "__main__":
    test_sample_discrete_action()
    test_sample_multi_discrete_action()
