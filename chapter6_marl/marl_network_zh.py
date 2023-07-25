"""
这是一个关于多智能体合作场景中神经网络的 PyTorch 教程，包括独立 Actor-Critic 网络 (是否共享参数两种) 和集中训练分散执行 (centralized training and decentralized execution, CTDE)
Actor-Critic 网络。所有示例都基于离散动作空间进行。

此教程主要由三部分组成，您可以按顺序学习这些部分，或者跳转到您感兴趣的部分：
  - 用于独立智能体的共享参数 Actor-Critic 网络
  - 用于独立智能体的独立参数 Actor-Critic 网络
  - 用于合作智能体的 CTDE Actor-Critic 网络
有关多智能体合作强化学习的更多细节，可以在 <link https://github.com/opendilab/PPOxFamily/blob/main/chapter6_marl/chapter6_lecture.pdf link> 中找到。
"""
import torch
import torch.nn as nn
import treetensor.torch as ttorch


class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **功能概述**:
            在策略梯度中算法 (如: PG/A2C/PPO) 中， 基础 Actor-Critic 网络的定义
            主要包括三个部分：编码器、策略分支网络、价值分支网络
        """
        # PyTorch 在继承 ``nn.Module`` 类的时候，必须执行这个初始化方法。
        super(ActorCriticNetwork, self).__init__()
        # 定义编码器模块，将原始的局部状态映射为一个向量。
        # 对于不同形式的状态，这一编码器可以有不同的结构。如对于图像输入状态，可以使用卷积神经网络 (CNN)；对于向量输入状态，可以使用多层感知机 (MLP)。
        # 在这里，我们使用了一个两层的 MLP 用来处理向量输入状态，即：
        # $$y = max(W_2 max(W_1 x+b_1, 0) + b_2, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        # 定义离散动作的输出网络，仅包含一个全连接层。
        self.policy_head = nn.Linear(64, action_shape)
        # 定义一个仅输出单个值的价值网络。
        self.value_head = nn.Linear(64, 1)

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> ttorch.Tensor:
        """
        **功能概述**:
            在离散动作空间中，Actor-Critic 网络的计算图。
        """
        # 将原始局部的观察状态转化为一个向量，形状变化：$$(B, A, *) -> (B, A, N)$$
        # 在 PyTorch 中，如 ``nn.Linear`` 的很多层，仅对张量的最后一个维度进行处理。因此，我们可以利用这种特性来处理整个多智能体的 batch。
        x = self.encoder(local_obs)
        # 计算每个可能的离散动作的 logit，形状变化：$$(B, A, N) -> (B, A, M)$$
        logit = self.policy_head(x)
        # 为每一个样本和数据计算价值，形状变化：$$(B, A, N) -> (B, A, 1)$$
        value = self.value_head(x)
        # 用 treetensor 的格式返回最终的计算结果。
        return ttorch.as_tensor({
            'logit': logit,
            'value': value,
        })


# delimiter
class SharedActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int) -> None:
        """
        **功能概述**:
            在多智能体场景下，使用策略梯度算法共享参数的 Actor-Critic 网络定义。
            由于各个智能体共享一个网络，因此它们的输入状态可以在一个 batch 中并行计算。
        """
        # PyTorch 在继承 ``nn.Module`` 类的时候，必须执行这个初始化方法。
        super(SharedActorCriticNetwork, self).__init__()
        # 输入的形状是： $$(B, A, O)$$.
        self.agent_num = agent_num
        # 定义一个所有智能体共享的 Actor-Critic 网络。
        self.actor_critic_network = ActorCriticNetwork(obs_shape, action_shape)

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> ttorch.Tensor:
        """
        **功能概述**:
            共享参数的 Actor-Critic 网络计算图。
            处理所有智能体的 ``local_obs``，并输出对应的策略分布和状态价值。
        """
        # 并行处理所有智能体的 ``local_obs``。
        return self.actor_critic_network(local_obs)


# delimiter
class IndependentActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_shape: int, action_shape: int) -> None:
        """
        **功能概述**:
            在多智能体场景下，使用策略梯度算法，各个智能体独立参数的 Actor-Critic 网络定义。
            各个智能体拥有自己独立的 Actor-Critic 网络，拥有独立的参数。
        """
        # PyTorch 在继承 ``nn.Module`` 类的时候，必须执行这个初始化方法。
        super(IndependentActorCriticNetwork, self).__init__()
        # 定义数量为 ``agent_num`` 的独立的 Actor-Critic 网络。每个智能体对应一个网络。
        # 为了利用 ``nn.Module`` 的一些特殊属性，我们使用 ``nn.ModuleList`` 作为作为存储这些网络的容器，而非 Python 自带的列表。
        self.agent_num = agent_num
        self.actor_critic_networks = nn.ModuleList(
            [ActorCriticNetwork(obs_shape, action_shape) for _ in range(agent_num)]
        )

    # delimiter
    def forward(self, local_obs: torch.Tensor) -> ttorch.Tensor:
        """
        **功能概述**:
            独立 Actor-Critic 网络的计算图。
            串行地处理各个智能体的 ``local_obs``，并输出对应的策略分布和状态价值。
        """
        # 切分数据，串行地调用网络逐一处理各个智能体的 ``local_obs``。
        return ttorch.cat([net(local_obs[:, i:i + 1]) for i, net in enumerate(self.actor_critic_networks)], dim=1)


# delimiter
class CTDEActorCriticNetwork(nn.Module):

    def __init__(self, agent_num: int, local_obs_shape: int, global_obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            在多智能体场景下，集中训练分散执行 (centralized training and decentralized execution, CTDE) 网络的定义。
            各个智能体共享同样的网络参数，因此它们可以在一个 batch 中并行地进行计算。
            价值网络的输入是 ``global_obs``，而策略网络的输入是 ``local_obs``。
            价值网络提取的全局信息，可以为局部的策略提供更多的指导；
            策略网络提取的局部信息，可以使得网络在分布式执行的时候，更具有鲁棒性。
        """
        # PyTorch 在继承 ``nn.Module`` 类的时候，必须执行这个初始化方法。
        super(CTDEActorCriticNetwork, self).__init__()
        # 分别定义局部编码器和全局编码器。
        self.agent_num = agent_num
        self.local_encoder = nn.Sequential(
            nn.Linear(local_obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_obs_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        # 定义离散动作的输出网络，仅包含一个全连接层。
        self.policy_head = nn.Linear(64, action_shape)
        # 定义一个仅输出单个值的价值网络。
        self.value_head = nn.Linear(64, 1)
        

    # delimiter
    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> ttorch.Tensor:
        """
        **Overview**:
            CTDE Actor-Critic 网络的计算图。
            并行地处理各个智能体的 ``local_obs`` 和 ``global_obs``，并输出对应的策略分布和状态价值。
            针对 ``global_obs`` 的设计，存在两种不同的方式：1) 对各个智能体采用一个共享的全局状态，即 ``global_obs`` 的形状为 $$(B, S)$$ 
            2) 对各个智能体设计不同的全局状态，即 ``global_obs`` 的形状为 $$(B, A, S')$$.
            关于这个问题的更多细节，可以参考链接 <link https://di-engine-docs.readthedocs.io/zh_CN/latest/04_best_practice/marl_zh.html#id10 link> 
        """
        # 用策略网络 (Actor) 处理局部的状态生成动作，用价值网络 (Critic) 处理全局状态生成价值。
        policy = self.policy_head(self.local_encoder(local_obs))
        value = self.value_head(self.global_encoder(global_obs))
        return ttorch.as_tensor({
            'logit': policy,
            'value': value,
        })


# delimiter
def test_shared_ac_network() -> None:
    """
    **test_shared_ac_network 功能概述**:
        用于测试共享参数的 Actor-Critic 网络。首先创建一个网络，并输入一个 batch 的数据。随后验证其输出各部分的形状。
    """
    # 设置 batch size，智能体个数，状态的形状和动作空间的维度。
    batch_size = 4
    agent_num = 3
    obs_shape = 10
    action_shape = 5
    # 定义一个共享参数的 Actor-Critic 网络。
    network = SharedActorCriticNetwork(agent_num, obs_shape, action_shape)
    # 随机生成伪数据，为各个智能体生成随机的状态。
    local_obs = torch.randn(batch_size, agent_num, obs_shape)
    # 前向计算过程，将局部状态输入网络，得到输出。
    result = network(local_obs)
    # 验证输出的形状。
    assert result['logit'].shape == (batch_size, agent_num, action_shape)
    assert result['value'].shape == (batch_size, agent_num, 1)


# delimiter
def test_independent_ac_network() -> None:
    """
    **test_independent_ac_network 功能概述**:
        用于测试独立参数 Actor-Critic 网络。首先创建一个网络，并输入一个 batch 的数据。随后验证其输出各部分的形状。
    """
    # 设置 batch size，智能体个数，状态的形状和动作空间的维度。
    batch_size = 4
    agent_num = 3
    obs_shape = 10
    action_shape = 5
    # 定义一个独立参数的 Actor-Critic 网络。
    network = IndependentActorCriticNetwork(agent_num, obs_shape, action_shape)
    # 随机生成伪数据，为各个智能体生成随机的状态。
    local_obs = torch.randn(batch_size, agent_num, obs_shape)
    # 前向计算过程，将局部状态输入网络，得到输出。
    result = network(local_obs)
    # 验证输出的形状。
    assert result['logit'].shape == (batch_size, agent_num, action_shape)
    assert result['value'].shape == (batch_size, agent_num, 1)


# delimiter
def test_ctde_ac_network() -> None:
    """
    **test_ctde_ac_network 功能概述**:
        用于测试 CTDE Actor-Critic 网络。首先创建一个网络，并输入一个 batch 的数据。随后验证其输出各部分的形状。
    """
    # 设置 batch size，智能体个数，状态的形状和动作空间的维度。
    batch_size = 4
    agent_num = 3
    local_obs_shape = 10
    global_obs_shape = 20
    action_shape = 5

    # 测试共享全局状态的情况。
    network = CTDEActorCriticNetwork(agent_num, local_obs_shape, global_obs_shape, action_shape)
    local_obs = torch.randn(batch_size, agent_num, local_obs_shape)
    global_obs = torch.randn(batch_size, global_obs_shape)
    result = network(local_obs, global_obs)

    # 验证输出的形状。
    assert result['logit'].shape == (batch_size, agent_num, action_shape)
    assert result['value'].shape == (batch_size, 1)

    # 测试不共享全局状态的情况。
    agent_specific_global_obs_shape = 25
    network = CTDEActorCriticNetwork(agent_num, local_obs_shape, agent_specific_global_obs_shape, action_shape)
    local_obs = torch.randn(batch_size, agent_num, local_obs_shape)
    agent_specific_global_obs = torch.randn(batch_size, agent_num, agent_specific_global_obs_shape)
    result = network(local_obs, agent_specific_global_obs)

    # 验证输出的形状。
    assert result['logit'].shape == (batch_size, agent_num, action_shape)
    assert result['value'].shape == (batch_size, agent_num, 1)


if __name__ == "__main__":
    test_shared_ac_network()
    test_independent_ac_network()
    test_ctde_ac_network()
