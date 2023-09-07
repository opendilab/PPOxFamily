"""
以下是关于独立策略梯度算法 (Independent Policy Gradient) 的 PyTorch 实现教程，适用于多智能体协同场景，也是最简单的多智能体强化学习算法。
这个教程使用了在 ``marl_network`` 中定义的两种类型的网络和在 ``pg`` 中定义的损失函数。主函数描述了独立策略梯度算法的核心部分，使用了随机生成的伪数据。
关于多智能体协同强化学习的更多细节，可以在此处找到 <link https://github.com/opendilab/PPOxFamily/blob/main/chapter6_marl/chapter6_lecture.pdf link>。

这个教程主要由两部分组成，你可以按顺序逐一学习这些部分，或者直接跳到你感兴趣的部分：
  - 适用于多智能体强化学习的独立策略梯度算法
  - 适用于多智能体强化学习的独立 Actor-Critic 算法
"""
import torch
from marl_network import IndependentActorCriticNetwork
# 你需要复制 chapter1_overview 中关于 pg 的实现代码
from pg import pg_data, pg_error


def independentpg_training_opeator() -> None:
    """
    **independentpg_training_opeator 功能概述**:
        关于独立策略梯度算法训练过程的主函数。
        定义一些超参数，神经网络和优化器，然后生成的伪数据并计算策略梯度损失。
        最后，使用优化器更新网络参数。在实际应用中，这些伪数据应该被与环境交互得到的真实数据替换。
    """
    # 设置必要的超参数。
    batch_size, agent_num, local_state_dim, global_state_dim, action_dim = 4, 5, 10, 20, 6
    # Entropy bonus 的权重，有助于智能体进行探索。
    entropy_weight = 0.001
    # 对未来奖励的折扣因子
    discount_factor = 0.99
    # 根据运行环境设定，决定 tensor 放置于 cpu 或是 cuda 。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义多智能体神经网络和优化器。
    # 在这里，我们使用 ``IndependentActorCriticNetwork`` 作为示例，你也可使用共享参数的网络。
    model = IndependentActorCriticNetwork(agent_num, local_state_dim, action_dim)
    model.to(device)
    # Adam 是深度强化学习中最常用的优化器。如果你想使用 weight decay，你应当使用 ``torch.optim.AdamW`` 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义对应随机生成的伪数据，其格式与真实和环境交互得到的数据相同。
    # 要注意，数据和网络应当在相同的设备上 (cpu 或 cuda)。
    # 简单起见，我们这里假定一个 batch 的数据构成了一个完整的 episode。
    # 在真实实践中，一个训练 batch 可能是多个 episode 混在一起的结果。我们常常使用 ``done`` 这个变量来区分不同的 episodes。
    local_state = torch.randn(batch_size, agent_num, local_state_dim).to(device)
    action = torch.randint(0, action_dim, (batch_size, agent_num)).to(device)
    reward = torch.randn(batch_size, agent_num).to(device)
    # 对于最基础的策略梯度算法，return 值由带折扣因子的累积奖励计算得来。
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-Critic 网络前向计算.
    output = model(local_state)
    # 准备用于计算策略梯度损失函数的数据。
    data = pg_data(output.logit, action, return_)
    # 计算策略梯度算法的损失函数。
    loss = pg_error(data)
    # 策略损失函数部分和熵损失函数部分的加权和。
    # 注意，这里我们只使用了网络的“策略部分”（即 Actor 部分）来计算策略损失函数。
    # 如果你想要使用网络的“价值部分”（即 Critic 部分），你需要定义相应的价值损失函数并将其加入最终的总损失函数之中。
    total_loss = loss.policy_loss - entropy_weight * loss.entropy_loss

    # PyTorch 的反向传播及参数更新。
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print('independentpg_training_operator is ok')


# delimiter
def independentac_training_opeator() -> None:
    """
    **independentac_training_opeator 功能概述**:
        关于独立 Actor-Critic 算法的训练过程的主函数。
        定义一些超参数，神经网络和优化器，然后生成随机伪数据并计算相关损失函数。在实践中，训练数据应被替换为与环境互动获得的结果。
        最后，使用优化器更新网络参数。在本文中，网络的策略部分指的是 Actor，而价值部分网络则指的是 Critic。
    """
    # 设置必要的超参数。
    batch_size, agent_num, local_state_dim, global_state_dim, action_dim = 4, 5, 10, 20, 6
    # Entropy bonus 的权重，有助于智能体进行探索。
    entropy_weight = 0.001
    # 价值损失函数的权重，用于平衡损失函数值的大小。
    value_weight = 0.5
    # 对未来奖励的折扣因子
    discount_factor = 0.99
    # 根据运行环境设定，决定 tensor 放置于 cpu 或是 cuda 。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义多智能体神经网络和优化器。
    # 在这里，我们使用 ``IndependentActorCriticNetwork`` 作为示例，你也可使用共享参数的网络。
    model = IndependentActorCriticNetwork(agent_num, local_state_dim, action_dim)
    model.to(device)
    # Adam 是深度强化学习中最常用的优化器。如果你想使用 weight decay，你应当使用 ``torch.optim.AdamW`` 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义对应随机生成的伪数据，其格式与真实和环境交互得到的数据相同。
    # 要注意，数据和网络应当在相同的设备上 (cpu 或 cuda)。
    # 简单起见，我们这里假定一个 batch 的数据构成了一个完整的 episode。
    # 在真实实践中，一个训练 batch 可能是多个 episode 混在一起的结果。我们常常使用 ``done`` 这个变量来区分不同的 episodes。
    local_state = torch.randn(batch_size, agent_num, local_state_dim).to(device)
    action = torch.randint(0, action_dim, (batch_size, agent_num)).to(device)
    reward = torch.randn(batch_size, agent_num).to(device)
    # 累积回报值可以使用多种不同的方式进行计算，在这里我们使用由带折扣因子的累积奖励。
    # 你也可以使用 generalized advantage estimation (GAE), n-step 等其他方式计算该值。
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-Critic 网络前向计算。
    output = model(local_state)
    # ``squeeze`` 操作将张量 从 $$(B, A, 1)$$ 变化为 $$(B, A)$$
    value = output.value.squeeze(-1)
    # 准备用于计算策略梯度损失函数的数据。
    # ``detach`` 操作可以使得在计算损失函数的梯度时， ``value`` 的梯度不进行反向传播。
    data = pg_data(output.logit, action, value.detach())
    # 计算策略梯度损失函数。
    loss = pg_error(data)
    # 计算价值损失函数。
    value_loss = torch.nn.functional.mse_loss(value, return_)
    # 策略损失函数、价值损失函数、熵损失函数的加权和。
    total_loss = loss.policy_loss + value_weight * value_loss - entropy_weight * loss.entropy_loss

    # PyTorch 的反向传播及参数更新。
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print('independentac_training_operator is ok')


if __name__ == "__main__":
    independentpg_training_opeator()
    independentac_training_opeator()
