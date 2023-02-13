"""
策略梯度（Policy Gradient，PG）算法的 PyTorch 实现。

策略梯度（也常指代 REINFORCE 算法）是学习策略的一类经典方法。
每个 $$(s_t，a_t)$$ 将用于计算相应的对数概率，然后概率被反向传播计算得到梯度，梯度会乘以一个权重值，这个权重值通常是这局游戏中的累计收益。
最终的目标函数形式化定义为:
$$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$

本文档主要包括:
- PG error 的实现。
- 主函数（测试函数）
"""
from collections import namedtuple
import torch

pg_data = namedtuple('pg_data', ['logit', 'action', 'return_'])
pg_loss = namedtuple('pg_loss', ['policy_loss', 'entropy_loss'])


def pg_error(data: namedtuple) -> namedtuple:
    """
    **概述**:
        策略梯度（Policy Gradient，PG）算法的 PyTorch 实现。
    """
    # 对数据 data 进行解包: $$<\pi(a|s), a, G_t>$$
    logit, action, return_ = data
    # 根据 logit 构建策略分布，然后得到对应动作的概率的对数值。
    dist = torch.distributions.categorical.Categorical(logits=logit)
    log_prob = dist.log_prob(action)
    # 策略损失函数: $$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
    policy_loss = -(log_prob * return_).mean()
    # 熵奖赏（bonus）损失函数：$$\frac 1 N \sum_{n=1}^{N} \sum_{a^n}\pi(a^n|s^n) log(\pi(a^n|s^n))$$
    # 注意：最终的损失是 ``policy_loss - entropy_weight * entropy_loss`` .
    entropy_loss = dist.entropy().mean()
    # 返回最终的各项损失函数：包含策略损失和熵损失。
    return pg_loss(policy_loss, entropy_loss)


# delimiter
def test_pg():
    """
    **概述**:
        策略梯度算法的测试函数，包括前向和反向传播测试。
    """
    # 设置相关参数：batch size=4, action=32
    B, N = 4, 32
    # 从随机分布中生成测试数据：logit, action, return_.
    logit = torch.randn(B, N).requires_grad_(True)
    action = torch.randint(0, N, size=(B, ))
    return_ = torch.randn(B) * 2
    # 计算 PG error。
    data = pg_data(logit, action, return_)
    loss = pg_error(data)
    # 测试 loss 是否是可微分的，是否能正确产生梯度
    assert all([l.shape == tuple() for l in loss])
    assert logit.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit.grad, torch.Tensor)


if __name__ == '__main__':
    test_pg()
