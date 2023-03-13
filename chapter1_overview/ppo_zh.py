"""
``Proximal Policy Optimization (PPO)`` 算法的 PyTorch 版实现。

PPO 是强化学习中最常用的算法之一，它结合了 Actor-Critic 方法和信赖域策略优化方法（Trust Region Policy Optimization）。
对于策略部分，PPO 通过结合裁剪过的优化目标和悲观界（pessimistic bound）来更新策略。对于价值函数部分，PPO 通常使用经典的时间差分方法（例如GAE）。
最终目标函数形式化定义为:
$$\min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$

本文档主要包括:
- PPO error 的实现。
- 主函数（测试函数）
"""
from typing import Optional, Tuple
from collections import namedtuple
import torch
import numpy as np

ppo_policy_data = namedtuple('ppo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight'])
ppo_policy_loss = namedtuple('ppo_policy_loss', ['policy_loss', 'entropy_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])


def ppo_policy_error(data: namedtuple,
                     clip_ratio: float = 0.2,
                     dual_clip: Optional[float] = None) -> Tuple[namedtuple, namedtuple]:
    """
    **概述**:
        ``Proximal Policy Optimization (PPO) <link https://arxiv.org/pdf/1707.06347.pdf link>`` 算法的 PyTorch 版实现。包含 entropy bounus, value_clip 和 dual_clip 功能。
    """
    # 对数据 data 进行解包: $$<\pi_{new}(a|s), \pi_{old}(a|s), a, A^{\pi_{old}}(s, a), w>$$
    logit_new, logit_old, action, adv, weight = data
    # 准备默认的权重（weight）。
    if weight is None:
        weight = torch.ones_like(adv)
    # 根据 logit 构建策略分布，然后得到对应动作的概率的对数值。
    dist_new = torch.distributions.categorical.Categorical(logits=logit_new)
    dist_old = torch.distributions.categorical.Categorical(logits=logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    # 熵奖赏（bonus）损失函数: $$\frac 1 N \sum_{n=1}^{N} \sum_{a^n}\pi_{new}(a^n|s^n) log(\pi_{new}(a^n|s^n))$$
    # 注意：最终的损失函数是 ``policy_loss - entropy_weight * entropy_loss`` .
    dist_new_entropy = dist_new.entropy()
    entropy_loss = (dist_new_entropy * weight).mean()
    # 重要性采样的权重: $$r(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$
    ratio = torch.exp(logp_new - logp_old)
    # 原始的代理目标: $$r(\theta) A^{\pi_{old}}(s, a)$$
    surr1 = ratio * adv
    # <b>裁剪后的代理目标:</b> $$clip(r(\theta), 1-\epsilon, 1+\epsilon) A^{\pi_{old}}(s, a)$$
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    # 论文 <link https://arxiv.org/abs/1912.09729 link> 中提出的双重裁剪目标（Dual clip）
    # 只有当 adv < 0 时才使用 Dual clip
    if dual_clip is not None:
        clip1 = torch.min(surr1, surr2)
        clip2 = torch.max(clip1, dual_clip * adv)
        policy_loss = -(torch.where(adv < 0, clip2, clip1) * weight).mean()
    # PPO-Clipped 损失: $$min(r(\theta) A^{\pi_{old}}(s, a), clip(r(\theta), 1-\epsilon, 1+\epsilon) A^{\pi_{old}}(s, a))$$
    # 在样本的维度乘以权重，然后在 batch 的维度执行求均值操作。
    else:
        policy_loss = (-torch.min(surr1, surr2) * weight).mean()
    # 添加一些可视化指标来监控优化状态，故使用关闭梯度计算的上下文。
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # 返回最终的损失函数和相关统计信息。
    return ppo_policy_loss(policy_loss, entropy_loss), ppo_info(approx_kl, clipfrac)


# delimiter
def test_ppo(clip_ratio, dual_clip):
    """
    **概述**:
        PPO 算法的测试函数，包括前向和反向传播过程。
    """
    # 设置相关参数：batch size=4, action=32
    B, N = 4, 32
    # 从随机分布中生成测试数据：logit_new, logit_old, action, adv.
    logit_new = torch.randn(B, N).requires_grad_(True)
    logit_old = logit_new + torch.rand_like(logit_new) * 0.1
    action = torch.randint(0, N, size=(B, ))
    adv = torch.rand(B)
    data = ppo_policy_data(logit_new, logit_old, action, adv, None)
    # 计算 PPO error。
    loss, info = ppo_policy_error(data, clip_ratio=clip_ratio, dual_clip=dual_clip)
    # 测试 loss 是否是可微分的，是否能正确产生梯度
    assert all([np.isscalar(i) for i in info])
    assert logit_new.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)


if __name__ == '__main__':
    test_ppo(0.2, 0.5)
