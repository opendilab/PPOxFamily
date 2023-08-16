"""
通常在强化学习训练中，我们需要对神经网络的某些预测结果（例如价值函数）应用归一化函数，以增强强化学习的训练过程。
在本文档中，我们将演示两种数据归一化方法及其对应的逆操作。
- 第一种是 ``value_transform``，它可以减小动作值函数的尺度。其对应的逆操作是 ``value_inv_transform``。 <link https://arxiv.org/pdf/1805.11593.pdf link>
- 第二种是 ``symlog``，这是另一种归一化输入张量的方法。其对应的逆操作是 ``inv_symlog``。 <link https://arxiv.org/pdf/2301.04104.pdf link>
"""
import torch


def value_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    **概述**:
        用于减小动作值函数范围的函数。有关详细内容，请参阅论文: Achieving Consistent Performance on Atari <link https://arxiv.org/abs/1805.11593 link>
        给定输入张量 ``x``，此函数将返回归一化的张量。
        参数 ``eps`` 是一个超参数，用于控制添加的正则化项，以确保相应的逆操作是利普希茨连续的。
    """
    # 核心实现。
    # 归一化的公式是: $$h(x) = sign(x)(\sqrt{(|x|+1)} - 1) + \epsilon * x$$
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


# delimiter
def value_inv_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    **概述**:
        value_transform 的逆操作。给定输入张量 ``x``，此函数将返回反归一化的张量。
    """
    # 反归一化的公式是: $$h^{-1}(x) = sign(x)({(\frac{\sqrt{1+4\epsilon(|x|+1+\epsilon)}-1}{2\epsilon})}^2-1)$$
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


# delimiter
def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    **概述**:
        用于归一化目标的函数。有关详细内容，请参阅论文: Mastering Diverse Domains through World Models <link https://arxiv.org/abs/2301.04104 link>
        给定输入张量 ``x``，此函数将返回归一化的张量。
    """
    # 归一化的公式是: $$symlog(x) = sign(x)(\ln{|x|+1})$$
    return torch.sign(x) * (torch.log(torch.abs(x) + 1))


# delimiter
def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    **概述**:
        symlog 的逆操作。给定输入张量 ``x``，此函数将返回反归一化的张量。
    """
    # 反归一化的公式是: $$symexp(x) = sign(x)(\exp{|x|}-1)$$
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# delimiter
def test_value_transform():
    """
    **概述**:
        生成测试数据并测试 ``value_transform`` 和 ``value_inv_transform`` 函数。
    """
    # 生成测试数据。
    test_x = torch.randn(10)
    # 对生成的数据进行归一化。
    normalized_x = value_transform(test_x)
    assert normalized_x.shape == (10,)
    # 对数据进行反归一化。
    unnormalized_x = value_inv_transform(normalized_x)
    # 测试变换前后数据是否相同。
    assert torch.sum(torch.abs(test_x - unnormalized_x)) < 1e-3


# delimiter
def test_symlog():
    """
    **概述**:
        生成测试数据并测试 ``symlog`` 和 ``inv_symlog`` 函数。
    """
    # 生成测试数据。
    test_x = torch.randn(10)
    # 对生成的数据进行归一化。
    normalized_x = symlog(test_x)
    assert normalized_x.shape == (10,)
    # 对数据进行反归一化。
    unnormalized_x = inv_symlog(normalized_x)
    # 测试变换前后数据是否相同。
    assert torch.sum(torch.abs(test_x - unnormalized_x)) < 1e-3


if __name__ == '__main__':
    test_value_transform()
    test_symlog()
