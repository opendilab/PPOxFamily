"""
本文档主要包括：
- 使用 PyTorch 定义一个可求导函数的方法
- 使用 Numpy 手动计算导数的方法
- 使用 PyTorch 自动计算导数的方法
本文档用于求导的函数示例是：
$$ c = \sum x * y + z $$
本文档还将介绍利用在 PyTorch 中自定义可导函数的方法。
通过继承 ``torch.autograd.Function`` <link https://pytorch.org/docs/stable/autograd.html?highlight=autograd+function#torch.autograd.Function link>，用户可以通过重写其中的前向传播、反向传播函数，自定义一个可导的函数
我们将以一个标准的线性函数为例进行介绍：
$$output = input \cdot weight^T + bias$$
"""
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Function
from copy import deepcopy


class LinearFunction(Function):
    """
    **LinearFunction 定义概述**:
        这是一个线性的可导函数，等价于神经网络中的线性层（全连接层）。公式是：
        $$output = input \cdot weight^T + bias$$
    """

    @staticmethod
    def forward(ctx, input_, weight, bias):
        """
        **forward 函数功能概述**:
            线性函数的前向传播计算过程。
        """
        # 保存参数，用于后续反向传播。
        ctx.save_for_backward(input_, weight)
        # 前向传播： $$output = input \cdot weight^T + bias$$
        output = input_.mm(weight.t())
        output += bias
        return output

    # delimiter
    @staticmethod
    def backward(ctx, grad_output):
        """
        **backward 函数功能概述**:
            线性函数的反向传播计算过程。
        """
        # 拿回在前向传播中保存的参数。
        input_, weight = ctx.saved_tensors
        # 初始化梯度为 None。这是因为并不是所有的参数都需要被求导，如果某参数无需被求导，其梯度应当返回 None。
        grad_input, grad_weight, grad_bias = None, None, None
        # 对输入 input 进行反向传播： $$ \nabla input = \nabla output \cdot weight $$
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # 对权重 weight 进行反向传播： $$ \nabla weight = \nabla output^T \cdot input $$
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input_)
        # 对权重 bias 进行反向传播： $$ \nabla bias = \sum \nabla output $$
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


# delimiter
def test_linear_function():
    """
    **test_linear_function 函数功能概述**:
        测试定义的线性函数，对前向传播结果，以及反向传播结果进行结果检查。
    """
    # 生成测试数据。
    w = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(1, 3, requires_grad=False)
    b = torch.randn(4, requires_grad=True)

    # 使用 PyTorch 内置方法完成线性计算。
    o = torch.sum(x @ w.t() + b)
    # 使用 PyTorch 内置的自动求导完成反向传播。
    o.backward()
    # 保留反向传播结果，用于后续结果检查。
    w_grad, b_grad = deepcopy(w.grad), deepcopy(b.grad)
    w.grad, x.grad, b.grad = None, None, None

    # 使用自定义的线性函数进行前向传播。
    linear_func = LinearFunction()
    o = torch.sum(linear_func.apply(x, w, b))
    # 反向传播。
    o.backward()

    # 对求导的结果进行正确性检查。
    assert x.grad is None
    assert torch.sum(torch.abs(w_grad - w.grad)) < 1e-6
    assert torch.sum(torch.abs(b_grad - b.grad)) < 1e-6


# delimiter
def test_auto_grad():
    """
    **test_auto_grad 函数功能概述**:
        测试自动求导的机制，对比用 Numpy 的手写求导与 PyTorch 的自动求导结果。
    """
    # 规定测试数据的格式。
    B, D = 3, 4
    # 生成 Numpy 版本的测试数据。
    x = np.random.randn(B, D)
    y = np.random.randn(B, D)
    z = np.random.randn(B, D)
    # Numpy 版本的前向传播。
    a = x * y
    b = a + z
    c = np.sum(b)
    # Numpy 版本的反向传播。
    grad_c = 1.0
    grad_b = grad_c * np.ones((B, D))
    grad_a = grad_b.copy()
    grad_z = grad_b.copy()
    grad_x = grad_a * y
    grad_y = grad_a * x
    # 将 Numpy 版本的测试数据转化为 PyTorch 版本。
    x = nn.Parameter(torch.from_numpy(x)).requires_grad_(True)
    y = nn.Parameter(torch.from_numpy(y)).requires_grad_(True)
    z = nn.Parameter(torch.from_numpy(z)).requires_grad_(True)
    # PyTorch 版本的前向传播。
    a = x * y
    b = a + z
    c = torch.sum(b)
    # PyTorch 版本的反向传播。
    c.backward()
    # 检查求导的结果是否一致。
    assert torch.sum(torch.abs(torch.from_numpy(grad_x) - x.grad)) < 1e-6
    assert torch.sum(torch.abs(torch.from_numpy(grad_y) - y.grad)) < 1e-6
    assert torch.sum(torch.abs(torch.from_numpy(grad_z) - z.grad)) < 1e-6


if __name__ == "__main__":
    test_auto_grad()
    test_linear_function()
