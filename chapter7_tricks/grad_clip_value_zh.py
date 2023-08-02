"""
``torch.nn.utils.grad_clip_value`` 的 PyTorch 实现。。
"""
from typing import Union, Iterable
import torch

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def grad_clip_value(parameters: _tensor_or_tensors, clip_value: float) -> None:
    """
    **概述**：
    实现了 grad_clip_value 函数 <link https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_value_ link>。
    该函数在 loss 反向传播后使用，它会将网络参数的所有梯度剪裁 (clip) 到一个固定范围 [-clip_value, clip_value] 之间。
    注意这个函数是原地操作，修改梯度并没有任何返回值。
    """
    # 将可训练参数的非空梯度保存到列表中。
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    # 将原始 clip_value 转换为float。
    clip_value = float(clip_value)
    # 将梯度原地剪裁到 [-clip_value, Clip_value]。
    for grad in grads:
        grad.data.clamp_(min=-clip_value, max=clip_value)


# delimiter
def test_grad_clip_value():
    """
    **Overview**:
        Test function of grad clip with a fixed value.
    """
    # 准备超参数, batch size=4, action=32
    B, N = 4, 32
    # 设置 clip_value 为 1e-3
    clip_value = 1e-3
    # Generate regression logit and label, in practice, logit is the output of the whole network and requires gradient.
    # 生成回归的 logit 值和标签，在实际应用中， logit 值是整个网络的输出，并需要梯度计算。
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # 定义标准并计算 loss。
    criterion = torch.nn.MSELoss()
    output = criterion(logit, label)
    # 进行 loss 的反向传播并计算梯度。
    output.backward()
    # 使用固定值对梯度进行剪裁（clip）。
    grad_clip_value(logit, clip_value)
    # 在剪裁后，断言（assert）剪裁后的梯度值是否合理。
    assert isinstance(logit.grad, torch.Tensor)
    for g in logit.grad:
        assert (g <= clip_value).all()
        assert (g >= -clip_value).all()


if __name__ == '__main__':
    test_grad_clip_value()
