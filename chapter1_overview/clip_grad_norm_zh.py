"""
torch.nn.utils.clip_grad_norm 的 PyTorch 版实现。
"""
import torch
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    **概述**:
        torch.nn.utils.clip_grad_norm 的 PyTorch 版实现。<link https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_ link>
    """
    # 将可训练参数的非空梯度保存到列表 grads 中。
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    # 将 max_norm 和 norm_type 转换为 float 类型。
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grads[0].device
    # 梯度的最大范数（max norm）：$$\mathrm{total\_norm}^{\infty} = \max_{\theta_i\in \Theta} |\mathrm{grad}(\theta_i)|$$
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    # 梯度的 p-范数（p-norm）：$$\begin{split}\mathrm{total\_norm} &= (\sum_{\theta\in\Theta}((\sum_{\theta_i}\mathrm{grad}(\theta_i)^p)^\frac{1}{p})^p)^\frac{1}{p}\\&=(\sum_{\theta\in\Theta}(\sum_{\theta_i}\mathrm{grad}(\theta_i)^p))^\frac{1}{p}\end{split}$$
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    # 裁减系数（1e-6用于避免分母为零)：$$\mathrm{clip\_coef} = \frac{\mathrm{max\_norm}}{\mathrm{total\_norm}}$$
    clip_coef = max_norm / (total_norm + 1e-6)
    # 将系数的最大值固定为1
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    # 如果 total_norm < max_norm，torch.clamp 操作将 clip_coef 的最大值固定为1，所以 clip_coef_clamped = 1，这样 total_norm 将保持不变。
    # 如果 total_norm > max_norm，将对原来的梯度进行裁减，使得裁减后的梯度对应的 total_norm 的大小为 max_norm：$$\begin{split}\mathrm{total\_norm'}&=(\sum_{\theta\in\Theta}(\sum_{\theta}(\mathrm{grad}(\theta_i)\cdot\frac{\mathrm{max\_norm}}{\mathrm{total\_norm}})^p))^{\frac{1}{p}}\\&=\frac{(\sum_{\theta\in\Theta}(\sum_{\theta}\mathrm{grad}(\theta_i)^p))^{\frac{1}{p}}}{\mathrm{total\_norm}}\cdot\mathrm{max\_norm}\\&=\mathrm{max\_norm}\end{split}$$
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm


# delimiter
def test_clip_grad_norm():
    """
    **概述**:
        梯度正则化的测试函数。
    """
    # 设置相关参数：batch size=4, action=32
    B, N = 4, 32
    # 从随机分布中生成测试数据：logit，label。
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # 计算损失和梯度
    loss = torch.nn.MSELoss()
    output = loss(logit, label)
    output.backward()
    # 根据梯度的 total_norm 对梯度进行裁减：
    # 如果其 total_norm 超过 max_norm，则裁减并使得裁减后的梯度对应的 total_norm 的大小为 max_norm，否则不裁减。
    clip_grad_norm(logit, 0.5, 2)
    # 测试裁减后的 total_norm 的大小是否在预期范围内
    assert isinstance(logit.grad, torch.Tensor)
    grads = logit.grad
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    assert total_norm < 0.5


if __name__ == '__main__':
    test_clip_grad_norm()
