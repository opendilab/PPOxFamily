"""
PyTorch implementation of torch.nn.utils.clip_grad_norm
"""
import torch
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    **Overview**:
        Implementation of clip_grad_norm <link https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_ link>
    """
    # Save the non-empty gradient of trainable parameters into a list.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    # Convert max_norm and norm_type to float.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grads[0].device
    # The max norm of gradient: $$\mathrm{total\_norm}^{\infty} = \max_{\theta_i\in \Theta} |\mathrm{grad}(\theta_i)|$$
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    # The p-norm of gradient: $$\begin{split}\mathrm{total\_norm} &= (\sum_{\theta\in\Theta}((\sum_{\theta_i}\mathrm{grad}(\theta_i)^p)^\frac{1}{p})^p)^\frac{1}{p}\\&=(\sum_{\theta\in\Theta}(\sum_{\theta_i}\mathrm{grad}(\theta_i)^p))^\frac{1}{p}\end{split}$$
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    # The clip coefficient (the 1e-6 is used to avoid zero in the denominator): $$\mathrm{clip\_coef} = \frac{\mathrm{max\_norm}}{\mathrm{total\_norm}}$$
    clip_coef = max_norm / (total_norm + 1e-6)
    # Clamp the coefficient to 1
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    # If total_norm < max_norm, clip_coef will be clamped to 1, then the total_norm remains the same.
    # If total_norm > max_norm, the total_norm is clipped to max_norm: $$\begin{split}\mathrm{total\_norm'}&=(\sum_{\theta\in\Theta}(\sum_{\theta}(\mathrm{grad}(\theta_i)\cdot\frac{\mathrm{max\_norm}}{\mathrm{total\_norm}})^p))^{\frac{1}{p}}\\&=\frac{(\sum_{\theta\in\Theta}(\sum_{\theta}\mathrm{grad}(\theta_i)^p))^{\frac{1}{p}}}{\mathrm{total\_norm}}\cdot\mathrm{max\_norm}\\&=\mathrm{max\_norm}\end{split}$$
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm


# delimiter
def test_clip_grad_norm():
    """
    **Overview**:
        Test function of grad norm.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Generate logit and label.
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # Compute loss and gradient
    loss = torch.nn.MSELoss()
    output = loss(logit, label)
    output.backward()
    # Clip the gradient total_norm
    clip_grad_norm(logit, 0.5, 2)
    # Assert the total_norm of the clipped gradient
    assert isinstance(logit.grad, torch.Tensor)
    grads = logit.grad
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    assert total_norm < 0.5


if __name__ == '__main__':
    test_clip_grad_norm()
