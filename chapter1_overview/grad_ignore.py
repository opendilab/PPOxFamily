"""
DI-engine implementation of grad_ignore_norm and grad_ignore_value
"""
import torch
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def grad_ignore_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    **Overview**:
        Implementation of grad_ignore_norm <link https://github.com/opendilab/DI-engine/blob/2ab7c44a64329fb90fa877e6070bc76bb6fdb31e/ding/torch_utils/optimizer_helper.py#L56 link>
        Different from clip_grad_norm, grad_ignore_norm **ignore**: those gradients that have a norm exceeds the specified threshold, instead of cliping their norm to the threshold.
    """
    # Save the parameters with non-empty gradient into a list.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    # Convert max_norm and norm_type to float.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    # The max norm of gradient: $$\mathrm{total\_norm}^{\infty} = \max_{\theta_i\in \Theta} |\mathrm{grad}(\theta_i)|$$
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    # The p-norm of gradient: $$\begin{split}\mathrm{total\_norm} &= (\sum_{\theta\in\Theta}((\sum_{\theta_i}\mathrm{grad}(\theta_i)^p)^\frac{1}{p})^p)^\frac{1}{p}\\&=(\sum_{\theta\in\Theta}(\sum_{\theta_i}\mathrm{grad}(\theta_i)^p))^\frac{1}{p}\end{split}$$
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    # The clip coefficient (the 1e-6 is used to avoid zero in the denominator): $$\mathrm{clip\_coef} = \frac{\mathrm{max\_norm}}{\mathrm{total\_norm}}$$
    clip_coef = max_norm / (total_norm + 1e-6)
    # If total_norm > max_norm, all the gradient is clipped to zero.
    if clip_coef < 1:
        for p in parameters:
            p.grad.zero_()
    return total_norm


def grad_ignore_value(parameters: _tensor_or_tensors, clip_value: float) -> None:
    """
    **Overview**:
        Implementation of grad_ignore_value <link https://github.com/opendilab/DI-engine/blob/2ab7c44a64329fb90fa877e6070bc76bb6fdb31e/ding/torch_utils/optimizer_helper.py#L77 link>
        Different from clip_grad_value, grad_ignore_value **ignore**: all the gradients when any of them exceeds the specified threshold, instead of cliping them to the threshold.
    """
    # Save the parameters with non-empty gradient into a list.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    # Convert clip_value to float.
    clip_value = float(clip_value)
    flag = False
    # Check if there is any gradient that exceeds the clip_value.
    for p in parameters:
        val = p.grad.data.abs().max()
        if val >= clip_value:
            flag = True
            break
    # If there exists a gradient that exceeds the clip_value, then clip all the gradients to zero.
    if flag:
        for p in parameters:
            p.grad.data.zero_()


# delimiter
def test_grad_ignore_norm():
    """
    **Overview**:
        Test function of grad ignore norm.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Generate logit and label.
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # Compute loss and gradient.
    loss = torch.nn.MSELoss()
    output = loss(logit, label)
    output.backward()
    # Set a gradient that exceeds the threshold.
    logit.grad[0] = 0.5
    # Clip the gradient.
    grad_ignore_norm(logit, 0.5, 2)
    # Assert that all gradients are clipped to zero.
    assert isinstance(logit.grad, torch.Tensor)
    for g in logit.grad:
        assert (g == 0).all()


# delimiter
def test_grad_ignore_value():
    """
    **Overview**:
        Test function of grad ignore clip.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Set clip_value as 0.5.
    clip_value = 0.5
    # Generate logit and label.
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # Compute loss and gradient.
    loss = torch.nn.MSELoss()
    output = loss(logit, label)
    output.backward()
    # Set a gradient that exceeds the threshold.
    logit.grad[0] = 0.6
    # Clip the gradient
    grad_ignore_value(logit, clip_value)
    # Assert that all gradients are clipped to zero.
    assert isinstance(logit.grad, torch.Tensor)
    for g in logit.grad:
        assert (g == 0).all()


if __name__ == '__main__':
    test_grad_ignore_norm()
    test_grad_ignore_value()
