"""
PyTorch implementation of torch.nn.utils.clip_grad_value
"""
import torch
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_value(parameters: _tensor_or_tensors, clip_value: float) -> None:
    """
    **Overview**:
        Implementation of clip_grad_value <link https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_value_ link>
    """
    # Save the non-empty gradient of trainable parameters into a list.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    # Convert clip_value to float.
    clip_value = float(clip_value)
    # Clip the gradient to [-clip_value, clip_value]
    for grad in grads:
        grad.data.clamp_(min=-clip_value, max=clip_value)


# delimiter
def test_clip_grad_value():
    """
    **Overview**:
        Test function of grad clip.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Set clip_value as 1e-3
    clip_value = 1e-3
    # Generate logit and label.
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # Compute loss and gradient
    loss = torch.nn.MSELoss()
    output = loss(logit, label)
    output.backward()
    # Clip the gradient
    clip_grad_value(logit, clip_value)
    # Assert the value of the clipped gradient
    assert isinstance(logit.grad, torch.Tensor)
    for g in logit.grad:
        assert (g <= clip_value).all()
        assert (g >= -clip_value).all()


if __name__ == '__main__':
    test_clip_grad_value()
