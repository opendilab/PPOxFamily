"""
PyTorch implementation of ``torch.nn.utils.grad_clip_value`` .
"""
from typing import Union, Iterable
import torch

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def grad_clip_value(parameters: _tensor_or_tensors, clip_value: float) -> None:
    """
    **Overview**:
        Implementation of grad_clip_value <link https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_value_ link>
        This function is used after the loss backpropagation, clip all the gradient of network parameters
        with a fixed range [-clip_value, clip_value].
        BTW, This function is a in-place operation, modify the gradient without any return value.
    """
    # Save the non-empty gradient of trainable parameters into a list.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    # Convert the original clip_value to float.
    clip_value = float(clip_value)
    # Clip the gradient to [-clip_value, clip_value] in-place.
    for grad in grads:
        grad.data.clamp_(min=-clip_value, max=clip_value)


# delimiter
def test_grad_clip_value():
    """
    **Overview**:
        Test function of grad clip with a fixed value.
    """
    # Prepare hyper-parameters, batch size=4, action=32
    B, N = 4, 32
    # Set clip_value as 1e-3
    clip_value = 1e-3
    # Generate regression logit and label, in practice, logit is the output of the whole network and requires gradient.
    logit = torch.randn(B, N).requires_grad_(True)
    label = torch.randn(B, N)
    # Define criterion, and compute loss.
    criterion = torch.nn.MSELoss()
    output = criterion(logit, label)
    # Loss backward and compute gradient.
    output.backward()
    # Clip the gradient with a fixed value.
    grad_clip_value(logit, clip_value)
    # Assert that the value of the clipped gradient is reasonable after clipping.
    assert isinstance(logit.grad, torch.Tensor)
    for g in logit.grad:
        assert (g <= clip_value).all()
        assert (g >= -clip_value).all()


if __name__ == '__main__':
    test_grad_clip_value()
