"""
This document mainly includes:
- Numpy version to calculate gradients manually.
- Pytorch version to calculate gradients automatically.
- Pytorch implementation to define a differentialble function.
The example function to calculate gradient is formulated as:
$$ c = \sum x * y + z $$
It also includes the method to manually define a differentiable function.
By inheriting `torch.autograd.Function` <link https://pytorch.org/docs/stable/autograd.html?highlight=autograd+function#torch.autograd.Function link>, users can overwrite corresponding forward and backward methods to manually define a differentiable function.
We take a linear function as example, which is formulated as:
$$output = input \cdot weight^T + bias$$
"""
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Function
from copy import deepcopy


class LinearFunction(Function):
    """
    Overview:
    Implementation of linear layer.
    """
    @staticmethod
    def forward(ctx, input, weight, bias):
        # Save parameters for backward.
        ctx.save_for_backward(input, weight)
        # Forward calculation: $$output = input \cdot weight^T + bias$$
        output = input.mm(weight.t())
        output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved parameters back.
        input, weight= ctx.saved_tensors
        # Initialize gradients to be None.
        grad_input, grad_weight, grad_bias = None, None, None
        # Calculate gradient for input: $$ \nabla input = \nabla output \cdot weight $$
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # Calculate gradient for weight: $$ \nabla weight = \nabla output^T \cdot input $$
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Calculate gradient for bias: $$ \nabla bias = \sum \nabla output $$
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


def test_linear_function():
    """
    Overview:
    Test linear function.
    """
    # Generate data.
    w = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(1, 3, requires_grad=False)
    b = torch.randn(4, requires_grad=True)

    # Forward.
    o = torch.sum(x @ w.t() + b)
    # Backward using auto grad.
    o.backward()
    # Save gradients for checking correctness.
    w_grad, x_grad, b_grad = deepcopy(w.grad), deepcopy(x.grad), deepcopy(b.grad)
    w.grad, x.grad, b.grad = None, None, None

    # Forward using our defined LinearFunction.
    linear_func = LinearFunction()
    o = torch.sum(linear_func.apply(x, w, b))
    # Backward.
    o.backward()

    # Check whether the results are correct.
    assert x.grad is None
    assert torch.sum(torch.abs(w_grad - w.grad)) < 1e-6
    assert torch.sum(torch.abs(b_grad - b.grad)) < 1e-6


def test_auto_grad():
    """
    Overview:
    Test auto grad.
    """
    # Generate data
    B, D = 3, 4
    # Numpy version.
    x = np.random.randn(B, D)
    y = np.random.randn(B, D)
    z = np.random.randn(B, D)
    # Forward.
    a = x * y
    b = a + z
    c = np.sum(b)
    # Backward.
    grad_c = 1.0
    grad_b = grad_c * np.ones((B, D))
    grad_a = grad_b.copy()
    grad_z = grad_b.copy()
    grad_x = grad_a * y
    grad_y = grad_a * x
    # PyTorch version.
    x = nn.Parameter(torch.from_numpy(x)).requires_grad_(True)
    y = nn.Parameter(torch.from_numpy(y)).requires_grad_(True)
    z = nn.Parameter(torch.from_numpy(z)).requires_grad_(True)
    # Forward.
    a = x * y
    b = a + z
    c = torch.sum(b)
    # Backward.
    c.backward()
    # Check whether the results are correct.
    assert torch.sum(torch.abs(torch.from_numpy(grad_x) - x.grad)) < 1e-6
    assert torch.sum(torch.abs(torch.from_numpy(grad_y) - y.grad)) < 1e-6
    assert torch.sum(torch.abs(torch.from_numpy(grad_z) - z.grad)) < 1e-6


if __name__ == '__main__':
    # Test auto grad.
    test_auto_grad()
    # Test linear function.
    test_linear_function()
