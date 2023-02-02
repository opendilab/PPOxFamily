"""
This document mainly includes the method to manually define a differentiable function.
By inheriting `torch.autograd.Function`, users can overwrite corresponding forward and backward methods to manually define a differentiable function.
We take a linear function as example, which is formulated as:
$$output = input \cdot weight^T + bias$$
"""
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
        # Calculate gradients for each parameter.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias

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
