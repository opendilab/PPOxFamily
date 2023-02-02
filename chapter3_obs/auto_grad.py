"""
This document mainly includes:
- Numpy version to calculate gradients manually.
- Pytorch version to calculate gradients automatically.
The example function to calculate gradient is formulated as:
$$ c = \sum x * y + z $$
"""
import numpy as np
import torch.nn as nn
import torch
np.random.seed(0)
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
x = nn.Parameter(torch.from_numpy(x))
y = nn.Parameter(torch.from_numpy(y))
z = nn.Parameter(torch.from_numpy(z))
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
