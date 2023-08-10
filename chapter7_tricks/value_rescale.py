"""
**Overview:**
Typically, we need to apply normalization functions in RL training to reduce the scale of some predictions of neural networks (e.g. value function) to enhance the RL training process.
In this document, we will demonstrate two kinds of data normalization methods and their corresponding inverse operations.
- The first one is ``value_transform``, which can reduce the scale of the action-value function. Its corresponding inverse operation is ``value_inv_transform``.
- The second one is ``symlog``, which is another approach to normalize the input tensor. Its corresponding inverse operation is ``inv_symlog``.
"""
import torch


def value_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    **Overview:**
        A function to reduce the scale of the action-value function. For extensive reading, please refer to: Achieving Consistent Performance on Atari <link https://arxiv.org/abs/1805.11593 link>
        Given the input tensor ``x``, this function will return the normalized tensor.
        The argument ``eps`` is a hyper-parameter that controls the additive regularization term to ensure the corresponding inverse operation is Lipschitz continuous.
        The formula of the normalization is: $$h(x) = sign(x)(\sqrt{(abs(x)+1)} - 1) + \eps * x$$
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


# delimiter
def value_inv_transform(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    **Overview:**
        The inverse form of value transform. Given the input tensor ``x``, this function will return the unnormalized tensor.
        The formula of the unnormalization is: $$`h^{-1}(x) = sign(x)({(\frac{\sqrt{1+4\eps(|x|+1+\eps)}-1}{2\eps})}^2-1)$$
    """
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


# delimiter
def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    **Overview:**
        A function to normalize the targets. For extensive reading, please refer to: Mastering Diverse Domains through World Models <link https://arxiv.org/abs/2301.04104 link>
        Given the input tensor ``x``, this function will return the normalized tensor.
        The formula of the normalization is: $$symlog(x) = sign(x)(\ln{|x|+1})$$
    """
    return torch.sign(x) * (torch.log(torch.abs(x) + 1))


# delimiter
def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    """
    **Overview:**
        The inverse form of symlog. Given the input tensor ``x``, this function will return the unnormalized tensor.
        The formula of the unnormalization is: $$symexp(x) = sign(x)(\exp{|x|}-1)$$
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# delimiter
def test_value_transform():
    """
    **Overview:**
        Generate fake data and test the ``value_transform`` ``value_inv_transform`` functions.
    """
    # Generate fake data.
    test_x = torch.randn(10)
    # Normalize the generated data.
    normalized_x = value_transform(test_x)
    assert normalized_x.shape == (10,)
    # Unnormalize the data.
    unnormalized_x = value_inv_transform(normalized_x)
    # Test whether the data before and after the transformation is the same.
    assert torch.sum(torch.abs(test_x - unnormalized_x)) < 1e-3


# delimiter
def test_symlog():
    """
    **Overview:**
        Generate fake data and test the ``symlog`` ``inv_symlog`` functions.
    """
    # Generate fake data.
    test_x = torch.randn(10)
    # Normalize the generated data.
    normalized_x = symlog(test_x)
    assert normalized_x.shape == (10,)
    # Unnormalize the data.
    unnormalized_x = inv_symlog(normalized_x)
    # Test whether the data before and after the transformation is the same.
    assert torch.sum(torch.abs(test_x - unnormalized_x)) < 1e-3


if __name__ == '__main__':
    test_value_transform()
    test_symlog()
