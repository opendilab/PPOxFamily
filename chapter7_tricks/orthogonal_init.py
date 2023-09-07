"""
Official implementation of orthogonal initialization using PyTorch.
"""
import torch


def orthogonal_(tensor: torch.Tensor, gain: float = 1) -> torch.Tensor:
    """
    **Overview**:
        Fills the input ``Tensor`` with a (semi) orthogonal matrix, as described in this paper <link https://arxiv.org/pdf/1312.6120.pdf link>.
        The input tensor must have at least 2 dimensions, and for tensors with more than 2 dimensions the trailing dimensions are flattened.
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    # Initialize a new tensor with normal distribution. The shape is the same as the input tensor.
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)

    # If rows < cols, transpose the original tensor for computational efficiency.
    if rows < cols:
        flattened.t_()

    # Compute the QR factorization, Q is an orthogonal matrix and R is an upper triangular matrix.
    # <link https://en.wikipedia.org/wiki/QR_decomposition link>
    q, r = torch.linalg.qr(flattened)
    # Although Q is orthogonal, each value of Q is not uniformly distributed. To make Q uniform, we can use the equation below: $$Q^* = Q sign(diag(R))$$. Proof for this equation can be viewed in this paper: <link https://arxiv.org/pdf/math-ph/0609050.pdf link>.
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    # If rows < cols, transpose the output tensor to match the shape of original tensor.
    if rows < cols:
        q.t_()

    # Using ``torch.no_grad()`` here can make sure that these operations won't be added to the computational graph used by PyTorch's autograd system, thus improving efficiency.
    with torch.no_grad():
        # Reshape the result and copy the weight from q.
        tensor.view_as(q).copy_(q)
        # Multiply an optional scaling factor.
        tensor.mul_(gain)
    return tensor


# delimiter
def test_orthogonal() -> None:
    """
    **Overview**:
        Test the ``orthogonal_`` function. We use a weight tensor of convolutional layer and a weight tensor of linear layer as test cases, and check whether the results are correctly orthogonalized.
    """
    # For Conv. weights.
    w1 = torch.empty((4, 4, 3, 3))
    orthogonal_(w1)
    # Test whether the result is orthogonal.
    w1 = w1.reshape(w1.shape[0], -1).T
    res = w1.T @ w1
    gt = torch.eye(w1.shape[1])
    assert torch.sum((res - gt) ** 2).item() < 1e-9

    # For Linear weights.
    w2 = torch.empty((4, 4))
    orthogonal_(w2)
    # Test whether the result is orthogonal.
    res = w2.T @ w2
    gt = torch.eye(w2.shape[1])
    assert torch.sum((res - gt) ** 2).item() < 1e-9


if __name__ == "__main__":
    test_orthogonal()
