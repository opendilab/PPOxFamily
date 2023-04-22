"""
We provide a method to convert a tensor to one hot encoding, using ``torch.Tensor._scatter`` .
Also, we provide examples using ``torch.nn.Embedding`` to implement one-hot encoding and binary encoding.
Internally, weight in ``torch.nn.Embedding`` is a M x N matrix, with M being the number of words and N being the size of each word vector.
It matches a word index to the corresponding embedding vector, i.e., the corresponding row in the matrix.
This document mainly includes:
- One-hot encoding implementation using ``torch.Tensor._scatter`` .
- One-hot encoding implementation using ``torch.nn.Embedding`` .
- Binary encoding implementation using ``torch.nn.Embedding`` .
"""
import torch
import torch.nn as nn


def one_hot(val: torch.LongTensor, num: int) -> torch.FloatTensor:
    """
    **Overview**:
        Convert a ``torch.LongTensor`` to one hot encoding with scatter API.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot`` .
    """
    # Remember original shape of ``val`` .
    old_shape = val.shape
    # Reshape ``val`` into 2D tensor.
    val_reshape = val.reshape(-1, 1)
    # Initialize return tensor with float32 dtype and the same device as val.
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # Fill value 1 into tensor ``ret`` , according to the index stored in ``val_reshape`` . It is an inplace operation.
    ret.scatter_(1, val_reshape, 1)
    # Return the reshaped result with the same prefix shape as original shape of val.
    return ret.reshape(*old_shape, num)


# delimiter
def get_one_hot_encoding(num: int):
    """
    **Overview**:
        Implementation of one hot encoding with nn.Embedding API.
    """
    # Use the identity matrix as weight tensor.
    # Use freezed embedding as fixed one-hot transformation.
    return nn.Embedding.from_pretrained(torch.eye(num), freeze=True, padding_idx=None)


# delimiter
def get_binary_encoding(bit_num: int):
    """
    **Overview**:
        Implementation of binary encoding with nn.Embedding API.
    """
    # Generate a matrix with shape $$2^{B} \times B $$ where B is the bit_num.
    # Each row with index n contains the binary representation of n.
    location_embedding = []
    for n in range(2 ** bit_num):
        s = '0' * (bit_num - len(bin(n)[2:])) + bin(n)[2:]
        location_embedding.append(list(int(i) for i in s))
    mat = torch.FloatTensor(location_embedding)
    # Use the generated result as transformation..
    return torch.nn.Embedding.from_pretrained(mat, freeze=True, padding_idx=None)


# delimiter
def test_encoding():
    """
    **Overview**:
        Test different encoding methods.
    """
    # Test one-hot encoding with nn.Embedding and scatter, compare two float32 dtype tensor.
    x = torch.LongTensor([9, 0, 1, 2, 1, 3, 5])
    one_hot_enc = get_one_hot_encoding(10)
    y = one_hot_enc(x)
    y_ = one_hot(x, num=10)
    assert torch.sum(torch.abs(y - y_)) < 1e-6
    # Test binary encoding, compare two int64 dtype tensor.
    bin_enc = get_binary_encoding(2)
    x = torch.arange(4)
    y = bin_enc(x)
    ground_truth = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert torch.eq(y, ground_truth).all()


if __name__ == "__main__":
    test_encoding()
