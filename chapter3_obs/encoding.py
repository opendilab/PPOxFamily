"""
We provide a method to convert a tensor to one hot encoding, using `torch.Tensor._scatter`.
Also, we provide examples using `torch.nn.Embedding` to implement one-hot encoding and binary encoding. 
Internally, weight in `torch.nn.Embedding` is a M x N matrix, with M being the number of words and N being the size of each word vector.
It matches a word index to the corresponding embedding vector, i.e., the corresponding row in the matrix.
This document mainly includes:
- One-hot encoding implementation using `torch.Tensor._scatter`.
- One-hot encoding implementation using `torch.nn.Embedding`.
- Binary encoding implementation using `torch.nn.Embedding`.
"""
import torch
import torch.nn as nn


def one_hot(val: torch.LongTensor, num: int, num_first: bool = False) -> torch.FloatTensor:
    """
    Overview:  
        Convert a ``torch.LongTensor`` to one hot encoding.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot`` 
    """
    # Check whether each augument is legal.
    assert (isinstance(val, torch.Tensor)), type(val)
    assert val.dtype == torch.long
    assert (len(val.shape) >= 1)
    # Remember original shape of val.
    old_shape = val.shape
    # Reshape val into 2D tensor.
    val_reshape = val.reshape(-1, 1)
    # Initialize return tensor.
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # To remember the location where the original value is -1 in val.
    # If the value is -1, then it should be converted to all zeros encodings and the corresponding entry in index_neg_one is 1, which is used to transform the ret after the operation of ret.scatter_(1, val_reshape, 1) to their correct encodings bellowing
    index_neg_one = torch.eq(val_reshape, -1).float()
    # if -1 exists in val
    if index_neg_one.sum() != 0:  
        # convert the original value -1 to 0
        val_reshape = torch.where(
            val_reshape != -1, val_reshape,
            torch.zeros(val_reshape.shape, device=val.device).long()
        )
    try:
        # Fill value 1 into tensor ret, according to the index stored in val_reshape. It is a inplace operation.
        ret.scatter_(1, val_reshape, 1)
        # if -1 exists in val
        if index_neg_one.sum() != 0:
             # change -1's encoding from [1,0,...,0] to [0,0,...,0]
            ret = ret * (1 - index_neg_one)
    # Deal with shape mismatch.
    except RuntimeError:
        raise RuntimeError('value: {}\nnum: {}\t:val_shape: {}\n'.format(val_reshape, num, val_reshape.shape))
    # Return the reshaped result.
    if num_first:
        return ret.permute(1, 0).reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)


def get_one_hot_encoding(num):
    """
    **Overview**:
        Implementation of one hot encoding.
    """
    # Use the identity matrix as weight tensor.
    return nn.Embedding.from_pretrained(torch.eye(num), freeze=True, padding_idx=None)


def get_binary_encoding(bit_num):
    """
    **Overview**:
        Implementation of binary encoding.
    """
    # Generate a matrix with shape 2^{bit_num} x bit_num. Each row with index n contains the binary representation of n.
    location_embedding = []
    for n in range(2**bit_num):
        s = '0' * (bit_num - len(bin(n)[2:])) + bin(n)[2:]
        location_embedding.append(list(int(i) for i in s))
    mat = torch.FloatTensor(location_embedding)
    # Use the generated function as weith tensor.
    return torch.nn.Embedding.from_pretrained(mat, freeze=True, padding_idx=None)


if __name__ == '__main__':
    # Test one-hot encoding.
    one_hot_enc = get_one_hot_encoding(10)
    x = torch.arange(10)
    y = one_hot_enc(x)
    assert torch.sum(torch.abs(y - torch.eye(10))) < 1e-6
    # Test another version of one-hot encoding.
    x = torch.arange(10)
    y = one_hot(x, num=10)
    assert torch.sum(torch.abs(y - torch.eye(10))) < 1e-6
    # Test binary encoding.
    bin_enc = get_binary_encoding(2)
    x = torch.arange(4)
    y = bin_enc(x)
    ground_truth = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    assert torch.sum(torch.abs(y - ground_truth)) < 1e-6
