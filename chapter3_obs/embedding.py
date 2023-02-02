"""
Using ``torch.nn.Embedding`` to implement different encoding methods. 
Internally, weight in ``nn.Embedding`` is a M x N matrix, with M being the number of words and N being the size of each word vector.
It matches a word index to the corresponding embedding vector, i.e., the corresponding row in the matrix.
This document mainly includes:
- One-hot encoding implementation.
- Binary encoding implementation.
"""
import torch
import torch.nn as nn


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
    mat = torch.tensor(location_embedding).float()
    # Use the generated function as weith tensor.
    torch.nn.Embedding.from_pretrained(mat, freeze=True, padding_idx=None)
