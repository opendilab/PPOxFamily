"""
Convert a tensor to one hot encoding, using ``torch._scatter().
"""
import torch


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