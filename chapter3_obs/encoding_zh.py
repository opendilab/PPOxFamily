"""
在本示例中，我们提供了一种利用 ``torch.Tensor._scatter`` ，将张量转化为 one-hot 编码形式的方案。
同时，我们还提供了代码例子，通过使用 ``torch.nn.Embedding`` ， 将张量转化为 one-hot 编码形式和二进制编码形式。
具体到 ``torch.nn.Embedding`` 的工作机制，这个模块的权重是一个 M x N 的矩阵，其中 M 是所有可能输入的数量（即单词表的长度），N 是 embedding 之后输出向量的维度。
它的工作方式就是，输入一个序号 i（此序号小于单词表的长度），输出权重矩阵的第 i 行（维度为 N 的向量）。
本文档主要由以下三部分组成：
- 使用 ``torch.Tensor._scatter`` 实现 one-hot 编码
- 使用 ``torch.nn.Embedding`` 实现 one-hot 编码
- 使用 ``torch.nn.Embedding`` 实现二进制编码
"""
import torch
import torch.nn as nn


def one_hot(val: torch.LongTensor, num: int) -> torch.FloatTensor:
    """
    **one_hot 函数功能概述**:
        将类型为 ``torch.LongTensor`` 的张量转化为其 one-hot 编码的形式。
        此实现的执行效率略高于 ``torch.nn.functional.one_hot`` 。
    """
    # 保存原始 ``val`` 的形状。
    old_shape = val.shape
    # 将 ``val`` 改变形状至二维张量。
    val_reshape = val.reshape(-1, 1)
    # 初始化结果张量，确定其形状，并设置和 val 在相同的 device 上。
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # 根据 ``val_reshape`` 中的值，将若干 1 填入结果张量中。注意，这一步是 in-place 操作（即直接原地改变结果张量的值）。
    ret.scatter_(1, val_reshape, 1)
    # 恢复原始形状，并将结果张量返回。
    return ret.reshape(*old_shape, num)


# delimiter
def get_one_hot_encoding(num: int):
    """
    **get_one_hot_encoding 函数功能概述**:
        使用 ``torch.nn.Embedding`` 实现 one-hot 编码。
    """
    # 权重矩阵应当设置为大小为 num x num 的单位矩阵。这样对于第 i 行，其内容是只有第 i 维是 1，其它维度都是 0 的向量，恰好就是 one-hot 编码。同时冻结参数，确保权重矩阵不可改变。
    return nn.Embedding.from_pretrained(torch.eye(num), freeze=True, padding_idx=None)


# delimiter
def get_binary_encoding(bit_num: int):
    """
    **get_binary_encoding 函数功能概述**:
        使用 ``torch.nn.Embedding`` 实现二进制编码。
    """
    # 生成形状为 $$2^{B} \times B $$ 的矩阵，其中 B 是比特数。
    # 矩阵的第 i 行代表了数字 i 的二进制表达，是一个维度为 B 的向量。
    location_embedding = []
    for n in range(2 ** bit_num):
        s = '0' * (bit_num - len(bin(n)[2:])) + bin(n)[2:]
        location_embedding.append(list(int(i) for i in s))
    mat = torch.FloatTensor(location_embedding)
    # 使用生成的矩阵作为 embedding 的权重，同时冻结参数确保权重矩阵不可改变。
    return torch.nn.Embedding.from_pretrained(mat, freeze=True, padding_idx=None)


# delimiter
def test_encoding():
    """
    **test_encoding 函数功能概述**:
        编码函数的主函数。对上述的若干种编码函数进行测试，检查输出的正确性。
    """
    # 测试上述两种 one-hot 编码方法，判断它们的输出结果是否一致。
    x = torch.LongTensor([9, 0, 1, 2, 1, 3, 5])
    one_hot_enc = get_one_hot_encoding(10)
    y = one_hot_enc(x)
    y_ = one_hot(x, num=10)
    assert torch.sum(torch.abs(y - y_)) < 1e-6
    # 测试二进制编码，判断其输出是否等于期望的结果。
    bin_enc = get_binary_encoding(2)
    x = torch.arange(4)
    y = bin_enc(x)
    ground_truth = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert torch.eq(y, ground_truth).all()


if __name__ == "__main__":
    test_encoding()
