import numpy as np
import torch
from torch import Tensor


def data_generate(batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    随机生成范围在[0, 999]内的整数，返回三种训练格式以及标签（0为偶1为奇）
    """
    x1 = torch.randint(1000, size=(batch_size, 1), dtype=torch.float)
    x2 = [ten_to_binary(int(x1[i, 0])) for i in range(batch_size)]
    x2 = torch.tensor(x2, dtype=torch.float)
    x3 = torch.cos(x1 * torch.pi)
    y = x1 % 2      # 0为偶1为奇
    y = torch.reshape(y, (-1,)).long()
    return x1, x2, x3, y


def ten_to_binary(num: int) -> list:
    """
    :param num: 0~999的整数
    :return: 长度为10的list
    """
    assert 0 <= num <= 999
    s = np.binary_repr(num, 10)
    result = []
    for i in s:
        result.append(int(i))
    return result
