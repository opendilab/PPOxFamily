# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2023/2/7 9:56 上午
# Description : 
"""
import torch
import torch.nn as nn
import random
import math
import torch.nn.functional as F


class Classify(nn.Module):
    def __init__(self, obs_shape, head_shape):
        super(Classify, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 16),
            nn.Tanh(),
            nn.Linear(16, head_shape),
        )


    def forward(self, x):

        x = self.encoder(x)
        pred = F.softmax(x)
        return pred




def get_train_data(batch_size = 100):

    X = []
    Y = []
    for i in range(batch_size):
        k = random.randint(0, 999)
        X.append([k])
        Y.append([k%2, 1-k%2])
    return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)


def conv2bin(_data):
    _data = _data.to(dtype=torch.uint8)
    mask = 2 ** torch.arange(9, -1, -1).to(_data.device, torch.uint8)
    bin_data = _data.bitwise_and(mask).ne(0).to(dtype=torch.float)
    return bin_data

def conv2cos(_data):
    cos_data = torch.cos(data_X * math.pi)
    return cos_data



if __name__ == "__main__":

    test_type = 3# 1表示直接预测,2表示二进制输入,3表示三角函数预测

    if test_type==1:
        classify = Classify(1, 2)
    elif test_type==2:
        classify = Classify(10, 2)
    else:
        classify = Classify(1, 2)

    episode = 1000
    batch_size = 1000
    criterion = torch.nn.MSELoss()
    lr = 0.005
    optim = torch.optim.Adam(classify.parameters(), lr=lr)
    train_loss = 0
    for i in range(episode):
        data_X, data_Y = get_train_data(batch_size)

        if test_type==2:
            data_X = conv2bin(data_X)
        elif test_type==3:
            data_X = conv2cos(data_X)
        optim.zero_grad()
        outputs = classify(data_X)

        loss = criterion(outputs, data_Y)
        loss.backward()
        optim.step()
        loss_tmp = loss.item()*batch_size
        train_loss += loss_tmp
        if i>0:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                i+1,
                loss_tmp
            ))

#结论
'''
方案一：
Epoch: 2 	Training Loss: 388.895333
Epoch: 500 	Training Loss: 249.649718
Epoch: 1000 	Training Loss: 250.191569

方案二：
Epoch: 2 	Training Loss: 215.550810
Epoch: 500 	Training Loss: 0.132932
Epoch: 1000 	Training Loss: 0.035490

方案三：
Epoch: 2 	Training Loss: 115.072586
Epoch: 500 	Training Loss: 0.058078
Epoch: 1000 	Training Loss: 0.016906


1、方案二和方案三都是可行的，都会收敛
2、方案三比方案二更快的收敛，输入网络数据也更少，因此实践当中方案三是最优的

'''


