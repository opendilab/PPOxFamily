import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




def one_hot(val: torch.LongTensor, num: int) -> torch.FloatTensor:
    """
    **Overview**:
        Convert a ``torch.LongTensor`` to one hot encoding with scatter API.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot`` .
    """
    # Remember original shape of val.
    old_shape = val.shape
    # Reshape val into 2D tensor.
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
    for n in range(2**bit_num):
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
    bin_enc = get_binary_encoding(4)
    x = torch.arange(10)
    y = bin_enc(x)
    ground_truth = torch.LongTensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    print(y)
    print((1%100)%10)
    print(torch.randn(10, 2))
    assert torch.eq(y, ground_truth).all()

class Parity(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.n1 = nn.Linear(in_size, hidden_size)
        self.n2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.n1(x)
        x = self.relu(x)
        x = self.n2(x)
        out = self.sigmoid(x)
        return out

train_data = np.arange(0, 700, 1)
test_data = np.arange(700, 1000, 1)
bin_enc = get_binary_encoding(10)
Parity1 = Parity(1,32,1)
Parity2 = Parity(10,32,1)
Parity3 = Parity(1,32,1)
pi = torch.pi

def train_1():
    loss_plt1 = []
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(Parity1.parameters(), lr=0.01)
    trainX = torch.tensor(train_data).float().reshape(-1, 1)
    trainY = torch.tensor(train_data%2).float().reshape(-1, 1)
    Parity1.train()
    for epoch in range(100):
        pred = Parity1(trainX)
        acc = (pred.round() == trainY).float().mean()
        loss = loss_func(pred, trainY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_plt1.append(loss.item())
        if epoch % 10 == 0:
            print("epoch:[%4d] loss:%.4f accuracy:%.4f" %(epoch, loss.item(), acc))

    plt.plot(loss_plt1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.close()

def test_1():
    Parity1.eval()
    testX = torch.tensor(test_data).float().reshape(-1, 1)
    pred = Parity1(testX)
    testY = torch.tensor(test_data%2).float().reshape(-1, 1)

    acc = (pred.round() == testY).float().mean()
    return acc

def train_2():
    loss_plt2 = []
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(Parity2.parameters(), lr=0.01)
    trainX = torch.tensor(train_data)
    trainX = bin_enc(trainX)
    trainY = torch.tensor(train_data%2).float().reshape(-1, 1)
    Parity2.train()
    for epoch in range(100):
        pred = Parity2(trainX)
        acc = (pred.round() == trainY).float().mean()
        loss = loss_func(pred, trainY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_plt2.append(loss.item())
        if epoch % 10 == 0:
            print("epoch:[%4d] loss:%.4f accuracy:%.4f" %(epoch, loss.item(), acc))

    plt.plot(loss_plt2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.close()

def test_2():
    Parity2.eval()
    testX = torch.tensor(test_data)
    testX = bin_enc(testX)
    pred = Parity2(testX)
    testY = torch.tensor(test_data%2).float().reshape(-1, 1)
    acc = (pred.round() == testY).float().mean()
    return acc

def train_3():
    loss_plt3 = []
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(Parity3.parameters(), lr=0.01)
    trainX = torch.tensor(train_data)
    trainX = np.sin(np.pi/2*(2*trainX-1)).reshape(-1, 1)
    trainY = torch.tensor(train_data%2).float().reshape(-1, 1)
    Parity3.train()
    for epoch in range(100):
        pred = Parity3(trainX)
        acc = (pred.round() == trainY).float().mean()
        loss = loss_func(pred, trainY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_plt3.append(loss.item())
        if epoch % 10 == 0:
            print("epoch:[%4d] loss:%.4f accuracy:%.4f" %(epoch, loss.item(), acc))
    
    plt.plot(loss_plt3)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.close()

def test_3():
    Parity3.eval()
    testX = torch.tensor(test_data)
    testX = np.sin(np.pi/2*(2*testX-1)).reshape(-1, 1)
    pred = Parity3(testX)
    testY = torch.tensor(test_data%2).float().reshape(-1, 1)
    acc = (pred.round() == testY).float().mean()
    return acc


def main():
    train_1()
    train_2()
    train_3()
    print("test_1_acc:%.4f", test_1().item())
    print("test_2_acc:%.4f", test_2().item())
    print("test_3_acc:%.4f", test_3().item())


if __name__ == "__main__":
    main()
