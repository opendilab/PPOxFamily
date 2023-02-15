import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(name, input_size, output_size, x_train, y_train, x_test, y_test, epoch):
    # 构造模型
    model = NeuralNetwork(input_size, output_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    for epoch_num in range(epoch):
        # model.train()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch_num % 10 == 0:
            loss, current = loss.item(), epoch_num
            print(f"epoch: {current:>5d} loss: {loss:>7f}  ")

    draw(loss_history, name)

    # model.eval()
    with torch.no_grad():
        pred = model(x_test)
        test_loss = loss_fn(pred, y_test).item()
        accuracy = (pred.round() == y_test).sum() / len(y_test)
        # accuracy = torch.count_nonzero(pred.argmax(dim=-1) == y_test.argmax(dim=-1)).item() / len(y_test)
    print(f"Test \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.bitwise_and(mask).ne(0).float()


def draw(loss_list, name):
    plt.plot(loss_list, label=name)
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("q1_" + name + ".png")
    plt.close()


if __name__ == "__main__":
    bin_encode = get_binary_encoding(10)
    # 训练集
    x_train = torch.randint(0, 999, (3750, 1)).float()
    x_train_sin = torch.sin(x_train)
    x_train_bin = binary(x_train.long(), 10)
    y_train = x_train % 2
    # 测试集
    x_test = torch.randint(0, 999, (1250, 1)).float()
    x_test_sin = torch.sin(torch.pi * x_test)
    x_test_bin = binary(x_test.long(), 10)
    y_test = x_test % 2

    epochs = 2000

    # 方案1 直接输入数字，损失值振荡大，可能是数字分布过于离散导致
    train("int", 1, 1, x_train, y_train, x_test, y_test, epochs)
    # 方案2 二进制处理，收敛快，跟计算机的硬件是01二值计算有关
    train("bin", 10, 1, x_train_bin, y_train, x_test_bin, y_test, epochs)
    # 方案2 平滑处理，并将范化到-1，1，分布均匀
    train("sin", 1, 1, x_train_sin, y_train, x_test_sin, y_test, epochs)

    print("Done!")
