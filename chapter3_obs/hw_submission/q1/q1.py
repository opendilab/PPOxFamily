import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(1)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, use_cos=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.use_cos = use_cos
        self.relu_layer = nn.ReLU()

    def forward(self, x):
        if self.use_cos:
            hidden = torch.cos(self.fc1(x))
        else:
            hidden = self.relu_layer(self.fc1(x))
        y_pred = self.fc2(hidden)
        return y_pred


def decimal2binary(x):
    # 将样本转为二进制编码
    x = bin(int(x)).replace("0b", "")  # '1000100101'
    x = '0' * (10 - len(x)) + x  # 不足10个二进制位的样本，高位用0补齐
    # [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    x = list(map(float, x))
    return x

def main_loop(x_train, y_train_true, x_test, y_test_true, input_size, output_size, use_cos):
    model = MLP(input_size, output_size, use_cos)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    loss_list = []
    for epoch in range(1000):
        y_train_pred = model(x_train)
        loss = loss_func(y_train_pred, y_train_true)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch = {epoch}, Training Loss = {loss.item()}")

    plt.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.savefig("results.png")
    plt.close()

    with torch.no_grad():
        y_test_pred = model(x_test)
    accuracy = torch.count_nonzero(y_test_pred.argmax(dim=-1) == y_test_true.argmax(dim=-1)).item() / len(y_test_true)
    print(f"Test Accuracy = {accuracy}")

def approach_1(x_train, y_train_true, x_test, y_test_true):
    y_train_true = F.one_hot(y_train_true.to(torch.int64)).float().squeeze(1)
    y_test_true = F.one_hot(y_test_true.to(torch.int64)).float().squeeze(1)
    main_loop(x_train,y_train_true,x_test,y_test_true,1,2,False)


def approach_2(x_train, y_train_true, x_test, y_test_true):
    x_train_binary = torch.tensor(np.array(list(map(decimal2binary, x_train)))).float()
    x_test_binary = torch.tensor(np.array(list(map(decimal2binary, x_test)))).float()
    y_train_true = F.one_hot(y_train_true.to(torch.int64)).float().squeeze(1)
    y_test_true = F.one_hot(y_test_true.to(torch.int64)).float().squeeze(1)
    main_loop(x_train_binary,y_train_true,x_test_binary,y_test_true,10,2,False)


def approach_3(x_train,y_train_true,x_test,y_test_true):
    main_loop(x_train,y_train_true,x_test,y_test_true,1,1,True)


if __name__ == "__main__":
    # 载入训练集、测试集
    x_train = torch.randint(0, 1000, size=(10000, 1)).float()
    y_train_true = x_train % 2
    x_test = torch.randint(0, 1000, size=(100, 1)).float()
    y_test_true = x_test % 2
    

   
    # 选择一个方案测试效果
    # approach_1(x_train,y_train_true,x_test,y_test_true)
    # approach_2(x_train, y_train_true,x_test, y_test_true)
    approach_3(x_train,y_train_true,x_test,y_test_true)
