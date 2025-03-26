import torch
import torch.nn as nn
import math
torch.pi = math.pi
from encoding import get_binary_encoding
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm


X = np.array(list(range(1000)))
y = np.array([x % 2 for x in X])
binary_encoder = get_binary_encoding(10)
pi = torch.pi
Xs = [
    torch.tensor(X.reshape((-1, 1))).float(), 
    binary_encoder(torch.tensor(X)).float(), 
    torch.tensor(np.stack([np.sin(X * pi), np.cos(X * pi)], axis=1)).float(), 
]
y = torch.tensor(y).float()
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = \
    train_test_split(*Xs, y, test_size=0.1, random_state=666)


def train(model, X, y, lr=0.01, epochs=100):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, X, y):
    model.eval()
    pred = model(X).sigmoid()
    return accuracy_score(y, pred >= 0.5)


class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


acc1 = []
acc2 = []
acc3 = []
epochs = 500
for i in tqdm(range(10)):
    model1 = Model(1) ## original
    model2 = Model(10) ## binary
    model3 = Model(2) ## triangle
    train(model1, X1_train, y_train, epochs=epochs)
    train(model2, X2_train, y_train, epochs=epochs)
    train(model3, X3_train, y_train, epochs=epochs)
    acc1.append(test(model1, X1_test, y_test).item())
    acc2.append(test(model2, X2_test, y_test).item())
    acc3.append(test(model3, X3_test, y_test).item())
# 取10次准确率的均值
print('model1 avg acc: {}'.format(np.mean(acc1)))
print('model2 avg acc: {}'.format(np.mean(acc2)))
print('model3 avg acc: {}'.format(np.mean(acc3)))


# 输出如下：
# model1 avg acc: 0.441
# model2 avg acc: 0.9270000000000002
# model3 avg acc: 1.0


# 原始版本训练出的模型就是在胡猜
# 二进制编码后，可以预测正确，但收敛较慢，我这里的设置需要500个epochs以上才可以，可能是因为样本空间的维度比较高
# sin、cos编码，既能预测得准，收敛也较快，我这里的设置只需100个epochs以上基本就可达到100%的准确率，因为特征维度只有2，同时又有较强的区分能力