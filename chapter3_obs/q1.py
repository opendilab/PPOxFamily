import torch
import torch.nn as nn
from encoding import get_binary_encoding


def accuracy(pred, y):
    N = len(y)
    return (y == pred).sum() / N


def train(model, dataX, dataY, epochs=10):
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    model.train()
    for epoch in range(epochs):
        pred = model(dataX)
        loss = loss_func(pred, dataY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    

def test(model, dataX, dataY):
    model.eval()
    pred = model(dataX).sigmoid()
    predY = pred >= 0.5
    return (dataY == predY).sum() / len(dataX)


class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


train_data = [_ for _ in range(800)]
test_data = [_ for _ in range(800, 1000)]
binary_encoder = get_binary_encoding(10)

pi = torch.pi
trainX1 = torch.tensor(train_data)
trainX2 = binary_encoder(trainX1)
trainX3 = torch.stack(((trainX1 * pi).sin(), (trainX1 * pi).cos())).T
trainX1 = trainX1.float().reshape(-1, 1)
trainY = torch.tensor([_ % 2 for _ in range(800)]).reshape(-1, 1).float()

testX1 = torch.tensor(test_data)
testX2 = binary_encoder(testX1)
testX3 = torch.stack(((testX1 * pi).sin(), (testX1 * pi).cos())).T
testX1 = testX1.float().reshape(-1, 1)
testY = torch.tensor([_ % 2 for _ in range(800, 1000)]).reshape(-1, 1).float()


model1 = Model(1) ## original
model2 = Model(10) ## binary
model3 = Model(2) ## sin cos
train(model1, trainX1, trainY)
train(model2, trainX2, trainY)
train(model3, trainX3, trainY)
print('model1 accuracy: %.2f' % test(model1, testX1, testY).item())
print('model2 accuracy: %.2f' % test(model2, testX2, testY).item())
print('model3 accuracy: %.2f' % test(model3, testX3, testY).item())

# 方案二和方案三可行，方案三最优
# 方案三经过cos，sin编码后，奇数被编码成（-1，0），偶数被编码成（1，0），
# 样本空间大大减少，大大简化了奇偶预测的难度。