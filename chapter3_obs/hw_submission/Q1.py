import torch
from torch import nn
from utils import data_generate


batch_size = 64
steps = 1000

model1 = nn.Linear(1, 2)    # 直接将需要预测的数字输⼊神经⽹络
model2 = nn.Linear(10, 2)   # 先此数字转化为⼆进制编码，然后将所有的⼆进制位输⼊⽹络
model3 = nn.Linear(1, 2)    # 使⽤三⻆函数cos(xπ)对需要预测的数字进⾏处理，然后输⼊⽹络
models = [model1, model2, model3]
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam([param for model in models for param in model.parameters()])

correct, total = torch.zeros(3), 0
for step in range(steps):
    x1, x2, x3, y = data_generate(batch_size)
    X = [x1, x2, x3]
    loss_list = []
    total += batch_size
    for i, (model, x) in enumerate(zip(models, X)):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss_list.append(float(loss))
        loss.backward()
        opt.step()
        opt.zero_grad()
        y_pred = torch.argmax(y_pred, dim=-1).detach()
        correct[i] += torch.sum((y_pred == y).int())
    if (step + 1) % 100 == 0:
        print('step:', step+1)
        print('loss:', loss_list)
        print('acc:', correct / total)
        print()
        correct, total = torch.zeros(3), 0
