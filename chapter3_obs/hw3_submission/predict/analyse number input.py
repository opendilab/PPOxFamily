# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = np.arange(1000)
label = np.zeros((1000, 2))
for i, num in enumerate(data):
    label[i, num%2] = 1
data = torch.FloatTensor(data).reshape(1000, -1) % 10
label = torch.FloatTensor(label)

model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
            )
optimizer = optim.Adam(model.parameters(), lr=0.01)
lossF = nn.CrossEntropyLoss(reduction='mean')

index = np.arange(1000)
batch_size = 64
losses = []
for epoch in range(100):
    np.random.shuffle(index)
    for start in range(0, 1000, batch_size):
        end = start + batch_size
        mb_index = index[start:end]
        b_x = data[mb_index]
        b_label = label[mb_index]
        b_y = model(b_x)
        loss = lossF(b_y, b_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

losses = np.array(losses)
index = np.arange(1, losses.size+1)
plt.plot(index, losses)

