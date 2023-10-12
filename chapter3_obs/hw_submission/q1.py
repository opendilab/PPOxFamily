"""
第二个方案，先把数字转化为二进制编码，然后将所有的二进制位输入网络，是可行的。
因为二进制编码的最后一位，其实包含了奇偶信息，所以网络可能很容易收敛。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sample_size = 1000
learning_rate = 1e-3
batch_size = 4
epochs = 5

class NeuralNetwork(nn.Module):

    def __init__(self, method=1):
        super(NeuralNetwork, self).__init__()
        if method == 1:
            input_size = 1
        else:
            input_size = 10
        self.encode = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2),        
        )
    
    def forward(self, x):
        logits = self.encode(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, x, y, method=1):
        self.method = method
        self.x, self.y = x, y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.method == 1:
            x, y = self.x[idx], self.y[idx]
        elif self.method == 2:
            x, y = torch.sin(self.x[idx]), self.y[idx]
        elif self.method == 3:
            x, y = torch.cos(self.x[idx]), self.y[idx]
        elif self.method == 4:
            x = bin(self.x[idx]).replace('0b', '')
            x = '0'*(10-len(x))+x
            x = torch.tensor([int(i) for i in x])
            y = self.y[idx]
        return x.float(), y

def train(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (xb, yb) in enumerate(dataloader):
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(xb)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            test_loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(1) == yb).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    x = torch.randint(low=0, high=999, size=(sample_size, 1))
    y = x % 2
    y = y.view(-1)
    whole_data = CustomDataset(x, y, method=4)
    train_data, test_data = random_split(whole_data, [800, 200])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = NeuralNetwork(method=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        train(model, train_dataloader, loss_fn, optimizer)
        test(model, test_dataloader, loss_fn)

if __name__ == '__main__':
    main()
    exit()
    
