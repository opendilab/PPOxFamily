import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_data(train_rate=0.9, num=1000):
    data = [*range(num)]
    random.shuffle(data)
    train_data = data[:int(num*train_rate)]
    train_label = [1 if i%2==0 else 0 for i in train_data]
    test_data = data[int(num*train_rate):]
    test_label = [1 if i%2==0 else 0 for i in test_data]
    return train_data, train_label, test_data, test_label


def input_process(data):
    # transform data to 10 bits
    bin_data = bin(data)[2:]
    bin_data = '0'*(10-len(bin_data)) + bin_data
    bin_data = [int(i) for i in bin_data]
    return torch.tensor(bin_data).float()


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.numbers = x
        self.labels = y
        
    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, index):
        input_data = self.numbers[index]
        input_data = input_process(input_data)
        label = self.labels[index]
        label = torch.tensor(label)
        return input_data, label

class NumberClassifier(nn.Module):
    def __init__(self):
        super(NumberClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

train_data, train_label, test_data, test_label = generate_data()
train_dataset = NumberDataset(train_data, train_label)
test_dataset = NumberDataset(test_data, test_label)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = NumberClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
losses = []
accuracies = []

for epoch in range(epochs):
    running_loss = 0.0
    running_accuracy = 0.0

    model.train()    
    for inputs, labels in train_dataloader:
        # print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_accuracy += torch.sum(preds == labels.data)
    epoch_accuracy = running_accuracy.double() / len(test_dataloader.dataset)
    
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("exp2_loss.png")
plt.close()
plt.plot(accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("exp2_acc.png")
