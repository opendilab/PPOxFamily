# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

class predict_model():
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 64
        self.epochs = 80
        
        self.net_num_input = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
            )
        self.net_binary_input = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
            ) 
        self.net_cos_input = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
            )
        
        self.optim_num = optim.Adam(self.net_num_input.parameters(), lr=self.lr)
        self.optim_binary = optim.Adam(self.net_binary_input.parameters(), lr=self.lr)
        self.optim_cos = optim.Adam(self.net_cos_input.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
           
    def predict(self, x, input_type):
        if input_type == 'num':
            with torch.no_grad():
                prob = self.net_num_input(x)
            return np.array(prob).argmax()
        if input_type == 'binary':
            with torch.no_grad():
                prob = self.net_binary_input(x)
            return np.array(prob).argmax()
        if input_type == 'cos':
            with torch.no_grad():
                prob = self.net_cos_input(x)
            return np.array(prob).argmax()
       
    def train(self, data, label, input_type):
        data_len = data.size()[0]
        index = np.arange(data_len)
        losses = []
        
        for epoch in range(self.epochs):
            np.random.shuffle(index)
            for start in range(0, data_len, self.batch_size):
                end = start + self.batch_size
                mb_index = index[start:end]
                b_x = data[mb_index]
                b_label = label[mb_index]
                
                if input_type == 'num':
                    b_y = self.net_num_input(b_x)
                    loss = self.loss(b_y, b_label)
                    self.optim_num.zero_grad()
                    loss.backward()
                    self.optim_num.step()
                
                if input_type == 'binary':
                    b_y = self.net_binary_input(b_x)
                    loss = self.loss(b_y, b_label)
                    self.optim_binary.zero_grad()
                    loss.backward()
                    self.optim_binary.step()
                    
                if input_type == 'cos':
                    b_y = self.net_cos_input(b_x)
                    loss = self.loss(b_y, b_label)
                    self.optim_cos.zero_grad()
                    loss.backward()
                    self.optim_cos.step()
                
                losses.append(float(loss))
                
        return losses

    
def eval_accuracy(model):
    index = np.arange(1000)
    np.random.shuffle(index)
    sample_n = 500
    record = np.zeros((4, sample_n)) 
    for i in range(sample_n):
        num = index[i]
        x_num = torch.FloatTensor([[num]])
        x_binary = torch.FloatTensor(bin_array(num, 10)).reshape(1, -1)
        x_cos = torch.FloatTensor([[np.cos(num*np.pi)]])
        
        y_num = model.predict(x_num, 'num')
        y_binary = model.predict(x_binary, 'binary')
        y_cos = model.predict(x_cos, 'cos')
        
        record[0, i] = (num%2) == y_num
        record[1, i] = (num%2) == y_binary
        record[3, i] = (num%2) == y_cos
        
    return record.mean(axis=1)
    

if __name__ == '__main__':
    # numpy data
    data = np.arange(0, 1000)
    label = np.zeros((1000, 2))
    for i, num in enumerate(data):
        label[i, num%2] = 1
    data_binary_encode = np.array([bin_array(num, 10) for num in data])
    data_cos_encode = np.array([[np.cos(num*np.pi)] for num in data])
    
    # convert to tensor data
    b_data = torch.FloatTensor(data).reshape(1000, -1)
    b_data_binary = torch.FloatTensor(data_binary_encode)
    b_data_cos = torch.FloatTensor(data_cos_encode)
    b_label = torch.FloatTensor(label)
    
    # train
    model = predict_model()
    loss_num = model.train(b_data, b_label, 'num')
    loss_binary = model.train(b_data_binary, b_label, 'binary')
    loss_cos = model.train(b_data_cos, b_label, 'cos')
    
    # plot train_step-loss
    loss_num = np.array(loss_num)
    loss_binary = np.array(loss_binary)
    loss_cos = np.array(loss_cos)
    index = np.arange(1, loss_num.size+1)
    plt.plot(index, loss_num)
    plt.plot(index, loss_binary)
    plt.plot(index, loss_cos)
    plt.legend(['num', 'binary', 'cos'])
    plt.show()
    
    # eval accuracy
    acc_num, acc_binary, acc_coscos, acc_cos = eval_accuracy(model)
    print('accuracy-num:', acc_num)
    print('accuracy-binary:', acc_binary)
    print('accuracy-cos:', acc_cos)

