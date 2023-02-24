from typing import Dict
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from popart import PopArt

class MLPNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**：
            A MLP network used to fit the Q value.
        """
        super(MLPNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape+action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.popart = PopArt(32, 1)
        #self.decoder = nn.Linear(32, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, actions),1)
        x = self.encoder(x)
        normalized_output  = self.popart(x)
        return normalized_output 

def train(obs_shape: int, action_shape: int, train_data):
    model = MLPNetwork(obs_shape, action_shape)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    MSEloss = nn.MSELoss()
    train_data = DataLoader(train_data, batch_size = 64, shuffle = True)
    for epoch in range(2000):
        for idx, data in enumerate(train_data):
            optimizer.zero_grad()
            normalized_output = model(data['observations'], data['actions'])
            mu = model.popart.mu
            sigma = model.popart.sigma

            with torch.no_grad():
                normalized_reward = (data['rewards'] - mu)/sigma

            # print(normalized_output, normalized_reward, mu, sigma)

            loss = MSEloss(normalized_output, normalized_reward)
            # if epoch % 100 == 99:   
            #     print(normalized_output, normalized_reward)
            loss.backward()
            optimizer.step()

            model.popart.update_parameters(data['rewards'])

            running_loss = 0.0
            running_loss += loss.item()     # extract the loss value
        if epoch % 100 == 99:    
            # print every 1000 
            print('Epoch [%d] loss: %.6f' %
                    (epoch + 1, running_loss / 100))
            # zero the loss
            running_loss = 0.0
            
if __name__ == '__main__':
    f = open('processed_data.pkl', 'rb')
    dataset = pickle.load(f)
    train(obs_shape=8, action_shape=1, train_data=dataset)

    
