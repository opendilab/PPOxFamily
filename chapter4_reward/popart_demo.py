"""
A demo for popart.py, implement a supervised learning model to fit the Q value of lunarlander.
The network is a MLP with popart as the final layer.
"""
from typing import Dict
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from popart import PopArt

class MLP(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            A MLP network with popart as the final layer.
            Input: observations and actions
            Output: Estimated Q value
            ``cat(obs,actions) -> encoder -> popart`` .
        """
        super(MLP, self).__init__()
        # Define the encoder and popart layer.
        # Here we use MLP with two layer and ReLU as activate function.
        # The final layer is popart.
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape+action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.popart = PopArt(32, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # The encoder first concatenate the observation vectors and actions,
        # then map the input to an embedding vector.
        x = torch.cat((obs, actions),1)
        x = self.encoder(x)
        # The popart layer maps the embedding vector to a normalized value.
        normalized_output  = self.popart(x)
        return normalized_output 


def train(obs_shape: int, action_shape: int, NUM_EPOCH: int, train_data):
    model = MLP(obs_shape, action_shape)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    MSEloss = nn.MSELoss()
    # Read the preprocessed data of trained agent on lunarlander.
    # Data format: list of dict with keys: ``{'observations', 'actions', 'rewards'}`` , 
    # where the rewards is the discounted return from the current state. 
    train_data = DataLoader(train_data, batch_size = 64, shuffle = True)

    for epoch in range(NUM_EPOCH):
        for idx, data in enumerate(train_data):
            optimizer.zero_grad()
            # Compute the normalized output and the 
            output, normalized_output = model(data['observations'], data['actions'])
            mu = model.popart.mu
            sigma = model.popart.sigma
            # Normalize the target return to align with the normalized Q value.
            with torch.no_grad():
                normalized_reward = (data['rewards'] - mu)/sigma
            # The loss is calculated as the MSE loss between normalized Q value and normalized target return.
            loss = MSEloss(normalized_output, normalized_reward)
            loss.backward()
            optimizer.step()
            # After the model parameters are updated with the gradient, the weights and bias should be updated to preserve unnormalised outputs.
            model.popart.update_parameters(data['rewards'])

            running_loss = 0.0
            running_loss += loss.item()

        if epoch % 100 == 99:    
            print('Epoch [%d] loss: %.6f' % (epoch + 1, running_loss / 100))
            running_loss = 0.0
            
if __name__ == '__main__':
    f = open('processed_data.pkl', 'rb')
    dataset = pickle.load(f)
    train(obs_shape=8, action_shape=1, NUM_EPOCH = 2000, train_data=dataset)

    