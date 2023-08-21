"""
This document provides the naive implementation of the Random Network Distillation (RND).

RND is a form of reward model that generates intrinsic rewards to encourage exploration behaviors of the agent.
The main principle is to calculate the discrepancy between a predictor network's prediction and
a randomly initialized and fixed target network's output as intrinsic rewards.

For more details, please refer to the paper:
    Burda Y, Edwards H, Storkey A, et al. Exploration by random network distillation[J]. 
    arXiv preprint arXiv:1810.12894, 2018.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple


class RndNetwork(nn.Module):
    """
    **Overview**:
        Random Network Distillation (RND) Network.
    """

    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super(RndNetwork, self).__init__()

        # Define the target and predictor networks as simple 2-layer fully connected networks
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    # delimiter
    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        **Overview**:
            The forward function of the RND network.
            It returns the target features and the predicted features.
        """
        with torch.no_grad():
            target_feature = self.target_network(obs)
        pred_feature = self.predictor_network(obs)
        return target_feature, pred_feature

    # delimiter
    def calculate_rnd_intrinsic_reward(self, obs: Tensor) -> Tensor:
        """
        **Overview**:
            Calculate the RND intrinsic reward.
        """
        target_feature, pred_feature = self.forward(obs)
        rnd_reward = (target_feature - pred_feature).pow(2).sum(dim=1)
        return rnd_reward


# delimiter
def compute_rnd_reward(model: RndNetwork, obs: Tensor) -> None:
    """
    **Overview**:
        Test function to calculate the RND intrinsic reward.
    """
    rnd_reward = model.calculate_rnd_intrinsic_reward(obs)
    print(f"RND intrinsic reward: {rnd_reward}")
    assert rnd_reward.shape == (obs.shape[0],)
    assert rnd_reward.dtype == torch.float32
    assert rnd_reward.min() >= 0.0


# delimiter
def train(model: RndNetwork, optimizer: optim.Optimizer, obs: Tensor, total_train_steps: int = 1000) -> float:
    """
    **Overview**:
        Training function for the RND model.
    """
    for _ in range(total_train_steps):
        # Forward pass
        target_feature, pred_feature = model(obs)

        # Compute the loss as the MSE between the target and predicted features
        loss = ((target_feature - pred_feature) ** 2).mean()

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# delimiter
def eval(model: RndNetwork, obs: Tensor) -> float:
    """
     **Overview**:
        Testing function for the RND model.
    """
    with torch.no_grad():
        # Forward pass
        target_feature, pred_feature = model(obs)

        # Compute the loss as the MSE between the target and predicted features
        loss = ((target_feature - pred_feature) ** 2).mean()

    return loss.item()


if __name__ == '__main__':
    # Define the dimensions
    obs_dim = 10
    hidden_dim = 20

    # Initialize the model and optimizer
    model = RndNetwork(obs_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters())

    # Generate some random data
    obs = torch.randn(100, obs_dim)

    # Train the model
    train_loss = train(model, optimizer, obs)
    print(f"Train loss: {train_loss}")

    # Test the model
    eval_loss = eval(model, obs)
    print(f"Eval loss: {eval_loss}")

    # Test RND intrinsic reward calculation
    compute_rnd_reward(model, obs)
