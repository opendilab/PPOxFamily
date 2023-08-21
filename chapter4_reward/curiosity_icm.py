"""
This document provides the naive implementation of the Intrinsic Curiosity Model (ICM).

ICM is a form of reward model that generates intrinsic rewards to encourage exploration behaviors of the agent.
The main principle is to use the prediction error of the agent's own actions as the intrinsic reward signal.

For more details, please refer to the paper:
    Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple


class ICMNetwork(nn.Module):
    """
    **Overview**:
        Intrinsic Curiosity Model (ICM) Network.
    """

    # Initialize the ICMNetwork with observation dimension, action dimension, and the hidden layer dimension
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(ICMNetwork, self).__init__()

        # Define the feature model as a simple 2-layer fully connected network
        self.feature_model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Define the inverse model as a simple 2-layer fully connected network
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Define the forward model as a simple 2-layer fully connected network
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Define the forward MSE loss function
        self.forward_mse = nn.MSELoss(reduction='none')

    # delimiter
    def forward(self, state: Tensor, next_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        **Overview**:
            The forward function of the ICM network
            It returns the real next state's embedded feature, the predicted next state's embedded feature,
            and the predicted action.
        """
        state_feature = self.feature_model(state)
        next_state_feature = self.feature_model(next_state)
        pred_action = self.inverse_model(torch.cat([state_feature, next_state_feature], dim=-1))
        pred_next_state_feature = self.forward_model(torch.cat([state_feature, action], dim=-1))
        return next_state_feature, pred_next_state_feature, pred_action

    # delimiter
    def calculate_icm_intrinsic_reward(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """
        **Overview**:
            Calculate the ICM intrinsic reward
        """
        real_next_state_feature, pred_next_state_feature, _ = self.forward(states, next_states, actions)
        raw_icm_reward = self.forward_mse(real_next_state_feature, pred_next_state_feature).mean(dim=1)
        icm_reward = (raw_icm_reward - raw_icm_reward.min()) / (raw_icm_reward.max() - raw_icm_reward.min() + 1e-8)
        return icm_reward


# delimiter
def compute_icm_reward(model: ICMNetwork, states: Tensor, actions: Tensor, next_states: Tensor) -> None:
    """
    **Overview**:
        A test function that calculates the ICM intrinsic reward.
    """
    icm_reward = model.calculate_icm_intrinsic_reward(states, actions, next_states)
    print(f"ICM intrinsic reward: {icm_reward}")
    assert icm_reward.shape == (states.shape[0],)
    assert icm_reward.dtype == torch.float32
    assert icm_reward.max() <= 1.0 and icm_reward.min() >= 0.0


# delimiter
def train(model: ICMNetwork, optimizer: optim.Optimizer, states: Tensor, actions: Tensor, next_states: Tensor, total_train_steps: int = 1000) -> float:
    """
    **Overview**:
        Training function for the ICM model.
    """
    for _ in range(total_train_steps):
        # Forward pass
        real_next_state_feature, pred_next_state_feature, pred_action = model(states, next_states, actions)

        # Compute the forward and inverse losses
        forward_loss = ((real_next_state_feature - pred_next_state_feature) ** 2).mean()
        inverse_loss = ((actions - pred_action) ** 2).mean()

        # Combine the losses and backpropagate
        loss = forward_loss + inverse_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# delimiter
def eval(model: ICMNetwork, states: Tensor, actions: Tensor, next_states: Tensor) -> float:
    """
    **Overview**:
        Evaluation function for the ICM model.
    """
    with torch.no_grad():
        # Forward pass
        real_next_state_feature, pred_next_state_feature, pred_action = model(states, next_states, actions)

        # Compute the forward and inverse losses
        forward_loss = ((real_next_state_feature - pred_next_state_feature) ** 2).mean()
        inverse_loss = ((actions - pred_action) ** 2).mean()

        # Combine the losses
        loss = forward_loss + inverse_loss

    return loss.item()


if __name__ == '__main__':
    # Define the dimensions
    obs_dim = 10
    action_dim = 5
    hidden_dim = 20

    # Initialize the model and optimizer
    model = ICMNetwork(obs_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters())

    # Generate some random data
    states = torch.randn(100, obs_dim)
    next_states = torch.randn(100, obs_dim)
    actions = torch.randn(100, action_dim)

    # Train the model
    train_loss = train(model, optimizer, states, actions, next_states)
    print(f"Train loss: {train_loss}")

    # Test the model
    eval_loss = eval(model, states, actions, next_states)
    print(f"Eval loss: {eval_loss}")

    # Test ICM intrinsic reward calculation
    compute_icm_reward(model, states, actions, next_states)
