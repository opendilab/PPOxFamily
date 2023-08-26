"""
This document provides the naive PyTorch implementation of the Intrinsic Curiosity Model (ICM).

ICM is a form of reward model that generates intrinsic rewards to encourage exploration behaviors of the agent.
The main principle is to use the prediction error of the agent's own actions in a visual feature space learned
by a self-supervised inverse dynamics model.

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


# Define the ICMNetwork class which is a subclass of PyTorch's nn.Module class.
class ICMNetwork(nn.Module):
    """
    **Overview**:
        The definition of Intrinsic Curiosity Model (ICM) Network. ICM is a form of reward model that generates
        intrinsic rewards to encourage exploration behaviors of the agent. The main principle is to use the
        prediction error of the agent's own actions as the intrinsic reward signal.
        For more details, please refer to the paper: <link https://arxiv.org/pdf/1705.05363.pdf link>.
    """

    # The constructor for the ICMNetwork class. It takes in the observation dimension, action dimension, and the hidden layer dimension.
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        # Call the constructor of the parent class (nn.Module) to inherit its properties.
        super(ICMNetwork, self).__init__()

        # Define the feature model.
        # This is a simple 2-layer fully connected network that will be used to extract features from the state.
        self.feature_model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Define the inverse model.
        # This is a simple 2-layer fully connected network that will be used to predict the action given the current and next state features.
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Define the forward model.
        # This is a simple 2-layer fully connected network that will be used to predict the next state feature given the current state feature and action.
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Define a loss function for the forward model.
        # This will be used to compute the discrepancy between the real and predicted next state features.
        self.forward_mse = nn.MSELoss(reduction='none')

    # delimiter
    # Define the forward function. This function defines the forward pass of the network.
    def forward(self, state: Tensor, next_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        **Overview**:
            The forward function of the ICM network. It returns the real next state's embedded feature,
            the predicted next state's embedded feature, and the predicted action.
        Arguments:
            - state (:obj:`Tensor`): The input state tensor.
            - next_state (:obj:`Tensor`): The input next state tensor.
            - action (:obj:`Tensor`): The input action tensor.
        Returns:
            - next_state_feature (:obj:`Tensor`): The real next state's embedded feature.
            - pred_next_state_feature (:obj:`Tensor`): The predicted next state's embedded feature.
            - pred_action (:obj:`Tensor`): The predicted action.
        """
        # Compute the features for the current and next states.
        state_feature = self.feature_model(state)
        next_state_feature = self.feature_model(next_state)
        # Predict the action using the inverse model.
        pred_action = self.inverse_model(torch.cat([state_feature, next_state_feature], dim=-1))
        # Predict the next state feature using the forward model.
        pred_next_state_feature = self.forward_model(torch.cat([state_feature, action], dim=-1))
        # Return the real next state feature, the predicted next state feature, and the predicted action.
        return next_state_feature, pred_next_state_feature, pred_action

    # delimiter
    # Define a function to compute the ICM intrinsic reward.
    def compute_icm_intrinsic_reward(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """
        **Overview**:
            Given the current state, action, and next state, this function computes the ICM intrinsic reward
            based on the prediction error of the ICM model.
        Arguments:
            - states (:obj:`Tensor`): The input state tensor.
            - actions (:obj:`Tensor`): The input action tensor.
            - next_states (:obj:`Tensor`): The input next state tensor.
        Returns:
            - icm_reward (:obj:`Tensor`): The ICM intrinsic reward.
        """
        # Compute the real next state feature, the predicted next state feature, and the predicted action.
        real_next_state_feature, pred_next_state_feature, _ = self.forward(states, next_states, actions)
        # Compute the prediction error as the MSE loss between the real and predicted next state features.
        raw_icm_reward = self.forward_mse(real_next_state_feature, pred_next_state_feature).mean(dim=1)
        # Normalize the raw ICM reward to the [0, 1] range.
        icm_reward = (raw_icm_reward - raw_icm_reward.min()) / (raw_icm_reward.max() - raw_icm_reward.min() + 1e-8)
        # Return the normalized ICM reward.
        return icm_reward


# delimiter
# Define a function for training the ICM model.
def train(model: ICMNetwork, optimizer: optim.Optimizer, states: Tensor, actions: Tensor, next_states: Tensor, total_train_steps: int = 1000) -> float:
    """
    **Overview**:
        Training function for the ICM model.
    """
    for _ in range(total_train_steps):
        # Perform a forward pass through the network.
        real_next_state_feature, pred_next_state_feature, pred_action = model(states, next_states, actions)

        # Compute the forward loss as the MSE loss between the real and predicted next state features.
        forward_loss = ((real_next_state_feature - pred_next_state_feature) ** 2).mean()
        # Compute the inverse loss as the MSE loss between the real and predicted actions.
        inverse_loss = ((actions - pred_action) ** 2).mean()

        # Combine the forward and inverse losses into a total loss.
        loss = forward_loss + inverse_loss
        # Zero out the gradients before the backward pass.
        optimizer.zero_grad()
        # Perform a backward pass through the network.
        loss.backward()
        # Update the network parameters with a step of the optimizer.
        optimizer.step()

    # Return the final loss.
    return loss.item()


# delimiter
# Define a function for evaluating the ICM model.
def eval(model: ICMNetwork, states: Tensor, actions: Tensor, next_states: Tensor) -> float:
    """
    **Overview**:
        Evaluation function for the ICM model.
    """
    with torch.no_grad():
        # Perform a forward pass through the network.
        real_next_state_feature, pred_next_state_feature, pred_action = model(states, next_states, actions)

        # Compute the forward loss as the MSE loss between the real and predicted next state features.
        forward_loss = ((real_next_state_feature - pred_next_state_feature) ** 2).mean()
        # Compute the inverse loss as the MSE loss between the real and predicted actions.
        inverse_loss = ((actions - pred_action) ** 2).mean()

        # Combine the forward and inverse losses into a total loss.
        loss = forward_loss + inverse_loss

    # Return the total loss.
    return loss.item()


# delimiter
# Define a function that calculates the ICM intrinsic reward using the trained ICM model.
def compute_icm_reward(model: ICMNetwork, states: Tensor, actions: Tensor, next_states: Tensor) -> None:
    """
    **Overview**:
        A function that calculates the ICM intrinsic reward using the trained ICM model.
    """
    # Compute the normalized ICM reward.
    icm_reward = model.compute_icm_intrinsic_reward(states, actions, next_states)
    # Print the ICM intrinsic reward.
    print(f"ICM intrinsic reward: {icm_reward}")
    # Assert that the ICM reward has the correct shape, dtype, and value range.
    assert icm_reward.shape == (states.shape[0],)
    assert icm_reward.dtype == torch.float32
    assert icm_reward.max() <= 1.0 and icm_reward.min() >= 0.0


# delimiter
# Define a test function for the ICM (Intrinsic Curiosity Module) model.
def test_icm():
    """
    **Overview**:
        This function serves as an end-to-end testing routine for the Intrinsic Curiosity Module (ICM) model. The test includes the following steps:

        1. Initialization of the ICM model with specified dimensions for the states, actions, and hidden layers.
        2. Generation of synthetic data to simulate states, actions, and next states.
        3. Training of the ICM model using generated data and the Adam optimizer.
        4. Evaluation of the trained model's performance based on its prediction loss.
        5. Computation of the ICM intrinsic reward based on the model's prediction error.

        The function is intended for debugging and performance assessment of the ICM model in a controlled setting with synthetic data.
    """

    # Define the dimensions for the states, actions, and hidden layer.
    # These numbers correspond to the size of the input that the ICM model expects.
    obs_dim = 10
    action_dim = 5
    hidden_dim = 20

    # Initialize the ICM model and the optimizer.
    # The ICM model is used to predict the next state of the environment given the current state and action.
    # The optimizer is used to optimize the parameters of the model to minimize the prediction error.
    model = ICMNetwork(obs_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters())

    # Generate some random data to train the ICM model.
    # 'states' and 'next_states' represent the current and next states of the environment.
    # 'actions' represent the actions taken by the agent.
    states = torch.randn(100, obs_dim)
    next_states = torch.randn(100, obs_dim)
    actions = torch.randn(100, action_dim)

    # Train the model using the generated data.
    # The 'train' function updates the parameters of the model to minimize the prediction error.
    train_loss = train(model, optimizer, states, actions, next_states)
    print(f"Train loss: {train_loss}")

    # Evaluate the model using the same data.
    # The 'eval' function computes the prediction error without updating the model parameters.
    eval_loss = eval(model, states, actions, next_states)
    print(f"Eval loss: {eval_loss}")

    # Test the computation of the ICM intrinsic reward.
    # The 'compute_icm_reward' function computes the reward based on the prediction error of the model.
    compute_icm_reward(model, states, actions, next_states)