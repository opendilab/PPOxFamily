"""
This document provides the naive PyTorch implementation of the Random Network Distillation (RND).

RND is a form of reward model that generates intrinsic rewards to encourage exploration behaviors of the agent.
The main principle is to calculate the discrepancy between a predictor network's prediction and
a randomly initialized and fixed target network's output as intrinsic rewards.

For more details, please refer to the paper:
    Burda Y, Edwards H, Storkey A, et al. Exploration by random network distillation[J]. 
    arXiv preprint arXiv:1810.12894, 2018. <link https://arxiv.org/abs/1810.12894 link>
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple


# Define the RND Network class
class RNDNetwork(nn.Module):
    """
    **Overview**:
        The definition of Random Network Distillation (RND) Network. RND is a form of reward model that generates
        intrinsic rewards to encourage exploration behaviors of the agent. The main principle is to calculate the
        discrepancy between a predictor network's prediction and a randomly initialized and fixed target network's
        output as intrinsic rewards. For more details, please refer to the paper: <link https://arxiv.org/abs/1810.12894 link>.
    """

    # Initialize the RND Network class
    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super(RNDNetwork, self).__init__()

        # Define the target and predictor networks as simple 2-layer fully connected networks
        # The target network's weights are fixed, while the predictor network's weights will be updated during training
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
    # Define the forward pass of the RND Network
    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        **Overview**:
            The forward function of the RND network.
            It returns the target features and the predicted features.
        Arguments:
            - obs (:obj:`Tensor`): The input observation tensor.
        Returns:
            - target_feature (:obj:`Tensor`): The target feature tensor.
            - pred_feature (:obj:`Tensor`): The predicted feature tensor.
        """
        # The target network's forward pass is done with no gradients computed
        with torch.no_grad():
            target_feature = self.target_network(obs)
        # The predictor network's forward pass
        pred_feature = self.predictor_network(obs)
        return target_feature, pred_feature

    # delimiter
    # Define the intrinsic reward computation for the RND Network
    def compute_rnd_intrinsic_reward(self, obs: Tensor) -> Tensor:
        """
        **Overview**:
            Given the input observation tensor, the RND model will compute the target and predicted features,
            and return the squared difference between them as the intrinsic reward.
        Arguments:
            - obs (:obj:`Tensor`): The input observation tensor.
        Returns:
            - rnd_reward (:obj:`Tensor`): The RND intrinsic reward tensor.
        """
        # The intrinsic reward is the squared difference between the target and predicted features
        target_feature, pred_feature = self.forward(obs)
        rnd_reward = (target_feature - pred_feature).pow(2).sum(dim=1)
        return rnd_reward


# delimiter
# Define the training function for the RND network
def train(model: RNDNetwork, optimizer: optim.Optimizer, obs: Tensor, total_train_steps: int = 1000) -> float:
    """
    **Overview**:
        Training function for the RND model.
    """
    # The model is trained for a fixed number of steps
    for _ in range(total_train_steps):
        # Forward pass
        target_feature, pred_feature = model(obs)

        # Compute the loss as the MSE between the target and predicted features
        loss = ((target_feature - pred_feature) ** 2).mean()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the final training loss
    return loss.item()


# delimiter
# Define the evaluation function for the RND network
def eval(model: RNDNetwork, obs: Tensor) -> float:
    """
     **Overview**:
        Testing function for the RND model.
    """
    # The evaluation is done with no gradients computed
    with torch.no_grad():
        # Forward pass
        target_feature, pred_feature = model(obs)

        # Compute the loss as the MSE between the target and predicted features
        loss = ((target_feature - pred_feature) ** 2).mean()

    # Return the evaluation loss
    return loss.item()


# delimiter
# Define the function to compute the RND intrinsic reward
def compute_rnd_reward(model: RNDNetwork, obs: Tensor) -> None:
    """
    **Overview**:
        Compute the RND intrinsic reward using the trained RND model.
    """
    # Compute the intrinsic reward and print it
    rnd_reward = model.compute_rnd_intrinsic_reward(obs)
    print(f"RND intrinsic reward: {rnd_reward}")

    # Assert that the reward has the correct shape, data type, and values
    assert rnd_reward.shape == (obs.shape[0],)
    assert rnd_reward.dtype == torch.float32
    assert rnd_reward.min() >= 0.0


# delimiter
# Define the test function for the RND model
def test_icm():
    """
    **Overview**:
        This function serves as a testing routine for the Random Network Distillation (RND) model. The test includes the following steps:

        1. Initialization of the RND model with specified dimensions for the states and hidden layers.
        2. Generation of synthetic observations to simulate states.
        3. Training of the RND model using generated observations and the Adam optimizer.
        4. Evaluation of the trained model's performance based on its prediction loss.
        5. Computation of the RND intrinsic reward based on the model's prediction error.

        The function is intended for debugging and performance assessment of the RND model in a controlled setting with synthetic data.
    """
    # Define the dimensions for the observation space and the hidden layers
    obs_dim = 10
    hidden_dim = 20

    # Initialize the RND model and the Adam optimizer
    model = RNDNetwork(obs_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters())

    # Generate some random observations
    obs = torch.randn(100, obs_dim)

    # Train the model and print the final training loss
    train_loss = train(model, optimizer, obs)
    print(f"Train loss: {train_loss}")

    # Evaluate the model and print the final evaluation loss
    eval_loss = eval(model, obs)
    print(f"Eval loss: {eval_loss}")

    # Test the computation of the RND intrinsic reward
    compute_rnd_reward(model, obs)