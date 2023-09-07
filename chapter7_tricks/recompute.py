"""
PyTorch implementation of PPO training loop with recompute advantage trick, which is beneficial to the training stability and overall performance.
"""
# Import necessary packages.
import math
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from gae import gae
# You need to copy the implementation of ppo in chapter1_overview
from ppo import ppo_policy_data, ppo_policy_error


# Define naive actor-critic model as example, you can modify it in your own way.
class NaiveActorCritic(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> ttorch.Tensor:
        logit = self.actor(obs)
        value = self.critic(obs)
        return ttorch.as_tensor({'logit': logit, 'value': value})


# delimiter
def ppo_training_loop_with_recompute():
    """
    **Overview**:
        The training loop function example of PPO algorithm on discrete action space with recompute advantage trick.
    """
    # The number of training epochs after per data collection.
    epoch_per_collect = 10
    # The total number of collected data once.
    collected_data_num = 127
    # Entropy bonus weight, which is beneficial to exploration.
    entropy_weight = 0.001
    # Value loss weight, which aims to balance the loss scale.
    value_weight = 0.5
    # Discount factor for future reward.
    discount_factor = 0.99
    # Whether to recompute the GAE advantage at the beginning of each epoch.
    recompute = True
    # The number of samples in each batch.
    batch_size = 16
    # The shape of observation and action, which is different between different environments.
    obs_shape, action_shape = 8, 4

    # Create the model and optimizer, here we use the naive implementation as example, you can modify it in your own way.
    model = NaiveActorCritic(obs_shape, action_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # The function of generating a random transition for training.
    # Here we use treetensor to express the structured transition, which is convenient for batch processing.
    # ``squeeze`` is to ensure that the shape of each tensor is $$(B, )$$ instead of $$(B, 1)$$.
    def get_ppo_training_transition():
        return ttorch.as_tensor(
            {
                'obs': torch.randn(obs_shape),
                'action': torch.randint(action_shape, size=(1, )).squeeze(),
                'reward': torch.rand(1).squeeze(),
                'next_obs': torch.randn(obs_shape),
                'done': torch.randint(2, size=(1, )).squeeze(),
                'logit': torch.randn(action_shape),
                'value': torch.randn(1).squeeze(),
                'adv': torch.randn(1).squeeze(),
            }
        )

    # Generate ``collected_data_num`` random transitions and pack them into a list.
    data = [get_ppo_training_transition() for _ in range(collected_data_num)]
    # Stack the list into a treetensor batch.
    data = ttorch.stack(data)
    # Print the shape of the structured data batch.
    print(data.shape)

    # For loop 1: train the latest collected data for ``epoch_per_collect`` epochs.
    for e in range(epoch_per_collect):
        # Recompute the GAE advantage at the beginning of each epoch.
        # Usually, advantage is pre-computed in data collection to save time. However, with the updates of value
        # network, the advantage will be out of date. So we need to recompute it to ensure the training effect.
        if recompute:
            # Advantage calculation doesn't need gradient back propagation, so we use ``torch.no_grad()`` to save memory.
            with torch.no_grad():
                # Use the latest value network to calculate value, then replace the old value with the new one.
                latest_value = model(data.obs).value.squeeze(-1)
                gae_data = (latest_value, data.value, data.reward, data.done, data.done)
                data.adv = gae(gae_data, discount_factor, 0.95)
        # Randomly shuffle the collected data, generate the indices for mini-batch.
        indices = torch.randperm(collected_data_num)
        # For loop 2: inside each epoch, divide all the collected data into many mini-batch,
        # i.e. train the model with ``batch_size`` samples per iteration.
        for iter_ in range(math.ceil(collected_data_num / batch_size)):
            # Get the mini-batch data with the cooresponding indices.
            batch = data[indices[iter_ * batch_size:(iter_ + 1) * batch_size]]

            # Call model forward procedure.
            output = model(batch.obs)

            # ``squeeze`` operation transforms shape from $$(B, A, 1)$$ to $$(B, A)$$.
            value = output.value.squeeze(-1)
            # Calculate the return value. Here we use the sum of value and adv for simplicity.
            # You can also use other methods to calculate return, such as n-step return method.
            return_ = value + batch.adv
            # Prepare the data for PPO policy loss calculation.
            ppo_data = ppo_policy_data(output.logit, batch.logit, batch.action, batch.adv, None)
            # Calculate the PPO policy loss.
            loss, info = ppo_policy_error(ppo_data)
            # Calculate the value loss.
            value_loss = torch.nn.functional.mse_loss(value, return_)
            # Weighted sum of policy loss, value loss and entropy loss.
            total_loss = loss.policy_loss + value_weight * value_loss - entropy_weight * loss.entropy_loss

            # PyTorch loss back propagation and optimizer update.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    print('ppo_training_loop_with_recompute finish')


if __name__ == '__main__':
    ppo_training_loop_with_recompute()
