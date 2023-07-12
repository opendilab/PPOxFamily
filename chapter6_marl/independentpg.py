"""
PyTorch tutorial for independent policy gradient algorithm for multi-agent cooperation scenarios, which is the most simple multi-agent reinforcement learning algorithm.
This tutorial utilizes two-types networks defined in ``marl_network`` and loss function defined in ``pg``. The main function describes the core part of independent policy gradient algorithm with fake data.
More details about multi-agent cooperation reinforcement learning can be found in <link https://github.com/opendilab/PPOxFamily/blob/main/chapter6_marl/chapter6_lecture.pdf link>.

This tutorial is mainly composed of two parts, you can learn from these parts in order or just jump to the part you are interested in:
  - Independent policy gradient algorithm for multi-agent RL
  - Independent actor-critic algorithm for multi-agent RL
"""
import torch
from marl_network import IndependentActorCriticNetwork
# You need to copy the implementation of pg in chapter1_overview
from pg import pg_data, pg_error


def independentpg_training_opeator():
    """
    **Overview**:
        The main function about the training process of independent policy gradient algorithm. Define some hyper-parameters,
        the neural network and optimizer, then generate fake data and calculate the policy gradient loss. Finally,
        update the network parameters with optimizer. In practice, the training data should be replaced by the results
        getting from the interacting with the environment.
    """
    # Set necessary hyper-parameters.
    batch_size, agent_num, local_state_dim, global_state_dim, action_dim = 4, 5, 10, 20, 6
    # Entropy bonus weight, which is beneficial to exploration.
    entropy_weight = 0.001
    # Discount factor for future reward.
    discount_factor = 0.99
    # Set the tensor device to cuda or cpu according to the runtime environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the multi-agent neural network and optimizer.
    # Here we use the independent actor-critic network as the example, you can also use the shared-parameter network.
    model = IndependentActorCriticNetwork(agent_num, local_state_dim, action_dim)
    model.to(device)
    # Adam is the most commonly used optimizer in deep reinforcement learning. If you want to add weight decay
    # mechanism, you should use torch.optim.AdamW.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the corresponding fake data following the same data format of the interacting with the environment.
    # Note that the data should keep the same device with the network.
    # For simplicity, we regard the whole batch data as a entire episode.
    # In practice, the training batch is the combination of multiple episodes. We often use ``done`` variable to
    # distinguish the different episodes.
    local_state = torch.randn(batch_size, agent_num, local_state_dim).to(device)
    action = torch.randint(0, action_dim, (batch_size, agent_num)).to(device)
    reward = torch.randn(batch_size, agent_num).to(device)
    # For naive policy gradient algorithm, the return is computed by the discounted cumulative sum of the reward.
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-critic network forward propagation.
    output = model(local_state)
    # Prepare the data for policy gradient loss calculation.
    data = pg_data(output.logit, action, return_)
    # Calculate the policy gradient loss.
    loss = pg_error(data)
    # Weighted sum of policy loss and entropy loss.
    # Note here we only use the policy network part of the actor-critic network and compute policy loss.
    # If you want to use the value network part, you should define the value loss and add it to the total loss.
    total_loss = loss.policy_loss - entropy_weight * loss.entropy_loss

    # PyTorch loss back propagation and optimizer update.
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print('independentpg_training_opeator is ok')


def independentac_training_opeator():
    """
    **Overview**:
        The main function about the training process of independent actor-critic algorithm. Define some hyper-parameters,
        the neural network and optimizer, then generate fake data and calculate the actor-critic loss. Finally,
        update the network parameters with optimizer. In practice, the training data should be replaced by the results
        getting from the interacting with the environment.
        BTW, policy network means actor and value network indicates critic in this file.
    """
    # Set necessary hyper-parameters.
    batch_size, agent_num, local_state_dim, global_state_dim, action_dim = 4, 5, 10, 20, 6
    # Entropy bonus weight, which is beneficial to exploration.
    entropy_weight = 0.001
    # Value loss weight, which aims to balance the loss scale.
    value_weight = 0.5
    # Discount factor for future reward.
    discount_factor = 0.99
    # Set the tensor device to cuda or cpu according to the runtime environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the multi-agent neural network and optimizer.
    # Here we use the independent actor-critic network as the example, you can also use the shared-parameter network.
    model = IndependentActorCriticNetwork(agent_num, local_state_dim, action_dim)
    model.to(device)
    # Adam is the most commonly used optimizer in deep reinforcement learning. If you want to add weight decay
    # mechanism, you should use torch.optim.AdamW.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the corresponding fake data following the same data format of the interacting with the environment.
    # Note that the data should keep the same device with the network.
    # For simplicity, we regard the whole batch data as a entire episode.
    # In practice, the training batch is the combination of multiple episodes. We often use ``done`` variable to
    # distinguish the different episodes.
    local_state = torch.randn(batch_size, agent_num, local_state_dim).to(device)
    action = torch.randint(0, action_dim, (batch_size, agent_num)).to(device)
    reward = torch.randn(batch_size, agent_num).to(device)
    # Return_ can be computed with different methods. Here we use the discounted cumulative sum of the reward.
    # You can also use the generalized advantage estimation (GAE) method, n-step return method, etc.
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-critic network forward propagation.
    output = model(local_state)
    # ``squeeze`` operation transforms shape from $$(batch_size, agent_num, 1)$$ to $$(batch_size, agent_num)$$.
    value = output.value.squeeze(-1)
    # Prepare the data for policy gradient loss calculation.
    # ``detach`` operation means stop gradient for ``value`` in policy gradient loss calculation.
    data = pg_data(output.logit, action, value.detach())
    # Calculate the policy gradient loss.
    loss = pg_error(data)
    # Calculate the value loss.
    value_loss = torch.nn.functional.mse_loss(value, return_)
    # Weighted sum of policy loss, value loss and entropy loss.
    total_loss = loss.policy_loss + value_weight * value_loss - entropy_weight * loss.entropy_loss

    # PyTorch loss back propagation and optimizer update.
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print('independentac_training_opeator is ok')


if __name__ == "__main__":
    independentpg_training_opeator()
    independentac_training_opeator()
