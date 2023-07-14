"""
PyTorch tutorial for basic centralized training and decentralized execution (CTDE) MAPPO algorithm for multi-agent cooperation scenarios.
This tutorial utilizes CTDEActorCriticNetwork defined in ``marl_network`` and loss function defined in ``pg``. The main function describes the core part of CTDE MAPPO algorithm with fake data.
More details about multi-agent cooperation reinforcement learning can be found in <link https://github.com/opendilab/PPOxFamily/blob/main/chapter6_marl/chapter6_lecture.pdf link>.
"""
import torch
from marl_network import CTDEActorCriticNetwork
# You need to copy the implementation of ppo in chapter1_overview
from ppo import ppo_policy_data, ppo_policy_error
# You need to copy the implementation of gae in chapter7_tricks
from gae import gae


def mappo_training_opeator() -> None:
    """
    **Overview**:
        The main function about the training process of CTDE actor-critic algorithm. Define some hyper-parameters,
        the neural network and optimizer, then generate fake data and calculate the actor-critic loss. Finally,
        update the network parameters with optimizer. In practice, the training data should be replaced by the results
        getting from the interacting with the environment.
        BTW, policy network means actor and value network indicates critic in this file.
    """
    # Set necessary hyper-parameters.
    batch_size, agent_num, local_state_shape, agent_specific_global_state_shape, action_shape = 4, 5, 10, 25, 6
    # Entropy bonus weight, which is beneficial to exploration.
    entropy_weight = 0.001
    # Value loss weight, which aims to balance the loss scale.
    value_weight = 0.5
    # Discount factor for future reward.
    discount_factor = 0.99
    # Set the tensor device to cuda or cpu according to the runtime environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the multi-agent neural network and optimizer.
    model = CTDEActorCriticNetwork(agent_num, local_state_shape, agent_specific_global_state_shape, action_shape)
    model.to(device)
    # Adam is the most commonly used optimizer in deep reinforcement learning. If you want to add weight decay
    # mechanism, you should use torch.optim.AdamW.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the corresponding fake data following the same data format of the interacting with the environment.
    # Note that the data should keep the same device with the network.
    # For simplicity, we regard the whole batch data as a entire episode.
    # In practice, the training batch is the combination of multiple episodes. We often use ``done`` variable to
    # distinguish the different episodes.
    local_state = torch.randn(batch_size, agent_num, local_state_shape).to(device)
    agent_specific_global_state = torch.randn(batch_size, agent_num, agent_specific_global_state_shape).to(device)
    logit_old = torch.randn(batch_size, agent_num, action_shape).to(device)
    value_old = torch.randn(batch_size, agent_num).to(device)
    done = torch.zeros(batch_size).to(device)
    done[-1] = 1
    action = torch.randint(0, action_shape, (batch_size, agent_num)).to(device)
    reward = torch.randn(batch_size, agent_num).to(device)
    # Return_ can be computed with different methods. Here we use the discounted cumulative sum of the reward.
    # You can also use the generalized advantage estimation (GAE) method, n-step return method, etc.
    return_ = torch.zeros_like(reward)
    for i in reversed(range(batch_size)):
        return_[i] = reward[i] + (discount_factor * return_[i + 1] if i + 1 < batch_size else 0)

    # Actor-critic network forward propagation.
    output = model(local_state, agent_specific_global_state)
    # ``squeeze`` operation transforms shape from $$(batch_size, agent_num, 1)$$ to $$(batch_size, agent_num)$$.
    value = output.value.squeeze(-1)
    # Use generalized advantage estimation (GAE) method to calculate the advantage.
    # Advantage is a kind of "weight" for policy loss, therefore it is wrapperd in ``torch.no_grad()``.
    # ``done`` is the terminal flag of the episode. ``traj_flag`` is the flag of the trajectory.
    # Here we regard the whole batch data as a entire episode, so ``done`` and ``traj_flag`` are the same.
    with torch.no_grad():
        traj_flag = done
        gae_data = (value, value_old, reward, done, traj_flag)
        adv = gae(gae_data, discount_factor, 0.95)
    # Prepare the data for PPO policy loss calculation.
    data = ppo_policy_data(output.logit, logit_old, action, adv, None)
    # Calculate the PPO policy loss.
    loss, info = ppo_policy_error(data)
    # Calculate the value loss.
    value_loss = torch.nn.functional.mse_loss(value, return_)
    # Weighted sum of policy loss, value loss and entropy loss.
    total_loss = loss.policy_loss + value_weight * value_loss - entropy_weight * loss.entropy_loss

    # PyTorch loss back propagation and optimizer update.
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # Logging the training information.
    print(
        'total_loss: {:.4f}, policy_loss: {:.4f}, value_loss: {:.4f}, entropy_loss: {:.4f}'.format(
            total_loss, loss.policy_loss, value_loss, loss.entropy_loss
        )
    )
    print('approximate_kl_divergence: {:.4f}, clip_fraction: {:.4f}'.format(info.approx_kl, info.clipfrac))
    print('mappo_training_opeator is ok')


if __name__ == "__main__":
    mappo_training_opeator()
