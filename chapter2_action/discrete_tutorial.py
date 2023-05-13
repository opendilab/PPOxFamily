"""
PyTorch tutorial of ``Proximal Policy Optimization (PPO)``  algorithm for discrete action.
<link https://arxiv.org/pdf/1707.06347.pdf link>

PPO is one of the most popular policy gradient methods for deep reinforcement learning. It combines the classic Actor-Critic paradigm and the trust region policy optimization method into a simple yet effect algorithm design. Compared to some traditional RL algorithms like REINFORCE and A2C, PPO can deploy more stable and efficient policy optimization by using clipped surrogate objective mentioned below:
$$J(\theta) = \min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
The final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective, which only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse.
Detailed notation definition can be found in <link https://github.com/opendilab/PPOxFamily/blob/main/chapter1_overview/chapter1_notation.pdf link>.

Discrete action space, one of the most commonly used action spaces, is often used in video games such as Super Mario Bros, Atari and Procgen. It contains a group of possible discrete action choices and RL agent needs to select one action from them every execution. Discrete action space is often modelled by categorical distribution (classification problem).

This tutorial is mainly composed of the following three parts, you can learn from these demo codes step by step or using them as code segment in your own program:
  - Policy Network Architecture
  - Sample Action Function
  - Main (Test) Function
More visulization results about PPO in discrete action space can be found in <link https://github.com/opendilab/PPOxFamily/issues/4 link>.
"""
from typing import List
import torch
import torch.nn as nn


class DiscretePolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of discrete action policy network used in PPO, which is mainly composed of two parts: encoder and head.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(DiscretePolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network (CNN) for image state and Multilayer perceptron (MLP) for vector state, respectively.
        # Here we use one-layer MLP for vector state, i.e.
        # $$y = max(Wx+b, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
        )
        # Define discrete action logit output network, just one-layer FC, i.e.
        # $$y=Wx+b$$
        self.head = nn.Linear(32, action_shape)

    # delimiter
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        **Overview**:
            The computation graph of discrete action policy network used in PPO.
            ``x -> encoder -> head -> logit`` .
        """
        # Transform original state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # Calculate logit for each possible discrete action choices, i.e. $$(B, N) -> (B, A)$$
        logit = self.head(x)
        return logit


# delimiter
class MultiDiscretePolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: List[int]) -> None:
        """
        **Overview**:
            The definition of multi discrete action policy network used in PPO, which uses multiple discrete head.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(MultiDiscretePolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network (CNN) for image state and Multilayer perceptron (MLP) for vector state, respectively.
        # Here we use one-layer MLP for vector state, i.e.
        # $$y = max(Wx+b, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
        )
        # Define multiple discrete head according to the concrete sub action size.
        self.head = nn.ModuleList()
        for size in action_shape:
            self.head.append(nn.Linear(32, size))

    # delimiter
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        **Overview**:
            The computation graph of discrete action policy network used in PPO.
            ``x -> encoder -> multiple head -> multiple logit`` .
        """
        # Transform original state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # Calculate multiple logit for each possible discrete action, i.e. $$(B, N) -> [(B, A_1), ..., (B, A_N)]$$
        logit = [h(x) for h in self.head]
        return logit


# delimiter
def sample_action(logit: torch.Tensor) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling discrete action, input shape = (B, action_shape), output shape = (B, ).
        In this example, the distributions shapes are:
        batch_shape = (B, ), event_shape = (), sample_shape = ().
    """
    # Transform logit (raw output of policy network, e.g. last fully connected layer) into probability.
    # $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
    prob = torch.softmax(logit, dim=-1)
    # Construct categorical distribution. The probability mass function is: $$f(x=i|\boldsymbol{p})=p_i$$
    # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
    dist = torch.distributions.Categorical(probs=prob)
    # Sample one discrete action per sample (state input) and return it.
    return dist.sample()


# delimiter
def test_sample_discrete_action():
    """
    **Overview**:
        The function of testing sampling discrete action. Construct a naive policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define policy network with encoder and head.
    policy_network = DiscretePolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output logit.
    # $$ logit = \pi(a|s)$$
    logit = policy_network(state)
    assert logit.shape == (B, action_shape)
    # Sample action accoding to corresponding logit.
    action = sample_action(logit)
    assert action.shape == (B, )


# delimiter
def test_sample_multi_discrete_action():
    """
    **Overview**:
        The function of testing sampling multi-discrete action. Construct a naive policy and sample a group of multi-discrete action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = [4, 5, 6].
    B, obs_shape, action_shape = 4, 10, [4, 5, 6]
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define policy network with encoder and head.
    policy_network = MultiDiscretePolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output multiple logit.
    # $$ logit = \pi(a|s)$$
    logit = policy_network(state)
    for i in range(len(logit)):
        assert logit[i].shape == (B, action_shape[i])
    # Sample action accoding to corresponding logit one by one.
    for i in range(len(logit)):
        action_i = sample_action(logit[i])
        assert action_i.shape == (B, )


if __name__ == "__main__":
    test_sample_discrete_action()
    test_sample_multi_discrete_action()
