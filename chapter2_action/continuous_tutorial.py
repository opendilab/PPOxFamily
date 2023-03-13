"""
PyTorch tutorial of ``Proximal Policy Optimization (PPO)``  algorithm for continuous action.
<link https://arxiv.org/pdf/1707.06347.pdf link>

PPO is one of the most popular policy gradient methods for deep reinforcement learning. It combines the classic Actor-Critic paradigm and the trust region policy optimization method into a simple yet effect algorithm design. Compared to some traditional RL algorithms like REINFORCE and A2C, PPO can deploy more stable and efficient policy optimization by using clipped surrogate objective mentioned below:
$$J(\theta) = \min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
The final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective, which only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse.
Detailed notation definition can be found in <link https://github.com/opendilab/PPOxFamily/blob/main/chapter1_overview/chapter1_notation.pdf link>.

Continuous action space, one of the most commonly used action spaces, is often used in practical decision applications such as robot manipulation and drone control. It contains serveral controllable continuous actions and RL agent needs to output proper and accuracy values every execution. Continuous action space is often directly predicted or modelled by gaussian distribution (regression problem).

This tutorial is mainly composed of the following three parts, you can learn from these demo codes step by step or using them as code segment in your own program:
  - Policy Network Architecture
  - Sample Action Function
  - Main (Test) Function
More visulization results about PPO in continuous action space can be found in <link https://github.com/opendilab/PPOxFamily/issues/4 link>.
"""
from typing import Dict
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ContinuousPolicyNetwork(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of continuous action policy network used in PPO, which is mainly composed of three parts: encoder, mu and log_sigma.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(ContinuousPolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network (CNN) for image state and Multilayer perceptron (MLP) for vector state, respectively.
        # Here we use two-layer MLP for vector state.
        # $$ y = max(W_2 max(W_1x+b_1, 0) + b_2, 0)$$
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        # Define mu module, which is a FC and outputs the argument mu for gaussian distribution.
        # $$ \mu = Wx + b $$
        self.mu = nn.Linear(32, action_shape)
        # Define log_sigma module, which is a learnable parameter but independent to state.
        # Here we set it as log_sigma for the convenience of optimization and usage. You can also adjust its initial value for your demands.
        # $$\sigma = e^w$$
        self.log_sigma = nn.Parameter(torch.zeros(1, action_shape))

    # delimiter
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        **Overview**:
            The computation graph of continuous action policy network used in PPO.
            ``x -> encoder -> mu -> \mu`` .
            ``log_sigma -> exp -> sigma`` .
        """
        # Transform original state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        x = self.encoder(x)
        # Output the argument mu depending on the embedding vector, i.e. $$(B, N) -> (B, A)$$
        mu = self.mu(x)
        # Utilize broadcast mechanism to make the same shape between log_sigma and mu.
        # ``zeros_like`` operation doesn't pass gradient.
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.log_sigma + torch.zeros_like(mu)
        # Utilize exponential operation to produce the actual sigma.
        # $$\sigma = e^w$$
        sigma = torch.exp(log_sigma)
        return {'mu': mu, 'sigma': sigma}


# delimiter
def sample_continuous_action(logit: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling continuous action, input is a dict with two keys ``mu`` and ``sigma`` ,
        both of them has shape = (B, action_shape), output shape = (B, action_shape).
        In this example, the distributions shapes are:
        batch_shape = (B, ), event_shape = (action_shape, ), sample_shape = ().
    """
    # Construct gaussian distribution, i.e.
    # $$X \sim \mathcal{N}(\mu,\,\sigma^{2})$$
    # Its probability density function is: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)$$
    # <link https://en.wikipedia.org/wiki/Normal_distribution link>
    dist = Normal(logit['mu'], logit['sigma'])
    # Reinterpret ``action_shape`` gaussian distribution into a multivariate gaussian distribution with diagonal convariance matrix.
    # Ensure each event is independent with each other.
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    dist = Independent(dist, 1)
    # Sample one action of the shape ``action_shape`` per sample (state input) and return it.
    return dist.sample()


# delimiter
def test_sample_continuous_action():
    """
    **Overview**:
        The function of testing sampling continuous action. Construct a standard continuous action
        policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    # ``action_shape`` is different from discrete and continuous action. The former is the possible
    # choice of a discrete action while the latter is the dimension of continuous action.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define continuous action network (which is similar to reparameterization) with encoder, mu and log_sigma.
    policy_network = ContinuousPolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output dict-type logit.
    # $$ \mu, \sigma = \pi(a|s)$$
    logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['mu'].shape == (B, action_shape)
    assert logit['sigma'].shape == (B, action_shape)
    # Sample action accoding to corresponding logit (i.e., mu and sigma).
    action = sample_continuous_action(logit)
    assert action.shape == (B, action_shape)


if __name__ == "__main__":
    test_sample_continuous_action()
