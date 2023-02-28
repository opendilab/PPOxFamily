"""
Implementation of ``POPART`` algorithm for reward rescale.
<link https://arxiv.org/abs/1602.07714 link>

POPART is an adaptive normalization algorithm to normalized the targets used in the learning updates.

The two main components in POPART are:
**ART**: to update scale and shift such that the return is appropriately normalized
**POP**: to preserve the outputs of the unnormalized function when we change the scale and shift.
"""
from typing import Dict, Optional, Union
import pickle
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


class PopArt(nn.Module):
    """
    **Overview**:
        A linear layer with popart normalization.
        For more popart implementation info, you can refer to the paper <link https://arxiv.org/abs/1809.04474 link>
    """

    def __init__(
            self,
            input_features: Union[int, None] = None,
            output_features: Union[int, None] = None,
            beta: float = 0.5
    ) -> None:
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(PopArt, self).__init__()

        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        # Initialize the linear layer parameters, weight and bias.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # Register a buffer for normalization parameters which can not be considered as model parameters.
        # The normalization parameters will be used later to save the target value's scale and shift.
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        # In Kaiming Initialization, the mean of weights increment slowly and the std is close to 1,
        # which avoid the vanishing gradient problem and exploding gradient problem of deep models.
        # Specifically, the Kaiming Initialization funciton is as follows:
        # $$std = \sqrt{\frac{2}{(1+a^2)\times fan\_in}}$$
        # where a is the the negative slope of the rectifier used after this layer (0 for ReLU by default),
        # and fan_in is the number of input dimension.
        # For more kaiming intialization info, you can refer to the paper:
        # <link https://arxiv.org/pdf/1502.01852.pdf link>
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        **Overview**:
            The computation of the linear layer, which outputs both the output and the normalized output of the layer.
        """
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        # Unnormalize the output.
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return output, normalized_output

    def update_parameters(self, value):
        """
        **Overview**:
            The parameters update, which outputs both the output and the normalized output of the layer.
        """
        # Tensor device conversion of the normalization parameters.
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)

        old_mu = self.mu
        old_std = self.sigma
        # Calculate the first and second moments (mean and variance) of the target value:
        # $$\mu = \frac{G_t}{B}$$
        # $$v = \frac{G_t^2}{B}$$.
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        # Replace the nan value with old value.
        batch_mean[torch.isnan(batch_mean)] = self.mu[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]
        # Soft update the normalization parameters according to:
        # $$\mu_t = (1-\beta)\mu_{t-1}+\beta G^v_t$$
        # $$v_t = (1-\beta)v_{t-1}+\beta(G^v_t)^2$$.
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        # Calculate the standard deviation with the mean and variance:
        # $$\sigma = \sqrt{v-\mu^2}$$
        batch_std = torch.sqrt(batch_v - (batch_mean ** 2))
        # Clip the standard deviation to reject the outlier data.
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e+6)
        # Replace the nan value with old value.
        batch_std[torch.isnan(batch_std)] = self.sigma[torch.isnan(batch_std)]

        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        # Update weight and bias with mean and standard deviation to preserve unnormalised outputs:
        # $$w'_i = \frac{\sigma_i}{\sigma'_i}w_i$$
        # $$b'_i = \frac{\sigma_i b_i + \mu_i-\mu'_i}{\sigma'_i}$$
        self.weight.data = (self.weight.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.sigma

        return {'new_mean': batch_mean, 'new_std': batch_std}


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
            nn.Linear(obs_shape + action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.popart = PopArt(32, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # The encoder first concatenate the observation vectors and actions,
        # then map the input to an embedding vector.
        x = torch.cat((obs, actions), 1)
        x = self.encoder(x)
        # The popart layer maps the embedding vector to a normalized value.
        normalized_output = self.popart(x)
        return normalized_output


def train(obs_shape: int, action_shape: int, NUM_EPOCH: int, train_data):
    model = MLP(obs_shape, action_shape)
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    MSEloss = nn.MSELoss()
    # Read the preprocessed data of trained agent on lunarlander.
    # Each sample in the datasets should be a dict with following format:
    # $$key\quad dim$$
    # $$observations\quad (*,8)$$
    # $$actions\quad (*,)$$
    # $$rewards\quad (*,)$$
    # where the rewards is the discounted return from the current state.
    train_data = DataLoader(train_data, batch_size=64, shuffle=True)

    running_loss = 0.0
    for epoch in range(NUM_EPOCH):
        for idx, data in enumerate(train_data):
            optimizer.zero_grad()
            # Compute the original output and the normalized output.
            output, normalized_output = model(data['observations'], data['actions'])
            mu = model.popart.mu
            sigma = model.popart.sigma
            # Normalize the target return to align with the normalized Q value.
            with torch.no_grad():
                normalized_reward = (data['rewards'] - mu) / sigma
            # The loss is calculated as the MSE loss between normalized Q value and normalized target return.
            loss = MSEloss(normalized_output, normalized_reward)
            loss.backward()
            optimizer.step()
            # After the model parameters are updated with the gradient,
            # the weights and bias should be updated to preserve unnormalised outputs.
            model.popart.update_parameters(data['rewards'])

            running_loss += loss.item()

        if epoch % 100 == 99:
            print('Epoch [%d] loss: %.6f' % (epoch + 1, running_loss / 100))
            running_loss = 0.0


if __name__ == '__main__':
    # The preprocessed data can be downloaded from:
    # <link https://opendilab.net/download/PPOxFamily/ link>
    with open('ppof_ch4_data_lunarlander.pkl', 'rb') as f:
        dataset = pickle.load(f)
        train(obs_shape=8, action_shape=1, NUM_EPOCH=2000, train_data=dataset)
