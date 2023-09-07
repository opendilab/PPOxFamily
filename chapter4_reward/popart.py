"""
PyTorch implementation of ``Pop-Art`` algorithm for adaptive normalization techniques.
<link https://arxiv.org/abs/1602.07714 link>

Pop-Art is an adaptive normalization algorithm to normalized the targets used in the learning updates.
It can be used in value normalization in PPO algorithm to address multi-magnitude reward problem.

The two main components in Pop-Art are:
- **ART**: to update scale and shift such that the return is appropriately normalized
- **POP**: to preserve the outputs of the unnormalized function when we change the scale and shift.
"""
import pickle
import math
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from torch.optim import AdamW
from torch.utils.data import DataLoader


class PopArt(nn.Module):
    """
    **Overview**:
        The definition of Pop-Art layer, i.e., a linear layer with popart normalization, which should be
        used as the last layer of a network.
        For more information, you can refer to the paper <link https://arxiv.org/abs/1809.04474 link>
    """

    def __init__(
            self,
            input_features: int,
            output_features: int,
            beta: float = 0.5
    ) -> None:
        # PyTorch necessary requirements for extending ``nn.Module`` . Our network should also subclass this class.
        super(PopArt, self).__init__()

        # Define soft-update parameter beta.
        self.beta = beta
        # Define the input and output feature dimensions of the linear layer.
        self.input_features = input_features
        self.output_features = output_features
        # Initialize the linear layer parameters, weight and bias.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # Register a buffer for normalization parameters which can not be considered as model parameters.
        # Therefore, the tensor registered in buffer will not refer to gradient propagation but still can
        # be saved in state_dict.
        # The normalization parameters will be used later to save the target value's scale and shift.
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))

        # Reset the learned parameters, i.e., weight and bias.
        self.reset_parameters()

    # delimiter
    def reset_parameters(self) -> None:
        """
        **Overview**:
            The parameters initialization of the linear layer (i.e. weight and bias).
        """
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

    # delimiter
    def forward(self, x: torch.Tensor) -> ttorch.Tensor:
        """
        **Overview**:
            The computation graph of the linear layer with popart mechanism, which outputs both the output and the normalized output of the layer.
        """
        # Execute the linear layer computation $$y=Wx+b$$, note here we use expand and broadcast to add bias.
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        # Unnormalize the output for more convenient usage.
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return ttorch.as_tensor({'output': output, 'normalized_output': normalized_output})

    # delimiter
    def update_parameters(self, value: torch.Tensor) -> ttorch.Tensor:
        """
        **Overview**:
            The parameters update defined in Pop-Art, which outputs both the output and the normalized output of the layer.
        """
        # Tensor device conversion of the normalization parameters.
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)

        # Store the old normalization parameters for later usage.
        old_mu = self.mu
        old_std = self.sigma
        # Calculate the first and second moments (mean and variance) of the target value:
        # $$\mu = \frac{G_t}{B}$$
        # $$v = \frac{G_t^2}{B}$$.
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        # Replace the nan value with old value for more stable training.
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

        # Update the normalization parameters.
        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        # Update weight and bias with mean and standard deviation to preserve unnormalised outputs:
        # $$w'_i = \frac{\sigma_i}{\sigma'_i}w_i$$
        # $$b'_i = \frac{\sigma_i b_i + \mu_i-\mu'_i}{\sigma'_i}$$
        self.weight.data = (self.weight.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.sigma

        # Return treetensor-type statistics.
        return ttorch.as_tensor({'new_mean': batch_mean, 'new_std': batch_std})


# delimiter
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
        # Here we use MLP with two layer and ReLU as activate function. The final layer is popart layer.
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape + action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        self.popart = PopArt(32, 1)

    # delimiter
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> ttorch.Tensor:
        """
        **Overview**:
            Forward computation of the MLP network with popart layer.
        """
        # The encoder first concatenate the observation vectors and actions,
        # then map the input to an embedding vector.
        x = torch.cat((obs, actions), 1)
        x = self.encoder(x)
        # The popart layer maps the embedding vector to a normalized value.
        x = self.popart(x)
        return x


# delimiter
def train(obs_shape: int, action_shape: int, NUM_EPOCH: int, train_data):
    """
    **Overview**:
        Example training function for using MLP network with Pop-Art layer in fixed Q value approximation.
    """
    # Define the MLP network and optimizer, and loss function.
    model = MLP(obs_shape, action_shape)
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    MSEloss = nn.MSELoss()
    # Read the preprocessed data of trained agent on lunarlander environment.
    # Each sample in the datasets should be a dict with following format:
    # $$key\quad dim$$
    # $$observations\quad (*,8)$$
    # $$actions\quad (*,)$$
    # $$returns\quad (*,)$$
    # where the returns is the discounted return from the current state.
    train_data = DataLoader(train_data, batch_size=64, shuffle=True)

    # For loop 1: train MLP network for ``NUM_EPOCH`` epochs.
    running_loss = 0.0
    for epoch in range(NUM_EPOCH):
        # For loop 2: Inside each epoch, split the entire dataset into mini-batches, then train on each mini-batch.
        for idx, data in enumerate(train_data):
            # Compute the original output and the normalized output.
            output = model(data['observations'], data['actions'])
            mu = model.popart.mu
            sigma = model.popart.sigma
            # Normalize the target return to align with the normalized Q value.
            with torch.no_grad():
                normalized_return = (data['returns'] - mu) / sigma
            # The loss is calculated as the MSE loss between normalized Q value and normalized target return.
            loss = MSEloss(output.normalized_output, normalized_return)
            # Loss backward and optimizer update step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # After the model parameters are updated with the gradient,
            # the weights and bias should be updated to preserve unnormalised outputs.
            model.popart.update_parameters(data['returns'])

            # Use ``item`` method to get the pure Python scalar of the loss, then add it into ``running_loss`` .
            running_loss += loss.item()

        # Print the loss every 100 epochs.
        if epoch % 100 == 99:
            print('Epoch [%d] loss: %.6f' % (epoch + 1, running_loss / 100))
            running_loss = 0.0


if __name__ == '__main__':
    # The preprocessed data can be downloaded from:
    # <link https://opendilab.net/download/PPOxFamily/ link>
    with open('ppof_ch4_data_lunarlander.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train(obs_shape=8, action_shape=1, NUM_EPOCH=2000, train_data=dataset)
