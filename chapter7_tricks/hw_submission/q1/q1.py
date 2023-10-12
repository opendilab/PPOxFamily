from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import treetensor.torch as ttorch


class PPOFModel(nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            obs_shape: Tuple[int],
            action_shape: int,
            encoder_hidden_size_list: List = [128, 128, 64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        super(PPOFModel, self).__init__()
        self.obs_shape, self.action_shape = obs_shape, action_shape

        # encoder
        layers = []
        input_size = obs_shape[0]
        kernel_size_list = [8, 4, 3]
        stride_list = [4, 2, 1]
        for i in range(len(encoder_hidden_size_list)):
            output_size = encoder_hidden_size_list[i]
            layers.append(nn.Conv2d(input_size, output_size, kernel_size_list[i], stride_list[i]))
            layers.append(activation)
            input_size = output_size
        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

        flatten_size = input_size = self.get_flatten_size()
        # critic
        layers = []
        for i in range(critic_head_layer_num):
            layers.append(nn.Linear(input_size, critic_head_hidden_size))
            layers.append(activation)
            input_size = critic_head_hidden_size
        layers.append(nn.Linear(critic_head_hidden_size, 1))
        self.critic = nn.Sequential(*layers)
        # actor
        layers = []
        input_size = flatten_size
        for i in range(actor_head_layer_num):
            layers.append(nn.Linear(input_size, actor_head_hidden_size))
            layers.append(activation)
            input_size = actor_head_hidden_size
        self.actor = nn.Sequential(*layers)
        self.mu = nn.Linear(actor_head_hidden_size, action_shape)
        self.log_sigma = nn.Parameter(torch.zeros(1, action_shape))

        # init weights
        self.init_weights()

    def init_weights(self) -> None:
        # You need to implement this function
        # raise NotImplementedError

        # orthogonal init
        def orthogonal_init(layer, gain=1.0):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                orthogonal_init(m)
            elif isinstance(m, nn.Linear):
                orthogonal_init(m)

        # output layer init
        orthogonal_init(self.mu, gain=0.01)


    def get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.encoder(test_data)
        return output.shape[1]

    def forward(self, inputs: ttorch.Tensor, mode: str) -> ttorch.Tensor:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: ttorch.Tensor) -> ttorch.Tensor:
        x = self.encoder(x)
        x = self.actor(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma + torch.zeros_like(mu)  # addition aims to broadcast shape
        sigma = torch.exp(log_sigma)
        return ttorch.as_tensor({'mu': mu, 'sigma': sigma})

    def compute_critic(self, x: ttorch.Tensor) -> ttorch.Tensor:
        x = self.encoder(x)
        value = self.critic(x)
        return value

    def compute_actor_critic(self, x: ttorch.Tensor) -> ttorch.Tensor:
        x = self.encoder(x)
        value = self.critic(x)
        x = self.actor(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma + torch.zeros_like(mu)  # addition aims to broadcast shape
        sigma = torch.exp(log_sigma)
        return ttorch.as_tensor({'logit': {'mu': mu, 'sigma': sigma}, 'value': value})


def test_ppof_model() -> None:
    model = PPOFModel((4, 84, 84), 5)
    print(model)
    data = torch.randn(3, 4, 84, 84)
    output = model(data, mode='compute_critic')
    assert output.shape == (3, 1)
    output = model(data, mode='compute_actor')
    assert output.mu.shape == (3, 5)
    assert output.sigma.shape == (3, 5)
    output = model(data, mode='compute_actor_critic')
    assert output.value.shape == (3, 1)
    assert output.logit.mu.shape == (3, 5)
    assert output.logit.sigma.shape == (3, 5)
    print('End...')


if __name__ == "__main__":
    test_ppof_model()