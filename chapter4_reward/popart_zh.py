"""
本文件是 "Pop-Art" 自适应归一化算法的 PyTorch 实现，用于自适应归一化技术。
<link https://arxiv.org/abs/1602.07714 link>

Pop-Art 是一种用于归一化学习更新中使用的目标的自适应归一化算法。它可以用来在 PPO 算法中进行价值归一化，以解决多量级奖励问题。

Pop-Art 的两个主要组件: 
- **ART**: 用于更新比例 (scale) 和偏移 (shift)，以便适当归一化回报 (return)
- **POP**: 在我们改变比例和偏移时保留非归一化函数的输出。
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
    **概述**:
        Pop-Art 层的定义，即具有 Pop-Art 归一化功能的线性层，使用中要作为网络的最后一层。
        欲了解更多信息，请参阅论文: <link https://arxiv.org/abs/1809.04474 link>。
    """

    def __init__(
            self,
            input_features: int,
            output_features: int,
            beta: float = 0.5
    ) -> None:
        # 使用 PyTorch 构建网络需要扩展 ``nn.Module``。我们的网络也需要继承这个类。
        super(PopArt, self).__init__()

        # 定义软更新参数 beta。
        self.beta = beta
        # 定义线性层的输入和输出特征维度。
        self.input_features = input_features
        self.output_features = output_features
        # 初始化线性层的参数，权重和偏置。
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # 为不能被视为模型参数的归一化参数注册一个缓冲区。
        # 这样在缓冲区中注册的张量不会参与梯度传播，但仍然可以保存在 state_dict 中。
        # 归一化参数将在后续用于保存目标值的比例和偏移。
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))

        # 重置已学习的参数，即权重和偏置。

        self.reset_parameters()

    # delimiter
    def reset_parameters(self) -> None:
        """
        **概述**:
            线性层（即权重和偏置）的参数初始化。
        """
        # 在 Kaiming 初始化中，权重的均值缓慢增加，标准差接近 1，
        # 这避免了深层模型的梯度消失问题和梯度爆炸问题。
        # 具体而言，Kaiming 初始化函数如下: 
        # $$std = \sqrt{\frac{2}{(1+a^2)\times fan\_in}}$$
        # 其中 a 是在此层后使用的整流器的负斜率（ ReLU 默认为 0），
        # 而 fan_in 是输入维度的数量。
        # 欲了解更多 Kaiming 初始化的信息，可参阅论文: 
        # <link https://arxiv.org/pdf/1502.01852.pdf link>
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # delimiter
    def forward(self, x: torch.Tensor) -> ttorch.Tensor:
        """
        **概述**:
            带有 popart 机制的线性层的计算图，同时输出输出层的输出，和归一化的输出层的输出。
        """
        # 执行线性层计算 $$y=Wx+b$$，注意这里使用扩展 (expand) 和广播 (broadcast) 来添加偏置。
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        # 对输出进行反归一化，以便更方便地使用。
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return ttorch.as_tensor({'output': output, 'normalized_output': normalized_output})

    # delimiter
    def update_parameters(self, value: torch.Tensor) -> ttorch.Tensor:
        """
        **概述**:
            更新 Pop-Art 中定义的参数，输出输出层的输出和归一化的输出层的输出。
        """
        # 转换归一化参数的张量设备。
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)

        # 存储旧的归一化参数以备后用。
        old_mu = self.mu
        old_std = self.sigma
        # 计算目标值的一阶和二阶矩（均值和方差）: 
        # $$\mu = \frac{G_t}{B}$$
        # $$v = \frac{G_t^2}{B}$$。
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        # 将 NaN 值替换为旧值，以获得更稳定的训练。
        batch_mean[torch.isnan(batch_mean)] = self.mu[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]
        # 根据以下方式软更新归一化参数: 
        # $$\mu_t = (1-\beta)\mu_{t-1}+\beta G^v_t$$
        # $$v_t = (1-\beta)v_{t-1}+\beta(G^v_t)^2$$。
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        # 利用均值和方差计算标准差: 
        # $$\sigma = \sqrt{v-\mu^2}$$
        batch_std = torch.sqrt(batch_v - (batch_mean ** 2))
        # 剪切标准差以剔除异常数据。
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e+6)
        # 将 NaN 值替换为旧值。
        batch_std[torch.isnan(batch_std)] = self.sigma[torch.isnan(batch_std)]

        # 更新归一化参数。
        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        # 使用均值和标准差更新权重和偏置，以保留非归一化的输出: 
        # $$w'_i = \frac{\sigma_i}{\sigma'_i}w_i$$
        # $$b'_i = \frac{\sigma_i b_i + \mu_i-\mu'_i}{\sigma'_i}$$
        self.weight.data = (self.weight.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.sigma

        # 返回 treetensor 类型的统计信息。
        return ttorch.as_tensor({'new_mean': batch_mean, 'new_std': batch_std})


# delimiter
class MLP(nn.Module):

    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **概述**:
            使用 popart 层作为最后一层的 MLP 网络。
            输入: 观测和动作
            输出: 估计的 Q 值
            ``cat(obs,actions) -> 编码器 -> popart``。
        """
        super(MLP, self).__init__()
        # 定义编码器和 popart 层。
        # 在这里，我们使用两层以 ReLU 为激活函数的 MLP。最后一层是 popart 层。
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
        **概述**:
            带有 popart 层的 MLP 网络的前向计算。
        """
        # 编码器首先将观测向量和动作连接起来，
        # 然后将输入映射到编码后的向量。
        x = torch.cat((obs, actions), 1)
        x = self.encoder(x)
        # popart 层将编码后的向量映射到归一化值。
        x = self.popart(x)
        return x


# delimiter
def train(obs_shape: int, action_shape: int, NUM_EPOCH: int, train_data):
    """
    **概述**:
        示例训练函数，使用具有 Pop-Art 层的 MLP 网络进行固定 Q 值逼近。
    """
    # 定义 MLP 网络、优化器和损失函数。
    model = MLP(obs_shape, action_shape)
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    MSEloss = nn.MSELoss()
    # 读取在月球着陆环境 (lunarlander) 上训练的代理的预处理数据。
    # 数据集中的每个样本应具有以下格式的字典: 
    # $$key\quad dim$$
    # $$observations\quad (*,8)$$
    # $$actions\quad (*,)$$
    # $$returns\quad (*,)$$
    # 其中 returns 是从当前状态开始的折扣回报。
    train_data = DataLoader(train_data, batch_size=64, shuffle=True)

    # 循环 1: 对 MLP 网络进行 ``NUM_EPOCH`` 轮训练。
    running_loss = 0.0
    for epoch in range(NUM_EPOCH):
        # 循环 2: 在每个 epoch 内，将整个数据集分成小批次，然后对每个小批次进行训练。
        for idx, data in enumerate(train_data):
            # 计算原始输出和归一化输出。
            output = model(data['observations'], data['actions'])
            mu = model.popart.mu
            sigma = model.popart.sigma
            # 将目标回报归一化，以便与归一化的 Q 值对齐。
            with torch.no_grad():
                normalized_return = (data['returns'] - mu) / sigma
            # 损失计算为归一化 Q 值和归一化目标回报之间的 MSE 损失。
            loss = MSEloss(output.normalized_output, normalized_return)
            # 反向传播损失并更新优化器。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 在模型参数通过梯度更新后，需要更新权重和偏置以保留非归一化的输出。
            model.popart.update_parameters(data['returns'])

            # 使用 ``item`` 方法获取纯 Python 标量的损失，然后将其添加到 ``running_loss`` 中。
            running_loss += loss.item()

        # 每 100 轮打印一次损失。
        if epoch % 100 == 99:
            print('Epoch [%d] loss: %.6f' % (epoch + 1, running_loss / 100))
            running_loss = 0.0


if __name__ == '__main__':
    # 预处理数据可以从以下链接下载: 
    # <link https://opendilab.net/download/PPOxFamily/ link>
    with open('ppof_ch4_data_lunarlander.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train(obs_shape=8, action_shape=1, NUM_EPOCH=2000, train_data=dataset)
