from network import PolicyNet
from environment import JumpGame
import torch
import numpy as np
from matplotlib import pyplot as plt

TRAIN_STEPS = 1
LR = 0.01
BATCHSIZE = 128

envs = [JumpGame() for _ in range(BATCHSIZE)]
net = PolicyNet()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
record_reward_mean = []
record_reward_std = []
record_loss_mean = []
record_loss_std = []

for step in range(TRAIN_STEPS):
    # play and collect data
    batch_d = [env.reset(0.2, 0.3) for env in envs]
    batch_d = torch.tensor(batch_d, dtype=torch.float32)
    batch_action, batch_mu, batch_sigma, batch_p = net(batch_d, return_p=True)
    batch_reward = [env.step(action.detach().numpy()) \
                        for env, action in zip(envs, batch_action)]
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
    mean_reward = batch_reward.mean().item()
    std_reward = batch_reward.std().item()
    record_reward_mean.append(mean_reward)
    record_reward_std.append(std_reward)
    # print('step: {}, mean reward: {}'.format(step, mean_reward))
    # print(batch_action.shape, batch_mu.shape, batch_sigma.shape, batch_reward.shape)

    # update
    # loss = - reward * log(Normal(action, mu, sigma))
    loss = - batch_reward * torch.log(batch_p)
    loss_np = loss.detach().numpy()
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    for n, p in net.named_parameters():
        if p.requires_grad:
            print(n, p.data)
            print(p.grad)
            print('----------------')
    print('===============')
    optimizer.step()
    # mean_loss = loss_np.mean()
    # std_loss = loss_np.std()
    # record_loss_mean.append(mean_loss)
    # record_loss_std.append(std_loss)

# print('v:', batch_action[0, 0].item())
# print('theta:', batch_action[0, 1].item())