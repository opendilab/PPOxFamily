
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import numpy as np
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.encoder = nn.Sequential(
        nn.Linear(obs_shape, 2),
        nn.Tanh(),
        )
        self.log_sigma = nn.Parameter(torch.zeros(1, action_shape))

    # delimiter
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        mu_theta = (x[:, :1] + 1) * np.pi / 4
        mu_v = (x[:, 1:] + 1) * 10 / 2
        mu = torch.cat([mu_theta, mu_v], dim=-1)

        log_sigma = self.log_sigma + torch.zeros_like(mu)
        sigma_theta = torch.exp(log_sigma[:, :1]) * np.pi / 2
        sigma_v = torch.exp(log_sigma[:, 1:]) * 10
        sigma = torch.cat([sigma_theta, sigma_v], dim=-1)
        return {'mu': mu, 'sigma': sigma}


class Agent(object):
    def __init__(self, obs_shape, action_shape, batch_size, device) -> None:
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.trajectory = []
        self.batch_size = batch_size
        self.device = device
        self.policy = PolicyNetwork(obs_shape, action_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters())
        self.GAMMA = 0.95

    def select_actions(self, obs):
        obs_in = torch.Tensor(obs).unsqueeze(dim=0).to(self.device)
        logit = self.policy(obs_in)
        dist = Normal(logit["mu"], logit["sigma"])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.squeeze(0).detach().cpu().numpy()
        return action, log_prob

    def save_trajectory(self, obs, action, log_prob, next_obs, reward, done):
        transition = {}
        transition["obs"] = obs
        transition["action"] = action
        transition["log_prob"] = log_prob
        transition["next_obs"] = next_obs
        transition["reward"] = reward
        transition["done"] = done
        self.trajectory.append(transition)

    def compute_gains(self, rewards):
        R = 0
        Gt = []
        for r in rewards[::-1]:
            R = r + self.GAMMA * R
            Gt.insert(0, R)
        return Gt

    def train(self):
        obs_batch = np.array([transition["obs"] for transition in self.trajectory])
        obs_batch_tensor = torch.Tensor(obs_batch).to(self.device)
        log_prob_batch = [transition["log_prob"] for transition in self.trajectory]
        log_prob_batch_tensor = torch.cat(log_prob_batch, dim=0)
        rewards = [transition["reward"] for transition in self.trajectory]
        gains = np.array(self.compute_gains(rewards))
        gains_batch_tensor = torch.Tensor(gains).to(self.device)

        for _ in range(1):
            num = min(self.batch_size, obs_batch.shape[0])
            loss = -(gains_batch_tensor[:num] * log_prob_batch_tensor[:num]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class Env(object):
    def __init__(self, d1, d2, g) -> None:
        self.d1 = d1
        self.d2 = d2
        self.g = g

    def get_obs(self):
        return np.array([self.d1, self.d2])

    def step(self, action):
        v, theta = action
        v_x = v * np.cos(theta)
        v_y = v * np.sin(theta)
        t = self.d1 / v_x
        y_delta = v_y - self.g * (t**2) / 2
        done = False
        reward = 0
        if y_delta >= self.d2:
            done = True
            reward = 100 - v ** 2
        return reward, done


if __name__ == "__main__":
    env = Env(0.2, 0.2, 9.8)
    agent = Agent(2, 2, 32, "cuda")
    while True:
        obs = env.get_obs()
        action, log_prob = agent.select_actions(obs)
        reward, done = env.step(action)
        next_obs = env.get_obs()
        agent.save_trajectory(obs, action, log_prob, reward, next_obs, done)

        if done:
            break
    agent.train()






