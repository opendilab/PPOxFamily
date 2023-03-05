# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:57:10 2023

@author: WSY
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions import Normal, Independent
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from jump import Jump # enviroment


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, action_dim)
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_sigma = nn.Parameter(-2*torch.ones(action_dim))
        
    def forward(self, obs):
        obs = obs.reshape(-1, self.obs_dim)
        x = self.func(obs)
        x0 = x[:, 0]
        x1 = x[:, 1]
        mu_theta = 1/2 * (x0 + 1) * math.pi/2
        mu_v = 1/2 * (x1 + 1) * 10
        sigma_theta = torch.exp(self.log_sigma[0]) * math.pi/2
        sigma_v = torch.exp(self.log_sigma[1]) * 10
        
        batch_size = obs.shape[0]
        b_mu = torch.zeros((batch_size, self.action_dim))
        b_sigma = torch.zeros((batch_size, self.action_dim))
        b_mu[:, 0] = mu_theta
        b_mu[:, 1] = mu_v
        b_sigma[:] = torch.tensor([sigma_theta, sigma_v])
        
        return {'mu':b_mu, 'sigma':b_sigma}
        
    
class PolicyGradient():
    def __init__(self):
        self.cfg = edict(
            total_step = int(1.5e5),
            learning_rate = 3e-4,
            epochs = 4,
            batch_size = 64,
            tensorboard = False,
            reward_window_size = 2000,
        )
        # Enviroment information
        self.env = Jump()
        self.obs_dim = np.array(self.env.observation_space.shape).prod()
        self.action_dim = np.array(self.env.action_space.shape).prod()
        # Policy
        self.policy = PolicyNet(self.obs_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=self.cfg.learning_rate)    
        # Log
        if self.cfg.tensorboard:
            self.tb = SummaryWriter()
    
    def sample_action(self, logits):
        dist = Normal(logits['mu'], logits['sigma'])
        dist = Independent(dist, 1)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def get_action(self, b_obs):
        logits = self.policy(b_obs)
        actions, logprobs = self.sample_action(logits)
        return actions, logprobs
    
    def get_logprob(self, b_obs, b_actions):
        logits = self.policy(b_obs)
        dist = Normal(logits['mu'], logits['sigma'])
        dist = Independent(dist, 1)
        return dist.log_prob(b_actions)
    
    def update_policy(self, b_obs, b_actions, b_rewards):
        for epoch in range(self.cfg.epochs):
            b_logpobs = self.get_logprob(b_obs, b_actions)
            loss = - b_rewards * b_logpobs
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        return loss.mean()
    
    def run(self):
        train_step = 0
        reward_window = np.zeros(self.cfg.reward_window_size)
        # Data memory
        b_obs = torch.zeros((self.cfg.batch_size, self.obs_dim))
        b_actions = torch.zeros((self.cfg.batch_size, self.action_dim))
        b_rewards = torch.zeros(self.cfg.batch_size)
        
        # Collect training data
        for step in range(self.cfg.total_step):
            obs = self.env.reset()
            obs = torch.from_numpy(obs)
            action, logprob = self.get_action(obs)
            reward = self.env.step(action.numpy().reshape(-1))
            
            if step >= self.cfg.reward_window_size and step % self.cfg.reward_window_size == 0:
                print('Step:%d, Reward mean:%f, var:%f' % (step, reward_window.mean(), reward_window.var()))
                if self.cfg.tensorboard:
                    self.tb.add_scalar('reward', reward_window.mean(), step)
            reward_window[step % self.cfg.reward_window_size] = reward
            
            index = step % self.cfg.batch_size
            b_obs[index] = obs
            b_actions[index] = action
            b_rewards[index] = torch.tensor(reward)
            
            if step >= self.cfg.batch_size - 1 and index == self.cfg.batch_size - 1:
                train_step += 1
                loss = self.update_policy(b_obs, b_actions, b_rewards)
                if self.cfg.tensorboard:
                    self.tb.add_scalar('loss', loss, train_step)
    
if __name__ == '__main__':
    pg = PolicyGradient()
    pg.run()
    torch.save(pg.policy.state_dict(), 'model.pkl')
    
    
    
    
    
    
    