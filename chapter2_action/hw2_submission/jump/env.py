# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:50:22 2023

@author: WSY
"""

import math
import numpy as np
from gym import spaces

class Jump():
    def __init__(self, gravity: float = 9.8):
        self.gravity = gravity
        
        low = np.array(
            [
                0, # d1: Distance between two platforms 
                0, # d2: Height difference between two platforms
            ]
        ).astype(np.float32)
        high = np.array(
            [
                1, 
                1,
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self.state = None
        
        low = np.array(
            [
                0, # theta: Jump angle
                0, # speed: Initial speed
            ]
        ).astype(np.float32)
        high = np.array(
            [
                1/2 * math.pi,
                10,
            ]
        ).astype(np.float32)
        self.action_space = spaces.Box(low, high)
    
    def reset(self):
        self.state = self.observation_space.sample()
        return np.array(self.state)
    
    def step(self, action):
        # Since the task needs only one step action, return single reward is enough
        theta, v = list(action)
        # Restrict action in enviroment action space
        theta_min, theta_max = self.action_space.low[0], self.action_space.high[0]
        v_min, v_max = self.action_space.low[1], self.action_space.high[1]
        if theta < theta_min: theta = theta_min
        # np.tan(math.pi / 2) returns a very large number (1e16) that leads to a wrong success according to delta_y computation 
        # so that a discount factor 0.999 is applied
        if theta >= theta_max: theta = 0.999 * theta_max
        if v < v_min: v = v_min
        if v > v_max: v = v_max
        
        d1, d2 = self.state
        g = self.gravity
        delta_y = d1 * np.tan(theta) - (g * d1**2) / (2 * v**2 * np.cos(theta)**2 + 1e-8)
        if delta_y >= d2:
            reward = 100 - v**2
        else:
            # Adding negative reward as punishment performs better than 0
            # In svg image "step-reward", we can see two curves (blue and red) reach average score 93 with this reward function
            reward = 5 * np.log(- delta_y + d2) 
        return reward
    
# Test environment
if __name__ == '__main__':
    env = Jump()
    for episode in range(20):
        obs = env.reset()
        action = env.action_space.sample()
        reward = env.step(action)
        print(reward)
        
        
        
        