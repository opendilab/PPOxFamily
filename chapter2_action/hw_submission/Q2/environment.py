import random
import numpy as np

class JumpGame:
    def __init__(self):
        self.action_v_space = (1e-5, 10)
        self.action_theta_space = (0, np.pi / 2)
        self.d1_space = (0, 1)
        self.d2_space = (0, 1)

    def reset(self, d1=None, d2=None):
        if d1 is None:
            self.d1 = random.uniform(self.d1_space[0], self.d1_space[1])
        else:
            self.d1 = d1
        if d2 is None:
            self.d2 = random.uniform(self.d2_space[0], self.d2_space[1])
        else:
            self.d2 = d2
        return self.d1, self.d2

    def step(self, action):
        v, theta = action

        v = np.clip(v, self.action_v_space[0], self.action_v_space[1])
        theta = np.clip(theta, self.action_theta_space[0], self.action_theta_space[1])

        v_x = v * np.cos(theta)
        v_y = v * np.sin(theta)
        t = self.d1 / v_x
        h = v_y * t - 0.5 * 9.8 * t * t
        if h > self.d2:
            reward = 100 - v ** 2
        else:
            reward = 0
        
        return reward