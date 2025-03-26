import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math

class Jump_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Jump_Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([np.pi, 10]), dtype=np.double)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, ), dtype=np.double)
        self.state = None
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, state, action):
        g = 10
        reward = 0
        done = 0
        info = {}
        d = state
        Vx = action[0][1]*math.cos(action[0][0])
        Vy = action[0][1]*math.sin(action[0][0])
        t = d[0][0]/Vx
        dif_y = Vy*t-(1/2)*g*(t**2)
        if dif_y > d[0][1]:
            reward = 100-action[0][1]**2
        else:
            reward = 0
        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(low=0, high=1, size=(4,2))
        return np.array(self.state)

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
