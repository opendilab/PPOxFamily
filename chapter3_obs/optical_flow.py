"""
Implementation of OpticalFlowWrapper, which can add optical flow into environment observation.
Optical flow can extract moving features between two images, which can encode temporal information into the state.
"""
import gym
import numpy as np
import cv2


class OpticalFlowWrapper(gym.Wrapper):
    """
    Overview:
    Calculate optical flow using current frame and last frame. The final output contains one channel for current frame and two channels for optical flow.
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize last frame as None.
        self.last_frame = None

    def reset(self):
        # Reset.
        obs = self.env.reset()
        # Add opticalflow channels into obs.
        state = self._get_state(obs)
        # Update last frame.
        self.last_frame = obs
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Add opticalflow channels into obs.
        state = self._get_state(obs)
        # Update last frame.
        self.last_frame = obs
        return state, reward, done, info

    def _get_state(self, obs):
        # If current frame is the first frame, then the opticalflow will be set to zeros.
        if self.last_frame is None:
            return np.stack([obs, np.zeros_like(obs), np.zeros_like(obs)])
        else:
            # Calculate opticalflow using current frame and last frame.
            flow = cv2.calcOpticalFlowFarneback(self.last_frame, obs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = (flow + 15) * (255.0 / (2 * 15))
            flow = np.round(flow).astype(np.uint8)
            # Clip the value into [0, 255).
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            # Concatenate opticalflow with original frame.
            flow = flow.transpose(2, 0, 1)
            ret = np.concatenate([[obs], flow], axis=0)
            return ret
