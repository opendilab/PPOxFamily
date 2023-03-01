"""
Using ``ding.envs.DingEnvWrapper`` to manually define the environment with corresponding wrappers.
We also include implementation of ``OpticalFlowWrapper`` , which can add optical flow into environment observation.
Optical flow can extract moving features between two images, which can encode temporal information into the state.
"""
import cv2
import gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from ding.envs import DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv


class OpticalFlowWrapper(gym.Wrapper):
    """
    **Overview**:
        Calculate optical flow using current frame and last frame. The final output contains one channel for current frame and two channels for optical flow.
        <link https://en.wikipedia.org/wiki/Optical_flow link>
    """

    def __init__(self, env):
        """
        **Overview**:
            Initialize wrapper and related member variable.
        """
        super().__init__(env)
        # Initialize last frame as None.
        self.last_frame = None

    # delimiter
    def reset(self):
        """
        **Overview**:
            Reset environment and process obs.
        """
        # Reset environment.
        obs = self.env.reset()
        # Add optical flow channels into obs.
        state = self._process_obs(obs)
        # Update last frame.
        self.last_frame = obs
        return state

    # delimiter
    def step(self, action):
        """
        **Overview**:
            Execute one env step, process obs and return interaction result.
        """
        obs, reward, done, info = self.env.step(action)
        # Add optical flow channels into obs.
        state = self._process_obs(obs)
        # Update last frame.
        self.last_frame = obs
        return state, reward, done, info

    # delimiter
    def _process_obs(self, obs):
        """
        **Overview**:
            Process obs, i.e., add optical flow information to original image.
        """
        # If current frame is the first frame, then the optical flow will be set to zeros.
        if self.last_frame is None:
            return np.stack([obs, np.zeros_like(obs), np.zeros_like(obs)])

        # Calculate optical flow using current frame and last frame.
        flow = cv2.calcOpticalFlowFarneback(self.last_frame, obs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = (flow + 15) * (255.0 / (2 * 15))
        flow = np.round(flow).astype(np.uint8)
        # Clip the value into [0, 255).
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0
        # Reshape optical flow array from [H, W, 2] to [2, H, W]
        flow = flow.transpose(2, 0, 1)
        # Concatenate optical flow with original frame.
        return np.concatenate([[obs], flow], axis=0)


# delimiter
def wrapped_mario_env():
    """
    **Overview**:
        Wrap mario environment with various wrappers.
    """
    # Original environment declaration.
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    return DingEnvWrapper(
        # Restrict action space to 2 discrete action: right and jump (A).
        JoypadSpace(env, [["right"], ["right", "A"]]),
        cfg={
            'env_wrapper': [
                # Return the max value of states in adjacent 4 frames. This works like a max-pooling in temporal dimension.
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # Resize (bilinear interpolation) the image size into 84 x 84 and convert RGB to GREY.
                lambda env: WarpFrameWrapper(env, size=84),
                # Normalize the state value into [0, 1]. $$scaled\_x = \frac {x - \min(x)} {\max(x) - \min (x)} $$
                lambda env: ScaledFloatFrameWrapper(env),
                # Stack adjacent 4 frames together to be one state.
                lambda env: FrameStackWrapper(env, n_frames=4),
                # Calculate episode return for evaluation.
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )


# delimiter
def wrapped_mario_env_optical():
    """
    **Overview**:
        Wrap mario environment with optical flow wrapper.
    """
    # Original environment declaration.
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    return DingEnvWrapper(
        # Restrict action space to 2 discrete action: right and jump (A).
        JoypadSpace(env, [["right"], ["right", "A"]]),
        cfg={
            'env_wrapper': [
                # Return the max value of states in adjacent 4 frames. This works like a max-pooling in temporal dimension.
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # Resize (bilinear interpolation) the image size into 84 x 84 and convert RGB to GREY.
                lambda env: WarpFrameWrapper(env, size=84),
                # Add optical flow information.
                lambda env: OpticalFlowWrapper(env),
                # Normalize the state value into [0, 1]. $$scaled\_x = \frac {x - \min(x)} {\max(x) - \min (x)} $$
                lambda env: ScaledFloatFrameWrapper(env),
                # Calculate episode return for evaluation.
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )


# delimiter
def test_wrapper():
    """
    **Overview**:
        Test two types wrappers and check output obs shape.
    """
    # Test environment with stacked frames.
    env = wrapped_mario_env()
    obs = env.reset()
    assert obs.shape == (4, 84, 84)
    # Test environment with optical flow.
    env = wrapped_mario_env_optical()
    obs = env.reset()
    assert obs.shape == (3, 84, 84)


if __name__ == "__main__":
    test_wrapper()
