"""
Using ``ding.envs.DingEnvWrapper`` to manually define the environment with corresponding wrappers.
"""
from ding.envs import DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


def wrapped_mario_env():
    return DingEnvWrapper(
        # Original environment declaration.
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), [["right"], ["right", "A"]]),
        cfg={
            'env_wrapper': [
                # Return the max value of states in adjacent 4 frames. This works like a max-pooling in temporal dimension.
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # Resize the image size into 84 x 84 and convert RGB to GREY.
                lambda env: WarpFrameWrapper(env, size=84),
                # Normalize the state value into [0, 1].
                lambda env: ScaledFloatFrameWrapper(env),
                # Stack adjacent 4 frames together to be one state.
                lambda env: FrameStackWrapper(env, n_frames=4),
                # Evaluate episode return.
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )
