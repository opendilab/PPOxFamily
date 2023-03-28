"""
使用 ``ding.envs.DingEnvWrapper`` 来手动定义环境以及使用的包裹器（wrapper）。
本示例中还包括了光流包裹器的具体实现。光流的作用是提取视频相邻两帧之间的运动信息，可以建模环境的时序信息。光流包裹器的作用就是将光流信息加入到环境的 obs 中。
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


    def __init__(self, env):
        """
        **OpticalFlowWrapper 定义概述**:
            对于相邻的两帧之间计算光流。最终的输出包括当前帧（一个通道），以及光流信息（两个通道）。具体关于光流的信息，可以参考： <link https://en.wikipedia.org/wiki/Optical_flow link>
        """
        super().__init__(env)
        # 初始化前一帧为 None。
        self.last_frame = None

    # delimiter
    def reset(self):
        """
        **reset 函数功能概述**:
            重置环境，返回重置后的第一个 obs。
        """
        # 重置环境
        obs = self.env.reset()
        self.last_frame = None
        # 将光流信息添加到 obs 中。
        state = self._process_obs(obs)
        # 更新上一帧
        self.last_frame = obs
        return state

    # delimiter
    def step(self, action):
        """
        **step 函数功能概述**:
            执行动作，返回经过光流信息添加后的下一个 obs。
        """
        obs, reward, done, info = self.env.step(action)
        # 对 obs 添加光流信息
        state = self._process_obs(obs)
        # 更新上一帧。
        self.last_frame = obs
        return state, reward, done, info

    # delimiter
    def _process_obs(self, obs):
        """
        **_process_obs 函数功能概述**:
            向 obs 中添加光流信息。
        """
        # 如果当前帧是第一帧，即上一帧为 None，则直接返回，让光流通道全部等于 0。
        if self.last_frame is None:
            return np.stack([obs, np.zeros_like(obs), np.zeros_like(obs)])

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(self.last_frame, obs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = (flow + 15) * (255.0 / (2 * 15))
        flow = np.round(flow).astype(np.uint8)
        # 将所有的光流值截断到 [0, 255).
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0
        # 对输出的通道维度进行调整，从 [H, W, 2] 到 [2, H, W]
        flow = flow.transpose(2, 0, 1)
        # 将光流信息与原始图像拼接在一起。
        return np.concatenate([[obs], flow], axis=0)


# delimiter
def wrapped_mario_env():
    """
    **wrapped_mario_env 函数功能概述**:
        使用叠帧包裹器，以及多种其它包裹器，对马里奥环境进行包裹。
    """
    # 基础马里奥环境初始化。
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    return DingEnvWrapper(
        # 限制动作空间为“向右”、“向右且起跳”。
        JoypadSpace(env, [["right"], ["right", "A"]]),
        cfg={
            'env_wrapper': [
                # 返回相邻四帧的最大值。这个包裹器的作用类似于在时间维度上进行了一次最大池化。
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # 使用双线性插值对图像的尺寸进行修改，变成 84 x 84 并且从三通道的 RGB 转变为单通道的灰度图像。
                lambda env: WarpFrameWrapper(env, size=84),
                # 将值归一化，具体使用的公式是： $$scaled\_x = \frac {x - \min(x)} {\max(x) - \min (x)} $$
                lambda env: ScaledFloatFrameWrapper(env),
                # 将相邻四帧叠在一起形成一个 obs。
                lambda env: FrameStackWrapper(env, n_frames=4),
                # 在 evaluate 的时候计算最终的累计回报。
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )


# delimiter
def wrapped_mario_env_optical():
    """
    **wrapped_mario_env_optical 函数功能概述**:
        使用光流包裹器，以及多种其它包裹器，对马里奥环境进行包裹。
    """
    # 基础马里奥环境初始化。
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    return DingEnvWrapper(
        # 限制动作空间为“向右”、“向右且起跳”。
        JoypadSpace(env, [["right"], ["right", "A"]]),
        cfg={
            'env_wrapper': [
                # 返回相邻四帧的最大值。这个包裹器的作用类似于在时间维度上进行了一次最大池化。
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # 使用双线性插值对图像的尺寸进行修改，变成 84 x 84 并且从三通道的 RGB 转变为单通道的灰度图像。
                lambda env: WarpFrameWrapper(env, size=84),
                # 添加光流信息。
                lambda env: OpticalFlowWrapper(env),
                # 将值归一化，具体使用的公式是： $$scaled\_x = \frac {x - \min(x)} {\max(x) - \min (x)} $$
                lambda env: ScaledFloatFrameWrapper(env),
                # 在 evaluate 的时候计算最终的累计回报。
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )


# delimiter
def test_wrapper():
    """
    **test_wrapper 函数功能概述**:
        测试两种类型的包裹器，分别确认它们输出的 obs 维度。
    """
    # 测试使用了叠帧方案的包裹器。
    env = wrapped_mario_env()
    obs = env.reset()
    assert obs.shape == (4, 84, 84)
    # 测试使用了光流方案的包裹器。
    env = wrapped_mario_env_optical()
    obs = env.reset()
    assert obs.shape == (3, 84, 84)


if __name__ == "__main__":
    test_wrapper()
