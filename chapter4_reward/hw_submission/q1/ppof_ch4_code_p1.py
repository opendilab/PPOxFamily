# pip install minigrid
from typing import Union, Tuple, Dict, List, Optional
from multiprocessing import Process
import multiprocessing as mp
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import minigrid
import gymnasium as gym
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tensorboardX import SummaryWriter
from minigrid.wrappers import FlatObsWrapper

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

train_config = dict(
    train_iter=1024,
    train_data_count=128,
    test_data_count=4096,
)

little_RND_net_config = dict(
    exp_name="little_rnd_network",
    observation_shape=2835,
    hidden_size_list=[32, 16],
    learning_rate=1e-3,
    batch_size=64,
    update_per_collect=100,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1e5,
)

small_RND_net_config = dict(
    exp_name="small_rnd_network",
    observation_shape=2835,
    hidden_size_list=[64, 64],
    learning_rate=1e-3,
    batch_size=64,
    update_per_collect=100,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1e5,
)

standard_RND_net_config = dict(
    exp_name="standard_rnd_network",
    observation_shape=2835,
    hidden_size_list=[128, 64],
    learning_rate=1e-3,
    batch_size=64,
    update_per_collect=100,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1e5,
)

large_RND_net_config = dict(
    exp_name="large_RND_network",
    observation_shape=2835,
    hidden_size_list=[256, 256],
    learning_rate=1e-3,
    batch_size=64,
    update_per_collect=100,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1e5,
)

very_large_RND_net_config = dict(
    exp_name="very_large_RND_network",
    observation_shape=2835,
    hidden_size_list=[512, 512],
    learning_rate=1e-3,
    batch_size=64,
    update_per_collect=100,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1e5,
)

class FCEncoder(nn.Module):
    def __init__(
            self,
            obs_shape: int,
            hidden_size_list,
            activation: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        layers = []
        for i in range(len(hidden_size_list) - 1):
            layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            layers.append(self.act)
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.init(x))
        x = self.main(x)
        return x

class RndNetwork(nn.Module):
    def __init__(self, obs_shape: Union[int, list], hidden_size_list: list) -> None:
        super(RndNetwork, self).__init__()
        self.target = FCEncoder(obs_shape, hidden_size_list)
        self.predictor = FCEncoder(obs_shape, hidden_size_list)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), device=torch.device('cpu')):
        self._epsilon = epsilon
        self._shape = shape
        self._device = device
        self.reset()

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        if len(self._shape) > 0:
            self._mean = np.zeros(self._shape, 'float32')
            self._var = np.ones(self._shape, 'float32')
        else:
            self._mean, self._var = 0., 1.
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        if np.isscalar(self._mean):
            return self._mean
        else:
            return torch.FloatTensor(self._mean).to(self._device)

    @property
    def std(self) -> np.ndarray:
        std = np.sqrt(self._var + 1e-8)
        if np.isscalar(std):
            return std
        else:
            return torch.FloatTensor(std).to(self._device)

class RndRewardModel():

    def __init__(self, config) -> None:  # noqa
        super(RndRewardModel, self).__init__()
        self.cfg = config

        self.tb_logger = SummaryWriter(config["exp_name"])
        self.reward_model = RndNetwork(
            obs_shape=config["observation_shape"], hidden_size_list=config["hidden_size_list"]
        ).to(device)

        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config["learning_rate"])
        self.scheduler = ExponentialLR(self.opt, gamma=0.997)

        self.estimate_cnt_rnd = 0
        if self.cfg["obs_norm"]:
            self._running_mean_std_rnd_obs = RunningMeanStd(epsilon=1e-4, device=device)

    def __del__(self):
        self.tb_logger.flush()
        self.tb_logger.close()

    def train(self, data) -> None:
        for _ in range(self.cfg["update_per_collect"]):
            train_data: list = random.sample(data, self.cfg["batch_size"])
            train_data: torch.Tensor = torch.stack(train_data).to(device)
            if self.cfg["obs_norm"]:
                # Note: observation normalization: transform obs to mean 0, std 1
                self._running_mean_std_rnd_obs.update(train_data.cpu().numpy())
                train_data = (train_data - self._running_mean_std_rnd_obs.mean) / self._running_mean_std_rnd_obs.std
                train_data = torch.clamp(
                    train_data, min=self.cfg["obs_norm_clamp_min"], max=self.cfg["obs_norm_clamp_max"]
                )

            predict_feature, target_feature = self.reward_model(train_data)
            loss = F.mse_loss(predict_feature, target_feature.detach())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.scheduler.step()

    def estimate(self, data: list) -> List[Dict]:
        """
        estimate the rnd intrinsic reward
        """

        obs = torch.stack(data).to(device)
        if self.cfg["obs_norm"]:
            # Note: observation normalization: transform obs to mean 0, std 1
            obs = (obs - self._running_mean_std_rnd_obs.mean) / self._running_mean_std_rnd_obs.std
            obs = torch.clamp(obs, min=self.cfg["obs_norm_clamp_min"], max=self.cfg["obs_norm_clamp_max"])

        with torch.no_grad():
            self.estimate_cnt_rnd += 1
            predict_feature, target_feature = self.reward_model(obs)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            self.tb_logger.add_scalar('rnd_reward/mse', mse.cpu().numpy().mean(), self.estimate_cnt_rnd)

            # Note: according to the min-max normalization, transform rnd reward to [0,1]
            rnd_reward = mse * self.cfg["reward_mse_ratio"]  #(mse - mse.min()) / (mse.max() - mse.min() + 1e-11)

            self.tb_logger.add_scalar('rnd_reward/rnd_reward_max', rnd_reward.max(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_mean', rnd_reward.mean(), self.estimate_cnt_rnd)
            self.tb_logger.add_scalar('rnd_reward/rnd_reward_min', rnd_reward.min(), self.estimate_cnt_rnd)

            rnd_reward = torch.chunk(rnd_reward, rnd_reward.shape[0], dim=0)

def training(config, train_data, test_data):
    rnd_reward_model = RndRewardModel(config=config)
    for i in range(train_config["train_iter"]):
        rnd_reward_model.train([torch.Tensor(item["last_observation"]) for item in train_data[i]])
        rnd_reward_model.estimate([torch.Tensor(item["last_observation"]) for item in test_data])

def main():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env_obs = FlatObsWrapper(env)

    # train_data = []
    # test_data = []

    # for i in range(train_config["train_iter"]):

    #     train_data_per_iter = []

    #     while len(train_data_per_iter) < train_config["train_data_count"]:
    #         last_observation, _ = env_obs.reset()
    #         terminated = False
    #         while terminated != True and len(train_data_per_iter) < train_config["train_data_count"]:
    #             action = env_obs.action_space.sample()
    #             observation, reward, terminated, truncated, info = env_obs.step(action)
    #             train_data_per_iter.append(
    #                 {
    #                     "last_observation": last_observation,
    #                     "action": action,
    #                     "reward": reward,
    #                     "observation": observation
    #                 }
    #             )
    #             last_observation = observation
    #         env_obs.close()

    #     train_data.append(train_data_per_iter)

    # while len(test_data) < train_config["test_data_count"]:
    #     last_observation, _ = env_obs.reset()
    #     terminated = False
    #     while terminated != True and len(train_data_per_iter) < train_config["test_data_count"]:
    #         action = env_obs.action_space.sample()
    #         observation, reward, terminated, truncated, info = env_obs.step(action)
    #         test_data.append(
    #             {
    #                 "last_observation": last_observation,
    #                 "action": action,
    #                 "reward": reward,
    #                 "observation": observation
    #             }
    #         )
    #         last_observation = observation
    #     env_obs.close()
    
    # 已有数据，无需重新收集
    train_data = np.load("/home/dzp/Projects/PPOxFamily/chapter4_reward/hw_submission/q1/ppof_ch4_data_p1/train_data.npy", allow_pickle= True)
    test_data = np.load("/home/dzp/Projects/PPOxFamily/chapter4_reward/hw_submission/q1/ppof_ch4_data_p1/test_data.npy", allow_pickle= True)

    p0 = Process(target=training, args=(little_RND_net_config, train_data, test_data))
    p0.start()

    p1 = Process(target=training, args=(small_RND_net_config, train_data, test_data))
    p1.start()

    p2 = Process(target=training, args=(standard_RND_net_config, train_data, test_data))
    p2.start()

    p3 = Process(target=training, args=(large_RND_net_config, train_data, test_data))
    p3.start()

    p4 = Process(target=training, args=(very_large_RND_net_config, train_data, test_data))
    p4.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
