import gym
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy, single_env_forward_wrapper
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, EvalEpisodeReturnEnv
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator_ttorch, CkptSaver, \
gae_estimator, termination_checker, PPOFStepCollector, ppof_adv_estimator, wandb_online_logger, interaction_evaluator
from ding.utils import set_pkg_seed
from dizoo.rocket.envs.rocket_env import RocketEnv
from dizoo.rocket.config.rocket_landing_ppo_config import main_config, create_config
import numpy as np
from tensorboardX import SummaryWriter
import os
import torch
from rocket_recycling.rocket import Rocket


class RocketLandingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(8, ), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(9)
        self._action_space.seed(0)  # default seed
        self.reward_range = (float('-inf'), float('inf'))


def wrapped_rocket_env(task, max_steps):
    return DingEnvWrapper(
        Rocket(task=task, max_steps=max_steps),
        cfg={'env_wrapper': [
            lambda env: RocketLandingWrapper(env),
            lambda env: EvalEpisodeReturnEnv(env),
        ]}
    )

def main():
    logging.getLogger().setLevel(logging.INFO)
    exp_name = 'rocket_landing_ppo_demo'
    main_config.exp_name = exp_name
    main_config.policy.cuda = True
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'seed'+str(111)))
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[
                lambda: wrapped_rocket_env(cfg.env.task, cfg.env.max_steps)
                for _ in range(cfg.env.collector_env_num)
            ],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[
                lambda: wrapped_rocket_env(cfg.env.task, cfg.env.max_steps)
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager
        )

        set_pkg_seed(111, use_cuda=cfg.policy.cuda)

        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        seed = 222
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=1000))
        task.use(CkptSaver(policy, save_dir=exp_name, train_freq=100))
        task.use(wandb_online_logger(exp_name, metric_list=policy.learn_mode.monitor_vars(), anonymous=True))
        task.use(termination_checker(max_env_step=int(3e6)))
        task.run()

if __name__ == "__main__":
    main()
