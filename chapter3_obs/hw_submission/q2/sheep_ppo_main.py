import os
from easydict import EasyDict
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.envs import create_env_manager, DingEnvWrapper, EvalEpisodeReturnEnv
from ding.policy import PPOPolicy
from ding.worker import BaseLearner, create_serial_collector, InteractionSerialEvaluator
from ding.utils import set_pkg_seed

from sheep_env import SheepEnv
from sheep_model import SheepModel

sheep_ppo_config = dict(
    exp_name='sheep_ppo_seed0',
    env=dict(
        env_id='Sheep-v0',
        level=2,
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=15,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(),
        learn=dict(
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
            value_norm=True,
        ),
        collect=dict(
            n_sample=3200,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
)
sheep_ppo_config = EasyDict(sheep_ppo_config)
main_config = sheep_ppo_config

sheep_ppo_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
sheep_ppo_create_config = EasyDict(sheep_ppo_create_config)
create_config = sheep_ppo_create_config


def sheep_env_fn(level):
    return DingEnvWrapper(
        SheepEnv(level), cfg={'env_wrapper': [
            lambda env: EvalEpisodeReturnEnv(env),
        ]}
    )


def main(input_cfg, seed, max_env_step=int(1e7), max_train_iter=int(1e7)):
    cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    collector_env = create_env_manager(
        cfg.env.manager, [lambda: sheep_env_fn(cfg.env.level) for _ in range(cfg.env.collector_env_num)]
    )
    evaluator_env = create_env_manager(
        cfg.env.manager, [lambda: sheep_env_fn(cfg.env.level) for _ in range(cfg.env.evaluator_env_num)]
    )
    collector_env.seed(cfg.seed, dynamic_seed=False)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    obs_space = collector_env._env_ref.observation_space
    model = SheepModel(
        obs_space['item_obs'].shape[1],
        obs_space['item_obs'].shape[0],
        'TF',
        obs_space['bucket_obs'].shape[0],
        obs_space['global_obs'].shape[0]
    )
    policy = PPOPolicy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)

        learner.train(new_data, collector.envstep)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')


if __name__ == "__main__":
    main([main_config, create_config], seed=0)