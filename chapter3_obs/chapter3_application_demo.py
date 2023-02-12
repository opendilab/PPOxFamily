# Please install latest DI-engine's main branch first
from ding.bonus import PPOF


def bipedalwalker():
    # Please install bipedalwalker env first, `pip3 install box2d`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/bipedalwalker_zh.html) for more details
    agent = PPOF(env='bipedalwalker', exp_name='./bipedalwalker_demo')
    agent.train(step=int(1e6))
    # Classic RL interaction loop and save replay video
    agent.deploy(enable_save_replay=True)


def evogym_carrier():
    # Please install evogym env first, refer to its doc (https://github.com/EvolutionGym/evogym#installation)
    # Or you can use our provided docker (opendilab/ding:nightly-evogym)
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/Evogym_zh.html) for more details
    agent = PPOF(env='evogym_carrier', exp_name='./evogym_carrier_demo')
    agent.train(step=int(1e6))


def mario():
    # Please install mario env first, `pip install gym-super-mario-bros`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/gym_super_mario_bros_zh.html) for more details
    agent = PPOF(env='mario', exp_name='./mario_demo')
    agent.train(step=int(3e6))


def di_sheep():
    # Please prepare di_sheep env and modelfirst, you can copy the env and model file to to current directory,
    # which are placed in https://github.com/opendilab/DI-sheep/blob/master/service
    from sheep_env import SheepEnv
    from sheep_model import SheepModel
    env = SheepEnv(level=9)
    obs_space = env.observation_space
    model = SheepModel(
        item_obs_size=obs_space['item_obs'].shape[1],
        item_num=obs_space['item_obs'].shape[0],
        item_encoder_type='TF',
        bucket_obs_size=obs_space['bucket_obs'].shape[0],
        global_obs_size=obs_space['global_obs'].shape[0],
        ttorch_return=True,
    )
    agent = PPOF(env='di_sheep', exp_name='./di_sheep_demo', model=model)
    agent.train(step=int(1e6))


def procgen_bigfish():
    # Please install procgen env first, `pip install procgen`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/procgen_zh.html) for more details
    agent = PPOF(env='procgen_bigfish', exp_name='./procgen_bigfish_demo')
    agent.train(step=int(1e7))


if __name__ == "__main__":
    # You can select and run your favorite demo
    bipedalwalker()
    # evogym_carrier()
    # mario()
    # di_sheep()
    # procgen_bigfish()
