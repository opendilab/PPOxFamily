# Please install latest DI-engine's main branch first
from ding.bonus import PPOF


def acrobot():
    # Please install acrobot env first, `pip3 install gym`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/acrobot_zh.html) for more details
    agent = PPOF(env='acrobot', exp_name='./acrobot_demo')
    agent.train(step=int(1e5))


def metadrive():
    # Please install metadrive env first, `pip install metadrive-simulator`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/metadrive_zh.html) for more details
    agent = PPOF(env='metadrive', exp_name='./metadrive_demo')
    agent.train(step=int(1e6), context='spawn')


def minigrid_fourroom():
    # Please install minigrid env first, `pip install gym-minigrid`
    # Note: minigrid env doesn't support Windows platform
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/minigrid_zh.html) for more details
    agent = PPOF(env='minigrid_fourroom', exp_name='./minigrid_fourroom_demo')
    agent.train(step=int(3e6))


if __name__ == "__main__":
    # acrobot()
    # metadrive()
    minigrid_fourroom()
