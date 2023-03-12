# Please install latest DI-engine's main branch first
from ding.bonus import PPOF


def acrobot():
    # Please install acrobot env first, `pip3 install gym`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/acrobot_zh.html) for more details
    agent = PPOF(env='acrobot', exp_name='output/ch4/acrobot_demo')
    agent.train(step=int(1e5))

def acrobot_deploy():
    # Please install acrobot env first, `pip3 install gym`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/acrobot_zh.html) for more details
    agent = PPOF(env='acrobot', exp_name='output/ch4/acrobot_demo')
    agent.deploy(enable_save_replay=True)


def metadrive():
    # Please install metadrive env first, `pip install metadrive-simulator`
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/metadrive_zh.html) for more details
    agent = PPOF(env='metadrive', exp_name='output/ch4/metadrive_demo')
    agent.train(step=int(1e6), context='spawn')

def metadrive_deploy():
    agent = PPOF(env='metadrive', exp_name='output/ch4/metadrive_demo')
    agent.deploy(enable_save_replay=True)

def metadrive_install_test():
    from metadrive import MetaDriveEnv
    env = MetaDriveEnv()
    obs = env.reset()
    print(obs.shape)  # 输出 (259,)

def minigrid_fourroom():
    # Please install minigrid env first, `pip install gym-minigrid`
    # Note: minigrid env doesn't support Windows platform
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/minigrid_zh.html) for more details
    agent = PPOF(env='minigrid_fourroom', exp_name='output/ch4/minigrid_fourroom')
    agent.train(step=int(3e6))


def minigrid_fourroom_deploy():
    agent = PPOF(env='minigrid_fourroom', exp_name='output/ch4/minigrid_fourroom')
    agent.deploy(enable_save_replay=True)




if __name__ == "__main__":
    # acrobot()
    # acrobot_deploy()
    # metadrive_install_test()
    metadrive()
    # metadrive_deploy()
    # minigrid_fourroom()