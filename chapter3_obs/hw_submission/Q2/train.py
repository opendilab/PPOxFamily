from ding.bonus import PPOF

def evogym_carrier():
    # Please install evogym env first, refer to its doc (https://github.com/EvolutionGym/evogym#installation)
    # Or you can use our provided docker (opendilab/ding:nightly-evogym)
    # You can refer to the env doc (https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/Evogym_zh.html) for more details
    agent = PPOF(env='evogym_carrier', exp_name='./evogym_carrier_demo')
    agent.train(step=int(1e6))

if __name__ == '__main__':
    evogym_carrier()