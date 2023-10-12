from ding.bonus import PPOF


def evogym_carrier():
    agent = PPOF(env='evogym_carrier', exp_name='./evogym_carrier_deploy')

    agent.deploy(enable_save_replay=True, ckpt_path='evogym_carrier_demo/ckpt/eval.pth.tar')

if __name__ == "__main__":
    evogym_carrier()