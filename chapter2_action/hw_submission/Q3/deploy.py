from ding.bonus import PPOF


def rocket_landing():
    # Please install rocket env first, `pip3 install git+https://github.com/nighood/rocket-recycling@master#egg=rocket_recycling`
    agent = PPOF(env='rocket_landing', exp_name='./rocket_landing_deploy')
    agent.policy.enable_mode.remove('learn')
    agent.deploy(enable_save_replay=True, ckpt_path='rocket_landing_ppo_demo/ckpt/eval.pth.tar')

if __name__ == "__main__":
    rocket_landing()