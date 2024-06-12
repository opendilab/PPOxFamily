# Please install latest DI-engine's main branch first
# And we will release DI-engine v0.4.6 version with stable and tuned configuration of these demos.
# yyx: 参考https://github.com/opendilab/DI-engine的installation文档，下载Development Version，即可获得ding.bonus
import argparse
from ding.bonus import PPOF

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='lunarlander_discrete')
args = parser.parse_args()


def lunarlander_discrete():
    # Please install lunarlander env first, `pip3 install box2d`
    agent = PPOF(env='lunarlander_discrete', exp_name='./lunarlander_discrete_demo')
    agent.train(step=int(1e5))
    # Classic RL interaction loop and save replay video
    agent.deploy(enable_save_replay=True)


def lunarlander_continuous():
    # Please install lunarlander env first, `pip3 install box2d`
    agent = PPOF(env='lunarlander_continuous', exp_name='./lunarlander_continuous_demo', seed=314)
    agent.train(step=int(1e5))
    # Batch (Vectorized) evaluation
    agent.batch_evaluate(env_num=4, n_evaluator_episode=8)


def rocket_landing():
    # Please install rocket env first, `pip3 install git+https://github.com/nighood/rocket-recycling@master#egg=rocket_recycling`
    # yyx: 安装了develop version后不知道为什么conda环境的dizoo路径下没用rocket，从github手动复制这个文件夹进去
    agent = PPOF(env='rocket_landing', exp_name='./rocket_landing_demo')
    agent.train(step=int(5e6), context='spawn')


def drone_fly():
    # Please install gym_pybullet_drones env first, `pip3 install git+https://github.com/zjowowen/gym-pybullet-drones@master`
    agent = PPOF(env='drone_fly', exp_name='./drone_fly_demo')
    agent.train(step=int(5e6))


def hybrid_moving():
    # Please install gym_hybrid env first, refer to the doc `https://di-engine-docs.readthedocs.io/zh_CN/latest/13_envs/gym_hybrid_zh.html`
    agent = PPOF(env='hybrid_moving', exp_name='./hybrid_moving_demo')
    agent.train(step=int(5e6))


if __name__ == "__main__":
    # You can select and run your favorite demo
    if args.env == 'lunarlander_discrete':
        lunarlander_discrete()
    elif args.env == 'lunarlander_continuous':
        lunarlander_continuous()
    elif args.env == 'rocket_landing':
        rocket_landing()  # 接下来做这个
    elif args.env == 'drone_fly':
        drone_fly()
    elif args.env == 'hybrid_moving':
        hybrid_moving()
