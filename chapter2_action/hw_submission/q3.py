# Please install latest DI-engine's main branch first
# And we will release DI-engine v0.4.6 version with stable and tuned configuration of these demos.
from ding.bonus import PPOF


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
    lunarlander_discrete()
    # lunarlander_continuous()
    # rocket_landing()
    # drone_fly()
    # hybrid_moving()
