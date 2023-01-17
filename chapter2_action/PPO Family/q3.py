# -*- coding: utf-8 -*-
# @Time    : 2023-01-17 22:04
# @Author  : 吴佳杨

from ding.bonus import PPOF


def lunarlander_continuous():
    # Please install lunarlander env first, `pip3 install box2d`
    agent = PPOF(env='lunarlander_continuous', exp_name='./lunarlander_continuous_demo', seed=314)
    agent.train(step=int(1e5))
    # Batch (Vectorized) evaluation
    agent.batch_evaluate(env_num=4, n_evaluator_episode=8)