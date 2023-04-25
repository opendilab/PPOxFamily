import numpy as np
import torch


def get_agent_id_feature(agent_id, agent_num):
    agent_id_feature = torch.zeros(agent_num)
    agent_id_feature[agent_id] = 1
    return agent_id_feature


def get_movement_feature():
    # for simplicity, we use random movement feature here
    movement_feature = torch.randint(0, 2, (8, ))
    return movement_feature


def get_own_feature():
    # for simplicity, we use random own feature here
    return torch.randn(10)


def get_ally_visible_feature():
    # this function only return the visible feature of one ally
    # for simplicity, we use random tensor as ally visible feature while zero tensor as ally invisible feature
    if np.random.random() > 0.5:
        ally_visible_feature = torch.randn(4)
    else:
        ally_visible_feature = torch.zeros(4)
    return ally_visible_feature


def get_enemy_visible_feature():
    # this function only return the visible feature of one enemy
    # for simplicity, we use random tensor as enemy visible feature while zero tensor as enemy invisible feature
    if np.random.random() > 0.8:
        enemy_visible_feature = torch.randn(4)
    else:
        enemy_visible_feature = torch.zeros(4)
    return enemy_visible_feature


def get_ind_global_state(agent_id, ally_agent_num, enemy_agent_num):
    # You need to implement this function
    own_feature = get_own_feature()
    ally_visible_feature = torch.cat([get_ally_visible_feature() for _ in range(ally_agent_num-1)])
    enemy_visible_feature = torch.cat([get_enemy_visible_feature() for _ in range(enemy_agent_num)])
    movement_feature = get_movement_feature()
    agent_id_feature = get_agent_id_feature(agent_id,ally_agent_num+enemy_agent_num)
    return torch.cat([own_feature,ally_visible_feature,enemy_visible_feature,movement_feature,agent_id_feature])


def get_ep_global_state(agent_id, ally_agent_num, enemy_agent_num):
    # In many multi-agent environments such as SMAC, the global state is the simplified version of the combination
    # of all the agent's independent state, and the concrete implementation depends on the characteris of environment.
    # For simplicity, we use random feature here.
    ally_center_feature = torch.randn(8)
    enemy_center_feature = torch.randn(8)
    return torch.cat([ally_center_feature, enemy_center_feature])


def get_as_global_state(agent_id, ally_agent_num, enemy_agent_num):
    # You need to implement this function
    ind_global_state = get_ind_global_state(agent_id, ally_agent_num, enemy_agent_num)
    ep_global_state = get_ep_global_state(agent_id, ally_agent_num, enemy_agent_num)
    return torch.cat([ind_global_state,ep_global_state])


def test_global_state():
    ally_agent_num = 3
    enemy_agent_num = 5
    # get independent global state, which usually used in decentralized training
    for agent_id in range(ally_agent_num):
        ind_global_state = get_ind_global_state(agent_id, ally_agent_num, enemy_agent_num)
        print(ind_global_state)
        exit(0)
        assert isinstance(ind_global_state, torch.Tensor)
    # get environment provide global state, which is the same for all agents, used in centralized training
    for agent_id in range(ally_agent_num):
        ep_global_state = get_ep_global_state(agent_id, ally_agent_num, enemy_agent_num)
        assert isinstance(ep_global_state, torch.Tensor)
    # get naive agent-specific global state, which is the specific for each agent, used in centralized training
    for agent_id in range(ally_agent_num):
        as_global_state = get_as_global_state(agent_id, ally_agent_num, enemy_agent_num)
        assert isinstance(as_global_state, torch.Tensor)


if __name__ == "__main__":
    test_global_state()