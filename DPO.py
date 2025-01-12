import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)  # 输出动作的概率分布
        )

    def forward(self, x):
        return self.fc(x)

# 采样动作
def select_action(policy, state):
    # 确保 state 是 numpy 数组并有正确的形状
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32)
    else:
        state = torch.tensor(np.array(state), dtype=torch.float32)
    action_probs = policy(state)
    action = torch.multinomial(action_probs, 1).item()  # 按概率采样动作
    return action, action_probs[action]
# 计算返回值（未来奖励的累计折扣和）
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# 主训练循环
def train_policy_gradient(env, policy, optimizer, episodes, gamma=0.95):
    for episode in range(episodes):
        # 检查 reset 是否返回状态或字典
        result = env.reset()
        if isinstance(result, tuple):  # 如果 reset 返回 (state, info)
            state = result[0]
        else:
            state = result

        log_probs = []
        rewards = []

        # 一次完整的回合
        while True:
            action, log_prob = select_action(policy, state)
            result = env.step(action)
            #print(result)
            
            next_state, reward, done, _,info = result
            
            log_probs.append(torch.log(log_prob))
            rewards.append(reward)
            state = next_state

            if done:
                break

        # 计算返回值
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 标准化

        # 计算损失
        loss = -torch.sum(torch.stack(log_probs) * returns)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {sum(rewards)}")

# 使用Gym环境测试
if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v1",)  # 替换为你的环境
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(input_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    train_policy_gradient(env, policy, optimizer, episodes=250) 
    #会在200轮(也有时候不会出现)左右的时候发生一个reward的突增，reward从几百变成几万？（
