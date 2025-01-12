import numpy as np
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lm_env import calculate_perplexity, TextEnvironment
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),  # 输出动作概率分布
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 输出状态值
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, lam=0.95, clip_ratio=0.2):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio

    def get_action(self, state):
        
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.model(state)
        action = torch.multinomial(probs, 1, replacement=False).item()
        return action, probs[:, action].item()

    def compute_advantages(self, rewards, dones, values):
        advantages = torch.zeros_like(torch.FloatTensor(rewards))
        returns = torch.zeros_like(torch.FloatTensor(rewards))
        gae = 0
        next_value = 0

        # 逆序遍历时间步
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            returns[t] = gae + values[t]
            advantages[t] = gae
            next_value = values[t]

        return advantages, returns

    def train(self, states, actions, old_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(10):  # 每次更新多轮
            probs, values = self.model(states)
            probs_a = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = probs_a / (old_probs + 1e-8)

            # 裁剪比率
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # 值函数损失
            value_loss = (returns - values.squeeze(1)).pow(2).mean()

            # 总损失
            loss = actor_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_step(self, env, batch_size=64):
      
        states, actions, rewards, dones, values, old_probs = [], [], [], [], [], []
        
        episode_rewards = []  # 记录每个 episode 的累计奖励
        
        # 初始化环境并解包可能的额外信息
        state,_ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
        #state = np.asarray(state, dtype=np.float32)
        #print("state:",state)
        
        
        
        
        while len(states) < batch_size:
            action, prob = self.get_action(state)
            
            step_result = env.step(action)
            #print(len(step_result))   #   5
            # 解包 env.step()
            if len(step_result) == 5:  # 返回了 5 个值
                next_state, reward, done, info, _ = step_result
            elif len(step_result) == 4:  # 返回了标准 4 个值
                next_state, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step result format: {step_result}")            
                        
            next_state, reward, done, info,_ = step_result
            #print("next_state:",next_state)

            
            

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(self.model(torch.FloatTensor(state).unsqueeze(0))[1].item())
            old_probs.append(prob)
            
            
            
            if not done:
                state = next_state
            else:
              state,_ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
            

        # 计算优势和回报
        advantages, returns = self.compute_advantages(rewards, dones, values)

        # 执行训练
        self.train(states, actions, old_probs, returns, advantages)


# 测试 PPO
env = gym.make("CartPole-v1")
ppo = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
nums_episodes = 500
for episode in range(nums_episodes):
    ppo.train_step(env)
    if episode % 50 == 0:
        print(f"Episode {episode} complete.")



#######            评估

episode_rewards = []  # 记录每个 episode 的累计奖励

for episode in range(nums_episodes):
    state,_ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = ppo.get_action(state)
        state, reward, done, _,_ = env.step(action) #之前打印出来的step_result是5个值，所以这里也要改成5个值
        episode_reward += reward
    episode_rewards.append(episode_reward)

    if (episode + 1) % 10 == 0:  # 每 10 个 episode 计算平均回报
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward}")

# 绘制学习曲线
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()
