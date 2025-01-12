import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from stable_baselines3 import PPO #PPO作为专家模型
import numpy as np

# Step 1: Define a dataset class for expert data
class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        if isinstance(states, list):
            states = np.array(states)  # 合并为二维np数组
        if isinstance(actions, list):
            actions = np.array(actions)  # 转换为一维np数组
        
        # 转换为 PyTorch 张量
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# Step 2: Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

# Step 3: Collect expert data using a pre-trained PPO model
def collect_expert_data(env, expert_model, num_episodes):
    """
    收集专家数据，生成状态和动作对。
    
    Args:
        env: 环境对象。
        expert_model: 专家模型。
        num_episodes: 收集数据的回合数。
    
    Returns:
        states: 状态列表。
        actions: 动作列表。
    """
    states, actions = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()  # 解包，只获取状态
        done = False
        while not done:
            action, _ = expert_model.predict(obs)  # 使用专家模型预测动作
            
            states.append(obs)  # 记录状态
            actions.append(action)  # 记录动作
            #print(env.step(action))
            obs, reward , done, _,info = env.step(action)  # 采取动作，更新环境状态
    """
    print("Type of states:", type(states))
    print("Type of actions:", type(actions))
    print("Sample of states:", states[:3] if len(states) > 0 else "Empty")
    print("Sample of actions:", actions[:3] if len(actions) > 0 else "Empty")
    """
    return states, actions


# Step 4: Train the policy network using SFT
def train_sft(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for states, actions in dataloader:
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Step 5: Evaluate the trained policy
def evaluate_policy(env, model, num_episodes=5):
    model.eval()
    total_rewards = []
    for _ in range(num_episodes):
        obs,_ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                
                
                logits = model(torch.tensor(obs, dtype=torch.float32))  # 将obs从numpy转换为tensor
                action = torch.argmax(logits).item()
            obs, reward, done, _,info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")

if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3 import PPO

    # 创建 CartPole 环境
    env = gym.make("CartPole-v1")

    # 加载预训练的 PPO 模型作为专家
    expert_model = PPO("MlpPolicy", env, verbose=0)
    expert_model.learn(total_timesteps=1000)  # 预训练专家

    # 收集专家数据
    states, actions = collect_expert_data(env, expert_model, num_episodes=50)

    # 确保 states 和 actions 不为空
    if not states or not actions:
        raise ValueError("Collected data is empty. Check expert model and environment configuration.")

    # 使用专家数据训练 SFT 模型
    dataset = ExpertDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PolicyNetwork(state_dim, action_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_sft(model, dataloader, optimizer, criterion, epochs=30)
    evaluate_policy(env, model)


