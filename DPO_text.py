import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List
from lm_env2 import calculate_perplexity,TextEnvironment


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


def select_action(policy, state, action_pool):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy(state)
    action_idx = torch.multinomial(action_probs, 1).item()  # 按概率采样动作
    action = action_pool[action_idx]  # 从预定义池中选择动作文本
    return action, action_probs[action_idx]


def compute_returns(rewards, gamma=0.95):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_policy_gradient(env, policy, optimizer, action_pool, episodes, gamma=0.95):
  #从0.99改成0.95，能多训练几轮，0.99很容易出现nan
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            action, log_prob = select_action(policy, state, action_pool)
            state, reward, done, _ = env.step(action)

            log_probs.append(torch.log(log_prob))
            rewards.append(reward)

            if done:
                break

        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        optimizer.step()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {sum(rewards):.2f}")


def main():
    # 加载 GPT-2 模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # 定义奖励函数
    reward_function = lambda lm, query, response: - torch.log(torch.tensor(1 + calculate_perplexity(lm, query, response)))
    #直接用-calculate_perplexity(lm, query, response) 负困惑度，会在几轮之后 reward出现nan,因为困惑度是exp的值，会出现很大的值
  
    # 定义生成参数
    generation_kwargs = {
        'max_new_tokens': 20,
        'do_sample': True,
        'temperature': 0.7,
        'repetition_penalty': 2.0
    }

    # 初始化环境
    env = TextEnvironment(
        model=model,
        tokenizer=tokenizer,
        max_turns=4,
        reward_fn=reward_function,
        generation_kwargs=generation_kwargs
    )

    # 定义动作池（预定义文本）
    action_pool = [
        "Hello, how are you?",
        "What is the weather today?",
        "Tell me a joke.",
        "Explain reinforcement learning.",
        "Generate a poem."
    ]
    """
    是否可以实现无动作池的自由生成？
    是可以的，但需要更复杂的设计：

    自由文本生成：
    使用 GPT-2 或类似模型作为策略网络直接生成动作。
    环境接受生成的自由文本并评估奖励。
    困难：
    动作空间极大，强化学习效率低。
    奖励函数需要额外的自然语言处理模块，评估文本质量和任务相关性"""
    
    input_size = 1024  # 环境返回的状态向量长度
    action_size = len(action_pool)

    # 定义策略网络
    policy = PolicyNetwork(input_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    #lr = 0.005时，Episode 14/20, Total Reward: nan
    #lr = 0.01时，Episode 7/20, Total Reward: nan

    # 训练策略
    train_policy_gradient(env, policy, optimizer, action_pool, episodes=20)


if __name__ == "__main__":
    main()
