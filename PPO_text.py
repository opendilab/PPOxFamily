import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Callable, Dict, Optional, Tuple
from lm_env2 import calculate_perplexity, TextEnvironment
import traceback
import argparse
# 定义一个Actor-Critic模型，用于 PPO 算法
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor部分：输入观测，输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(128, action_dim),  # 隐藏层到动作空间的线性变换
            #nn.Softmax(dim=-1)  # softmax 归一化，输出动作概率分布 .softmax会出现Nan的情况，下面select_action函数中使用了softmax
        )
        # Critic部分：输入观测，输出状态值函数（单一标量）
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(128, 1)  # 隐藏层到标量输出的线性变换
        )

    def forward(self, x, attention_mask=None):
        # 如果需要，可以利用 attention_mask 做加权处理
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)  # 对输入加权
        probs = self.actor(x)  # 动作概率
        value = self.critic(x)  # 状态值
        print("Probs(when forward get called):", probs)  # 打印 probs 值进行调试
        print("Value(when forward get called):", value)  # 打印 value 值进行调试
        return probs, value

# PPO（近端策略优化）代理类
class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=0.003, gamma=0.99, lam=0.95, epsilon=0.2):
        self.gamma = gamma  # 折扣因子，用于计算回报
        self.lam = lam  # GAE（广义优势估计）中的 lambda 参数
        self.epsilon = epsilon  # PPO 中的截断范围

        # 初始化 Actor-Critic 模型
        self.model = ActorCritic(obs_dim, action_dim)
        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    #实际上并不需要剪裁
    def pad_or_crop(self, tensor, target_length):
        """将 tensor 填充或裁剪为目标长度"""
        current_length = tensor.shape[0]
        if current_length > target_length:
            result = tensor[:target_length]  # 裁剪
        elif current_length < target_length:
            padding = torch.zeros(target_length - current_length)  # 填充
            result = torch.cat([tensor, padding])
        else:
            result = tensor

        return result  # 确保返回值是 Tensor

    def select_action(self, obs):
        """
        根据当前状态选择动作
        """
        obs = torch.FloatTensor(obs).unsqueeze(0)  # 将观测转换为 tensor
        logits, _ = self.model(obs)  # 获取 logits（未经过 softmax）
        #print("Logits:", logits)  # 打印 logits 值进行调试
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # 归一化，减去最大值，防止数值不稳定
        # 检查 logits 是否包含无效值
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

        probs = torch.softmax(logits, dim=-1)  # 使用 softmax 计算概率分布
        probs = probs + 1e-8  # 添加平滑值

        assert not torch.isnan(probs).any(), "Probs contain NaN values"
        assert not torch.isinf(probs).any(), "Probs contain Inf values"
        assert (probs >= 0).all(), "Probs contain negative values"

        action = torch.multinomial(probs, 1).item()  # 按概率分布采样动作
        return action, probs[:, action].item()  # 返回动作和概率

    def compute_advantages(self, rewards, dones, values):
        """
        使用 GAE 计算优势函数和回报
        """
        advantages, returns = [], []  # 初始化优势和回报
        gae = 0  # 初始化广义优势估计
        next_value = 0  # 下一个状态的值
        # 反向遍历奖励、done标志和值函数
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            delta = reward + self.gamma * next_value * (1 - done) - value  # TD误差
            gae = delta + self.gamma * self.lam * (1 - done) * gae  # 计算GAE
            returns.insert(0, gae + value)  # 回报 = GAE + 值函数
            advantages.insert(0, gae)  # 记录优势
            next_value = value  # 更新下一个值函数
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def train(self, states, actions, old_probs, advantages, returns):
        """
        使用 PPO 算法训练模型
        """
        for _ in range(10):  # 对每批数据进行多次更新
            probs, values = self.model(states)  # 前向传播获取动作概率和状态值
            
            probs_a = probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # 获取执行动作的概率
            
            ratio = probs_a / (old_probs + 1e-8)  # 计算重要性采样比率
            # PPO 的截断机制，限制比率的变化范围
            clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            # 计算 Actor 损失
            actor_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            """
            log_ratio = torch.log(probs_a + 1e-8) - torch.log(old_probs + 1e-8)  # 计算对数比率
            clip_log_ratio = torch.clamp(
                log_ratio,
                torch.log(torch.tensor(1 - self.epsilon, dtype=torch.float32)),
                torch.log(torch.tensor(1 + self.epsilon, dtype=torch.float32))
            )# 截断对数比率
            actor_loss = -torch.min(log_ratio * advantages, clip_log_ratio * advantages).mean()
            """
            # 计算 Critic 损失
            value_loss = (returns - values.squeeze(1)).pow(2).mean()
            # 总损失 = Actor 损失 + Critic 损失
            loss = actor_loss + value_loss

            # 反向传播并更新参数
            self.optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def train_step(self, env, batch_size=64):
        #traceback.print_stack()  # 打印调用堆栈
        """
        从环境中收集数据并进行训练
        """
        # 初始化数据存储
        states, actions, rewards, dones, values, old_probs = [], [], [], [], [], []
        obs, _ = env.reset()  # 重置环境，获取初始状态
        
        done = False  # 初始化终止标志
        
        while not done:
            action, prob = ppo.select_action(obs)  # PPO 选择动作
            query = tokenizer.decode([action])  # 动作解码为 query（例如 Token 转字符串）
            obs, reward, done, info = env.step(query)  # 将 query 传递给环境，获取下一个状态、奖励和终止标志
        print("Obs:", obs)
        #print("train_step() was called.")  # 添加调试信息
        
        #全连接网络内部可能出现nan 导致logits出现nan,以下的处理感觉没有必要
        """
        assert not np.isnan(obs).any(), "Obs contains NaN values"
        assert not np.isinf(obs).any(), "Obs contains Inf values"
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)  # 替换 NaN 和 Inf
        obs = np.clip(obs, -1e6, 1e6)  # 限制值的范围
        """
        # 收集批量数据
        while len(states) < batch_size:
            action, prob = self.select_action(obs)  # 选择动作
            next_obs, reward, done, _ = env.step(action)  # 执行动作，获取下一个状态和奖励
            states.append(obs)  # 记录当前状态
            actions.append(action)  # 记录动作
            rewards.append(reward)  # 记录奖励
            dones.append(done)  # 记录是否终止
            # 使用 Critic 获取当前状态的值函数
            values.append(self.model(torch.FloatTensor(obs).unsqueeze(0))[1].item())
            old_probs.append(prob)  # 记录动作的概率

            obs = next_obs if not done else env.reset()[0]  # 如果未终止，则更新状态，否则重置
            
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)  # 替换 NaN 和 Inf
            obs = np.clip(obs, 0, 5e4)  # 限制值的范围
            
        # 计算优势和回报
        advantages, returns = self.compute_advantages(rewards, dones, values)
        
        #以下是在对齐张量，方便裁剪和填充
        # 获取所有状态的最大长度
        max_length = max(state.shape[0] for state in states)

        # 对所有状态进行填充或裁剪
        states = [self.pad_or_crop(state, max_length) for state in states]

        # 转换为 PyTorch Tensor
        states = [torch.FloatTensor(state) if not isinstance(state, torch.Tensor) else state for state in states]

        states_tensor = torch.stack(states)  # 堆叠为二维张量
        #print("state:", states_tensor)  # 打印当前状态（调试用）
        #以上是在对齐张量，方便裁剪和填充
        
        # 使用 PPO 算法训练模型
        self.train(
            torch.FloatTensor(states_tensor),  # 状态转换为 Tensor
            torch.LongTensor(actions),  # 动作转换为 Tensor
            torch.FloatTensor(old_probs),  # 动作概率转换为 Tensor
            advantages,  # 优势
            returns  # 回报
        )
        



# 定义奖励函数：负困惑度
def reward_function(model, query, response):
    return -calculate_perplexity(model, query, response)  # 计算困惑度的负值作为奖励



if __name__ == '__main__':
    
    # 添加参数支持
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument('--train', action='store_true', help="Run training")
    args = parser.parse_args()


        
            # 加载 GPT-2 模型和 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # 加载 GPT-2 的分词器
    tokenizer.pad_token = tokenizer.eos_token  # 将 padding token 设置为 eos token
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # 加载 GPT-2 模型
    # 定义生成参数
    generation_kwargs = {
        'max_new_tokens': 20,  # 最大生成的 Token 数
        'do_sample': True,  # 使用采样生成
        'temperature': 0.7,  # 温度参数，控制生成的多样性
        'repetition_penalty': 2.0  # 重复惩罚
    }

    # 初始化文本环境
    env = TextEnvironment(
        model=model,  # GPT-2 模型
        tokenizer=tokenizer,  # 分词器
        reward_fn=reward_function,  # 奖励函数
        max_turns=3,  # 最大对话轮数
        generation_kwargs=generation_kwargs  # 文本生成参数
    )

    # 初始化 PPO 代理
    ppo = PPOAgent(
        obs_dim=8,  # 观测空间大小，与 Tokenizer vocab 相符 
        action_dim=tokenizer.vocab_size  # 动作空间大小，即 Tokenizer 词汇表大小
    )


    # 开始训练 PPO
    if args.train:
        for episode in range(10):  # 训练 500 次
            ppo.train_step(env)  # 每次训练一个批次的数据
            torch.save(ppo.model.state_dict(), 'ppo_actor_critic.pth')  # 保存模型参数
            print("Model saved.")
            """
            if episode % 50 == 0:  # 每 50 次输出进度
                print(f"Episode {episode} complete.")  # 输出当前训练完成的轮数"""
            print(f"Episode {episode} complete.")  # 重复打印（意外多加了一次）
    else:
        print("Training script loaded. Use --train to start training.")
            
    