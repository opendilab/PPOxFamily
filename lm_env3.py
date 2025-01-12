"""
PyTorch implementation of Language Model Environment.

There are two main components in this documentation:
- We use GPT-2 as the base language model and construct an environment.
- We make a demonstration of this environment and users can type prompts in the command line to interact with the language model.
"""
#########################################################
#lm_env2.py与lm_env.py的区别在于lm_env2.py中不使用用户交互环境，而是直接用自动生成的query进行交互
#用自动生成的query进行交互
#用自动生成的query进行交互
#用自动生成的query进行交互
#########################################################
import torch
import torch.nn.functional as F
import gym
from typing import Callable, Optional, Dict, Tuple
# For more information about GPT2, please refer to this doc: <link https://huggingface.co/transformers/v3.0.2/model_doc/gpt2.html#gpt2lmheadmodel link>.
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PPO_text import ActorCritic
obs_dim = 8

def calculate_perplexity(model: ActorCritic, query: torch.Tensor, response: torch.Tensor, obs_dim: int = 8) -> float:
    """
    Calculate the perplexity of the response given a language model, query token ids, and response token ids.
    """
    # Concatenate query and response into total_input
    total_input = torch.cat([query, response], dim=0)

    # Pad or truncate total_input to obs_dim
    if total_input.size(0) < obs_dim:
        print("Padding to obs_dim")
        total_input = torch.nn.functional.pad(total_input, (0, obs_dim - total_input.size(0)), value=0)  # Pad with zeros
    else:
        print("Truncating to obs_dim")
        total_input = total_input[:obs_dim]

    print(f"total_input shape: {total_input.shape}")  # Debugging output

    # Convert total_input to FloatTensor for compatibility with model
    total_input = total_input.float().unsqueeze(0)  # Add batch dimension

    # Forward pass through the model to get logits
    with torch.no_grad():
        logits, _ = model(total_input)  # Get logits from ActorCritic model

    print(f"logits shape: {logits.shape}")  # Debugging output

    # Shift logits and labels for cross-entropy loss
    shifted_logits = logits[:, :-1, :]  # Remove the last time step for logits
    shifted_labels = total_input.squeeze(0)[1:].long()  # Remove the first token for labels

    # Flatten logits and labels for cross-entropy loss
    shifted_logits = shifted_logits.reshape(-1, logits.size(-1))  # Flatten logits
    shifted_labels = shifted_labels.reshape(-1)  # Flatten labels

    # Debugging output
    print(f"shifted_logits shape: {shifted_logits.shape}")  # Expecting [sequence_length-1, vocab_size]
    print(f"shifted_labels shape: {shifted_labels.shape}")  # Expecting [sequence_length-1]
    print(f"Labels: {shifted_labels}, Max: {shifted_labels.max().item()}, Min: {shifted_labels.min().item()}")

    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shifted_logits, shifted_labels)

    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    return perplexity


"""
def calculate_perplexity(self,logits: torch.Tensor, labels: torch.Tensor) -> float:
    
    **Overview:**
        Calculate the perplexity of the prediction based on logits and labels.
    **Arguments:**
        - logits (torch.Tensor): The output logits from the model.
            - For single token prediction: [1, vocab_size].
            - For sequence prediction: [sequence_length, vocab_size].
        - labels (torch.Tensor): The ground truth labels.
            - For single token prediction: [1].
            - For sequence prediction: [sequence_length].
    **Returns:**
        - perplexity (float): The calculated perplexity value.
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"logits shape11111111111: {logits.shape}")  # [batch_size, seq_len, vocab_size]
    # Check dimensions of logits and labels
    if logits.dim() == 2:  # Single token prediction or flattened logits
        if labels.dim() == 0 or (labels.dim() == 1 and labels.size(0) == 1):
            # Single token prediction: logits shape [1, vocab_size], labels shape [1]
            labels = labels.unsqueeze(0)  # Ensure labels have a batch dimension
        elif labels.size(0) != logits.size(0):
            raise ValueError(f"Logits and labels must have matching batch size. Got logits: {logits.size()}, labels: {labels.size()}")
    elif logits.dim() == 3:  # Sequence prediction
        logits = logits.squeeze(0)  # Remove batch dimension: [sequence_length, vocab_size]
        if labels.size(0) != logits.size(0):
            raise ValueError(f"Logits and labels must have matching sequence length. Got logits: {logits.size()}, labels: {labels.size()}")

    if logits.dtype != torch.float:
        logits = logits.float()

    # Ensure labels are of type Long
    if labels.dtype != torch.long:
        labels = labels.long()
    # Calculate the cross-entropy loss
    logits = logits.unsqueeze(0)  # Add batch dimension
    print(f"logits shape: {logits.shape}")  # Expecting: [sequence_length, vocab_size]

    
    print(f"logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
    print(f"labels shape: {labels.shape}")  # [batch_size, seq_len]
    print(f"Max label value: {labels.max().item()}")  # 最大值
    print(f"Min label value: {labels.min().item()}")  # 最小值
    print(f"Labels: {labels}")  # 打印整个标签张量
    
    #检查整个标签张量是否有负值或者大于词汇表大小的值
    if (labels < 0).any() or (labels >= tokenizer.vocab_size).any():
        raise ValueError(f"Labels contain out-of-bounds values. Valid range: [0, {tokenizer.vocab_size - 1}]")
    
    
    
    loss = F.cross_entropy(logits, labels, reduction='mean')

    # Compute perplexity
    perplexity = torch.exp(loss).item()

    return perplexity
"""
class TextHistory:
    """
    **Overview:**
        The TextHistory class keeps track of the history of an interaction between the language model and the environment.
    """

    def __init__(self, text: str, tokens: Optional[torch.Tensor]):
        """
        **Overview:**
            Initialize TextHistory.
        **Arguments:**
            - text: The text of the first segment.
            - tokens: The tokens of the first segment.
        """
        
        # Record the total text generated by both user and language model.
        self.text = text
        # Record the ranges of text for each reply.
        self.text_spans = []
        # Record the tokenized total text generated by both user and language model.
        if len(text) == 0:
            self.text = ""
            self.tokens = torch.tensor([], dtype=torch.int64)
            return
        self.tokens = tokens
        # This flag shows whether this record is finished.
        self.completed = False

        self.append_segment(text, tokens)

    # delimiter
    def append_segment(self, text: str, tokens: torch.Tensor) -> None:
        """
        **Overview:**
            Append a new segment to the history.
        **Arguments:**
            - text: The text of the new segment.
            - tokens: The tokens of the new segment.
        """
        # If the text is empty, raise Error.
        if len(text) == 0 or len(tokens) == 0:
            raise ValueError("Can't append empty text or token list to history.")

        # Add the new text to ``self.text``
        original_text_length = len(self.text)
        self.text += text
        # Update the range of this new text segment.
        self.text_spans.append((original_text_length, len(self.text)))
        # Add the new tokens to ``self.tokens``.
        self.tokens = torch.cat((self.tokens, tokens))
        if len(self.tokens) > obs_dim:#会出现异常索引的问题，感觉是token太长导致的
            self.tokens = self.tokens[-obs_dim:]

    # delimiter
    @property
    def last_text_segment(self) -> str:
        """
        **Overview:**
            Get the last text segment.
        """
        start, end = self.text_spans[-1]
        return self.text[start:end]


    def to_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        **Overview:**
            Convert the history object into an observation tensor and the corresponding mask. \
            The observation tensor will be padded to a fixed length (obs_dim). \
            For ids generated by user, the mask value is 1; for ids generated by language model, the mask value is 2; for padding ids, the mask value is 0.
        """
        # Pad the observation to obs_dim.
        #obs = self.tokens
        #如果 self.tokens 是 None，初始化为长度为 0 的张量
        obs = self.tokens if self.tokens is not None else torch.tensor([], dtype=torch.int64)
        
        if len(obs) < obs_dim:
            obs = torch.nn.functional.pad(obs, (0, obs_dim-len(obs)))
        
        obs = obs.float() #其实应该是整形，但是如果转换成.long()对应用LongTensor,总是报错:
        """RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float"""
        
        # Generate corresponding mask.
        mask = torch.zeros_like(obs)
        
        if self.text_spans is None:print("self.text_spans is None")
        else:
            for i in range(len(self.text_spans)):
                sli = self.text_spans[i]
                # For ids generated for users, the mask value is 1.
                if i % 2 == 0:
                    mask[sli[0]: sli[1]] = 1
                # For ids generated by language model, the mask value is 2.
                else:
                    mask[sli[0]: sli[1]] = 2
                    
            return obs, mask


# delimiter
from PPO_text import ActorCritic
class TextEnvironment(gym.Env):
    """
    **Overview:**
        The TextEnvironment enables interaction of a LLM with an environment.
    """
    def __init__(self, model: ActorCritic, tokenizer: GPT2Tokenizer, reward_fn: Callable,
                max_turns: int = 4, generation_kwargs: Optional[Dict] = None):
        """
        Initialize the TextEnvironment with PPO-trained ActorCritic model.
        """
        self.model = model  # 替换为训练好的 PPO 策略模型
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.max_turns = max_turns
        self.generation_kwargs = generation_kwargs or dict()
        self.turn = 0
        self.history = TextHistory("", None)
        self.current_device = next(model.parameters()).device
        
        print("TextEnvironment initialized.")  # 添加调试信息
        print("tokenizer.vocab_size:", tokenizer.vocab_size)  # 添加调试信息
    """
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, reward_fn: Callable,
                 max_turns: int = 4, generation_kwargs: Optional[Dict] = None):
        """"""
        **Overview:**
            Initialize the TextEnvironment.

        **Arguments:**
            - model: The model to use for generation.
            - tokenizer: The tokenizer to use for generation.
            - reward_fn: A callable function that takes a string and returns a reward.
            - max_turns: The maximum number of turns to allow.
            - generation_kwargs: A dictionary of keyword arguments to pass to the model's generate method.
        """"""
        # Initialize the arguments.
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.max_turns = max_turns

        # Prepare the arguments for text generation.
        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs

        # Count the times of ``self.step()``
        self.turn = 0
        # Preserve the history of interactions.
        self.history = TextHistory("", None)
        # Determine the device of running the model (cpu or cuda).
        self.current_device = self.model.device

        # Define the action-space, reward-space and observation-space.
        # The action space is a sentence (string type).
        self._action_space = gym.spaces.Text(max_length=obs_dim)
        # In this demo, we use the negative perplexity as reward, whose range is (-inf, 0).
        self._reward_space = gym.spaces.Box(-float('inf'), 0)
        # The observation is the history query and response, whose shape is obs_dim.
        # If the total length of history < obs_dim, it will be padded. Detailed implementation is shown in ``TextHistory.to_obs``.
        # For each element of the observation, the value range is [0, vcab_size).
        self._observation_space = gym.spaces.Box(0, tokenizer.vocab_size, [obs_dim])
        """
    # delimiter
    def reset(self):
        """
        **Overview:**
            Reset the environment.
        """
        # Reset the history and the counter of step number.
        self.history = TextHistory("", None)
        self.turn = 0
        obs, mask = self.history.to_obs()
        return obs, mask

    # delimiter
    def generate(self) -> torch.Tensor:
        """
        **Overview:**
            Generate responses for a history.
        """
        # The input of model is all the interaction histories.
        query_tensors = self.history.tokens
        # Generate reply.
        response_tensors = self._generate(query_tensors)
        # Decode the reply into string format.
        response_texts = self.tokenizer.decode(response_tensors)
        # Add the new generated reply to ``self.history``
        self.history.append_segment(response_texts, response_tensors)

        return response_tensors
    """
    # delimiter
    def step(self, query: str) -> Tuple[torch.Tensor, float, bool, Dict]:
        """"""
        **Overview:**
            The step function of the language model environment.
        """"""
        query = str(query)
        #确保是字符串
        #print("query:",query)
        # The history is not initialized. Create a new history.
        if self.history.tokens is None:

            query_tokens = self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.current_device)
            self.history = TextHistory(query, query_tokens)
        # The history is already initialized. Append to the original history.
        else:
            query_tokens = self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.current_device)
            self.history.append_segment(query, query_tokens)
        # Generate response.
        response_tokens = self.generate()
        # Calculate the reward function.
        rew = self.reward_fn(self.model, query_tokens, response_tokens)
        # Check whether the environment is finished.
        self.turn += 1
        self.history.completed = self.turn >= self.max_turns
        obs, mask = self.history.to_obs()
        return obs, rew, self.history.completed, {"mask": mask}
      """
    def step(self, query: str) -> Tuple[str, float, bool, Dict]:
        print(f"step() called with query: {query}")
        """
        Step function that takes user input and generates a response.
        Returns the response text, reward, done flag, and additional info.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty. Please provide a valid input.")
        
        query = str(query)
        if self.history.tokens is None:
            query_tokens = self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.current_device)
            self.history = TextHistory(query, query_tokens)
        else:
            query_tokens = self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.current_device)
            self.history.append_segment(query, query_tokens)
        response_tokens = self.generate()
        reward = self.reward_fn(self.model, query_tokens, response_tokens)
        self.turn += 1
        self.history.completed = self.turn >= self.max_turns
        obs, mask = self.history.to_obs()
        response_text = self.history.last_text_segment
        return response_text, reward, self.history.completed, {"mask": mask}
        
    # delimiter
    def pad_or_crop_query(self,query_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Pad or crop the input query_tensor to match the target_length.
        Args:
            query_tensor (torch.Tensor): The input query tensor (1D).
            target_length (int): The desired length of the tensor.

        Returns:
            torch.Tensor: A tensor with the desired length.
        """
        if query_tensor.size(0) < target_length:
            # Pad with zeros
            padded_tensor = torch.zeros(target_length, dtype=query_tensor.dtype, device=query_tensor.device)
            padded_tensor[:query_tensor.size(0)] = query_tensor
        else:
            # Crop to target length
            padded_tensor = query_tensor[:target_length]
        return padded_tensor

    # delimiter
    def _generate(self, query_tensors: torch.Tensor) -> torch.Tensor:
        """
        Use PPO-trained policy to generate responses.
        """
        # 获取目标长度
        target_length = obs_dim  # obs_dim
        # 填充或裁剪 query_tensors
        query_tensors = self.pad_or_crop_query(query_tensors, target_length)
        # Add batch dimension   确保输入类型是 FloatTensor
        query_tensors = query_tensors.unsqueeze(0).float()

        

        
        # 构造 Mask
        mask = (query_tensors != 0).float()  # 生成 Mask，非 0 的地方为 1
            
        print("Query tensors after padding:", query_tensors[:20])  # 添加调试信息
        print("Query tensors shape after padding:", query_tensors.shape)  # 添加调试信息
        with torch.no_grad():
            # 使用 PPO 的 Actor 部分调整 GPT-2 的 logits
            logits, _ = self.model(query_tensors,attention_mask=mask)  # 从 PPO 策略模型获取 logits
            print("logits shape when call _generate:", logits.shape)  # 添加调试信息
            #logits = logits.squeeze(0)[-1]  # 取最后一个 token 的 logits
            logits = logits.squeeze(0)  # 移除 batch 维度，变成 (action_dim)
            
            
            #logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)##########################
            # 通过 softmax 转换为概率分布
            probs = torch.softmax(logits[-1], dim=-1)
            
            probs = torch.nan_to_num(probs, nan=1e-8, posinf=1.0, neginf=0.0)########################## 
            # 按概率分布采样生成下一个 token
            next_token = torch.multinomial(probs, num_samples=1).item()

        # 将生成的 token 追加到输入序列中
        output = torch.tensor([next_token], device=query_tensors.device)
        return output
    """
    # delimiter
    def _generate(self, query_tensors: torch.Tensor) -> torch.Tensor:
        """"""
        **Overview:**
            Generate responses for a list of query tensors.
        **Arguments:**
            - query_tensors (torch.Tensor): A list of query tensors to generate responses for.
        """"""
        # Add the batch_size dimension to the original input. Shape: [T, N] -> [1, T, N]
        query_tensors = query_tensors.unsqueeze(0)
        # Generate the corresponding mask tensor.
        batch_mask = torch.ones_like(query_tensors)
        inputs = {"input_ids": query_tensors, "attention_mask": batch_mask}

        # Call the ``generate()`` API of GPT-2.
        generation = self.model.generate(**inputs, **self.generation_kwargs,
                                         pad_token_id=self.tokenizer.eos_token_id)

        # Remove prompt from the total completed sentence.
        output = generation[0, batch_mask[0].sum():]
        return output
"""

# delimiter
def test_env():
    """
    **Overview:**
        In this function, we test the language model environment and interact with it by typing prompts in the command line.
    """
    # Load the pretrained model and tokenizer.
    # When first call this function, the pretrained files will be automatically downloaded from <link https://huggingface.co/ link>.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    #############model = GPT2LMHeadModel.from_pretrained('gpt2')  #不用GPT2LMHeadModel，而是用PPO训练的模型
    
    
    
    ppo_model = ActorCritic(obs_dim=obs_dim, action_dim=tokenizer.vocab_size)
    ppo_model.load_state_dict(torch.load('ppo_actor_critic.pth'))  # 加载训练权重
    ppo_model.eval()  # 切换到评估模式
    
    
    
    # For simplicity, we set the reward function to be the negative perplexity.
    reward_function = lambda lm, query, response: - calculate_perplexity(lm, query, response)
    # Arguments for text generation.
    generation_kwargs = {
        # The maximum number of tokens can be generated by language model is 20.
        'max_new_tokens': 20,
        # Use nondeterministic method to sample generated results each time.
        'do_sample': True,
        # The temperature of softmax function for sampling.
        'temperature': 0.7,
        # Penalize the language model to generate repeated words.
        'repetition_penalty': 2.0
    }
    # Initialize the environment.
    env = TextEnvironment(model=ppo_model, tokenizer=tokenizer, max_turns=3, reward_fn=reward_function,
                          generation_kwargs=generation_kwargs)
    env.reset()
    """
    #用自动生成的query进行交互
    for _ in range(env.max_turns):
        obs, reward, done, info = env.step("Auto-generated query here")  # 直接调用环境的 step 方法
        print("Response (Reward={:.2f}):{}".format(reward, env.history.last_text_segment))
        if done:
            break
    """      
    print("Starting interaction loop...")  # 添加调试信息
    # Interaction loop.
    while True:
        # User input the question.
        q = input("Please type in your question:")
        # The env step once.
        obs, reward, done, info = env.step(q)
        # Print the response and reward.
        print("Response (Reward={:.2f}):{}".format(reward, env.history.last_text_segment))
        # If the environment is done, break the interaction loop.
        if done:
            print("Environment is done.")
            break
    

if __name__ == '__main__':
    test_env()
