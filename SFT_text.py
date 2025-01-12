import torch
import gym
from typing import Callable, Optional, Dict, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lm_env2 import calculate_perplexity, TextEnvironment
# Your existing environment code here...

def test_env():
    """
    Test the language model environment by interacting with it.
    """
    # Load the pretrained model and tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Define the reward function as the negative perplexity.
    def reward_function(lm, query, response):
        return -calculate_perplexity(lm, query, response)

    # Arguments for text generation.
    generation_kwargs = {
        'max_new_tokens': 20,
        'do_sample': True,
        'temperature': 0.7,
        'repetition_penalty': 2.0
    }

    # Initialize the environment.
    env = TextEnvironment(
        model=model,
        tokenizer=tokenizer,
        max_turns=3,
        reward_fn=reward_function,
        generation_kwargs=generation_kwargs
    )

    # Reset the environment.
    obs, mask = env.reset()
    print("Environment reset complete.")

    # Automatic interaction loop.
    print("Starting automatic interaction loop...")
    for turn in range(env.max_turns):
        obs, reward, done, info = env.step("Auto-generated query here")
        print(f"Turn {turn + 1}:")
        print(f"Response (Reward={reward:.2f}): {env.history.last_text_segment}")
        if done:
            print("Conversation ended.")
            break

    print("\nEnvironment interaction complete.")

if __name__ == '__main__':
    test_env()


#网上直接去找数据集，不要动作池，语言的公开数据集
#demo
#较小数据规模
#自由输出