import json
import time
import os
import requests

# 从环境变量中读取 API 密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("API 密钥未设置，请在环境变量中定义 DEEPSEEK_API_KEY。")

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com"  

def generate_question():
    """
    使用 DeepSeek API 动态生成幽默问题。
    """
    try:
        payload = {
            "prompt": "Generate a single humorous question suitable for a Q&A pair. The question should be concise, creative, and amusing.",
            "max_tokens": 50,
            "temperature": 0.7
        }
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # 如果响应状态码非 200，将抛出异常
        question = response.json().get("text", "").strip()
        return question
    except Exception as e:
        print(f"调用 DeepSeek API 生成问题失败: {e}")
        return "Fallback question: Why did the chicken cross the road?"

def generate_answer(question, humor=True):
    """
    使用 DeepSeek API 动态生成回答。
    """
    try:
        style = "humorous" if humor else "non-humorous"
        prompt = f"Provide a {style} answer to the following question: {question}"
        payload = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        }
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        answer = response.json().get("text", "").strip()
        return answer
    except Exception as e:
        print(f"调用 DeepSeek API 生成回答失败: {e}")
        return f"Fallback {style} answer to the question: {question}"

def create_training_data(num_pairs):
    """
    使用 DeepSeek API 动态生成训练数据。
    """
    data = []
    for i in range(num_pairs):
        print(f"生成第 {i + 1} 个 QA 对...")
        question = generate_question()
        humorous_answer = generate_answer(question, humor=True)
        non_humorous_answer = generate_answer(question, humor=False)
        data.append({
            "question": question,
            "answer_humorous": humorous_answer,
            "answer_non_humorous": non_humorous_answer,
            "reward_humorous": 1.0,  # 高奖励
            "reward_non_humorous": 0.2  # 低奖励
        })
    return data

def save_training_data_to_json(data, output_file):
    """
    保存生成的训练数据到 JSON 文件。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, indent=4, ensure_ascii=False)
    print(f"训练数据已保存到 {output_file}")

if __name__ == "__main__":
    # 生成训练数据的数量
    NUM_PAIRS = 30
    OUTPUT_FILE = "training_data_deepseek.json"

    # 创建训练数据
    training_data = create_training_data(NUM_PAIRS)

    # 保存训练数据到 JSON 文件
    save_training_data_to_json(training_data, OUTPUT_FILE)
