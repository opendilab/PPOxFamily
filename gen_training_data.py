import openai
import json
import os

# 从环境变量中读取 API 密钥
openai.api_key = os.getenv("OPENAI_KEY")
print("api_key:", openai.api_key)
if not openai.api_key:
    raise ValueError("API 密钥未设置，请在环境变量中定义 OPENAI_KEY")

def generate_question():
    """
    使用 OpenAI API 动态生成幽默问题。
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative assistant that generates humorous questions."},
                {"role": "user", "content": "Generate a single humorous question suitable for a Q&A pair. The question should be concise, creative, and amusing."}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        question = response['choices'][0]['message']['content'].strip()
        return question
    except Exception as e:
        print(f"调用 OpenAI API 生成问题失败: {e}")
        return "Fallback question: Why did the chicken cross the road?"

def generate_answer(question, humor=True):
    """
    使用 OpenAI API 动态生成回答。
    """
    try:
        style = "humorous" if humor else "non-humorous"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that provides {style} answers."},
                {"role": "user", "content": f"Provide a {style} answer to the following question: {question}"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        print(f"调用 OpenAI API 生成回答失败: {e}")
        return f"Fallback {style} answer to the question: {question}"

def create_training_data(num_pairs):
    """
    使用 OpenAI API 动态生成训练数据。
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
            "reward_humorous": 1.0,
            "reward_non_humorous": 0.2
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
    NUM_PAIRS = 30
    OUTPUT_FILE = "training_data_openai.json"

    training_data = create_training_data(NUM_PAIRS)
    save_training_data_to_json(training_data, OUTPUT_FILE)
