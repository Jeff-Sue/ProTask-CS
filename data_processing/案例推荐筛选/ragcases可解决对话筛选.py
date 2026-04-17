import json
from openai import OpenAI

query_prompt = """your_query_prompt"""

# 第一步：从utils.scu_rag 导入 rag函数，检索回答相关的案例，并将案例添加到数据中
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.scu_rag import rag


## 用一个Qwen2.5-0.5B模型根据query_prompt和对话内容，总结输入给rag的用户query

query_model = "your_query_model"  # 替换为你的模型名称
def generate_query(query_prompt, dialogue, query_model):
    # 这里你可以使用任何你喜欢的模型来生成query，这里只是一个示例
    # 加入openai的模型调用代码，生成query
    query = query_prompt + " " + dialogue[-1]['content']  # 简单地将prompt和最后一句对话内容拼接起来作为query
    # 这里可以调用你的模型来生成query，例如：
    client = OpenAI()
    response = client.completions.create(
        model=query_model,
        prompt=query,
        max_tokens=50
    )
    query = response.choices[0].text.strip()
    return query



data_path = 'caseId_filtered_data.json'  # 替换为你的数据文件路径
output_path = 'ragcases_filtered_data.json'  # 输出文件路径
save_data = {}
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for key, value in data.items():
        dialogue = value['dialogue']  # 获取对话内容
        cases = rag(query_prompt, dialogue, use_chat=True)  # 调用rag函数获取相关案例
        value['cases'] = cases  # 将获取的案例添加到数据中
        save_data[key] = value  # 保存到新的数据字典中

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=4)


# 第二步：对添加了案例的数据进行筛选，加入prompt，筛选出能够解决对话问题的案例，并将这些案例添加到数据中
