import json
import os
from utils.multi_api import parallel_inference_dual_servers


def summarize_query(dialogue):
    """
    使用LLM总结对话中的用户查询。
    """
    # 提取所有用户消息
    user_messages = [msg.replace("用户:", "").strip() for msg in dialogue if msg.startswith("用户:")]
    query_text = " ".join(user_messages)

    if not query_text:
        return "无用户查询"

    # 构建prompt，用query.prompt的方式引导LLM进行总结
    with open("query.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{p_dialogue}", query_text)
    messages = [{"role": "user", "content": prompt}]

    return messages


def process_dataset(input_path, output_dir):
    """
    处理数据集：加载数据，使用LLM总结query，添加summary字段。
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 准备样本列表
    samples = []
    key_to_idx = {}
    processed_data = {}
    for idx, (key, item) in enumerate(data.items()):
        dialogue = item.get("text", [])
        messages = summarize_query(dialogue)
        if isinstance(messages, str):  # 无用户查询
            processed_data[key] = {
                "original": item,
                "summary_query": messages
            }
        else:  # 有消息，加入批量处理
            samples.append({
                "id": key,
                "messages": messages
            })
            key_to_idx[key] = idx

    # 批量并发推理
    if samples:
        results = parallel_inference_dual_servers(samples)

        # 处理结果
        for result in results:
            key = result["id"]
            summary = result["response"].strip() if result["success"] else "总结失败"
            processed_data[key] = {
                "original": data[key],
                "summary_query": summary
            }

    return processed_data


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将数据集分割成训练、验证和测试集。
    """
    items = list(data.items())
    total = len(items)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = dict(items[:train_end])
    val = dict(items[train_end:val_end])
    test = dict(items[val_end:])

    return train, val, test


def save_dataset(data, file_path):
    """
    保存数据集到JSON文件。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_path = "Huawei_data/data.json"
    output_dir = "ProTask-CS/静态单轮/QueryModel/"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理数据集
    processed_data = process_dataset(input_path, output_dir)

    # 分割数据集
    train, val, test = split_dataset(processed_data)

    # 保存分割后的数据集
    save_dataset(train, os.path.join(output_dir, "train.json"))
    save_dataset(val, os.path.join(output_dir, "val.json"))
    save_dataset(test, os.path.join(output_dir, "test.json"))

    print("数据集构建完成！")
    print(f"训练集: {len(train)} 条")
    print(f"验证集: {len(val)} 条")
    print(f"测试集: {len(test)} 条")