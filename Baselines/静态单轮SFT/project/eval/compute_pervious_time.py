import json

def calculate_avg_latency(data_file):
    """
    从JSON文件中读取所有样本的latency数据，计算每个key的平均值
    
    Args:
        data_file: JSON文件路径，可以是每行一个JSON的格式，也可以是JSON数组格式
    
    Returns:
        dict: 包含每个latency key的平均值
    """
    latencies = []
    
    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 尝试解析为JSON数组
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if 'latency' in item:
                    latencies.append(item['latency'])
    except json.JSONDecodeError:
        # 如果是每行一个JSON的格式
        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if 'latency' in item:
                        latencies.append(item['latency'])
                except json.JSONDecodeError:
                    continue
    
    if not latencies:
        print("未找到任何latency数据")
        return {}
    
    # 计算平均值
    avg_latency = {}
    keys = latencies[0].keys()
    
    for key in keys:
        values = [l[key] for l in latencies if key in l]
        if values:
            avg_latency[key] = sum(values) / len(values)
    
    return avg_latency

# 使用示例
if __name__ == "__main__":
    # data_file = "/home/aarc/CuhkszTeam/nas1/zefeng/RAG_agent/后处理_selected_compress_keywords/results/jieli_20to5_select_keywords.json" # 替换为您的文件路径
    data_file = "result.json"
    avg_latency = calculate_avg_latency(data_file)
    
    print("各latency指标的平均值（毫秒）：")
    for key, value in avg_latency.items():
        print(f"{key}: {value:.2f} ms")
    
    # 可选：输出为更友好的格式
    print("\n详细统计：")
    print(f"总耗时平均: {avg_latency.get('total_time', 0):.2f} ms ({avg_latency.get('total_time', 0)/1000:.2f} 秒)")
