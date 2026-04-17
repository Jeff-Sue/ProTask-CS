## 从知识性筛选后的数据集中，选取caseId标签不为空的对话

import json

data_path = 'knowledge_filtered_data.json'  # 替换为你的数据文件路径
output_path = 'caseId_filtered_data.json'  # 输出文件路径

save_data = {}
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for key, value in data.items():
        if value['caseId']:  # 检查caseId标签是否存在且不为空
            save_data[key] = value

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=4)