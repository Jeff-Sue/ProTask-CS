import json

data_path = "knowledge_annotation_results.json"  # 替换为你的标注结果文件路径
ori_data_path = "your_original_data_path_here"  # 替换为你的原始数据文件路径

# 方法是先记下所有knowledge_query为"是"的对话的id，然后在原始数据中找到这些id对应的对话，最后把这些对话保存到新的文件中
with open(data_path, "r", encoding="utf-8") as f:
    annotation_results = json.load(f)
    items = []
    for result in annotation_results:
        if result["parsed_response"]["knowledge_query"] == "是":
            items.append(result["id"])

save_data = {}
with open(ori_data_path, "r", encoding="utf-8") as f:    
    original_data = json.load(f)
    for dialog_id, dialog in original_data.items():
        if dialog_id in items:
            save_data[dialog_id] = dialog

save_path = "your_save_path_here"  # 替换为你要保存的文件路径
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)