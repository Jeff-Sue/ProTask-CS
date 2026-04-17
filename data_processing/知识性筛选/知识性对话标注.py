from email.mime import text
import json
from tqdm import tqdm

knowledge_prompts = """你是一个IT客服对话标注助手。你的任务是：只根据对话中所有“用户”发言，判断用户整体意图是否属于“知识性提问”。

【核心原则】
- 只看用户发言，不看客服发言
- 综合所有用户轮次，归纳用户整体意图
- 不要被 IT 专有名词误导，出现平台名/系统名/产品名不等于知识性提问
- 判断关键在于：用户是否在寻求知识、解释、判断、排查或方案

【知识性提问的定义】
若用户的诉求需要客服提供以下任一种信息，判为“知识性提问=是”：
- 原因解释
- 故障分析
- 排查方法
- 操作指导
- 规则说明
- 解决方案
- 产品/功能说明
- 条件判断或建议

【非知识性提问的定义】
若用户的诉求仅属于以下类型，判为“知识性提问=否”：
- 纯闲聊/寒暄
- 纯执行要求（让客服帮忙做某事、联系某人、催办某事）
- 纯进度询问/状态确认
- 纯信息补充，不构成知识诉求

【判定细则】
1. “你好”“在吗”“谢谢”等寒暄，判为否
2. “帮我联系一下”“给我打WeLink”“帮我上线”这类执行请求，判为否
3. “处理了吗”“什么时候恢复”这类进度跟进，通常判为否
4. “为什么会这样”“怎么处理”“怎么解决”“是什么原因”“这个功能怎么用”“为什么登不上”这类，判为是
5. 用户即使没有显式问号，但如果描述故障现象的目的明显是寻求判断、排查或解决，也判为是
6. 多轮对话中，只要用户整体意图包含明确的知识诉求，就判为是；否则判为否

【输出要求】
请只输出以下JSON，不要输出任何其他内容：

{
  "intent": "一句话概括用户整体意图",
  "category": "纯闲聊/纯要求/事务跟进/知识咨询/故障排查/规则询问/操作指导",
  "knowledge_query": "是/否",
  "reason": "简要说明判断依据，突出用户诉求中的知识性成分或缺乏知识性成分的原因"
}
"""

def validate_knowledge_output(text):
    try:
        data = json.loads(text)
    except Exception as e:
        return False, f"JSON解析失败: {e}"

    required_keys = ["intent", "category", "knowledge_query", "reason"]
    for key in required_keys:
        if key not in data:
            return False, f"缺少字段: {key}"

    if data["knowledge_query"] not in ["是", "否"]:
        return False, f'knowledge_query取值非法: {data["knowledge_query"]}'

    return True, data

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.multi_api import parallel_inference


data_path = "your_data_path_here"  # 替换为你的数据路径
items = []
with open(data_path, "r", encoding="utf-8") as f:
    samples = json.load(f)
    for dialog_id, dialog in samples.items():
        dialog_content = ""
        item = {}
        item['id'] = dialog_id
        item["messages"] = [{"role": "system", "content": knowledge_prompts}]
        for dict in dialog["text"]:
            dialog_content += f"{list(dict.keys())[0]}:{dict[list(dict.keys())[0]]}\n"
        item["messages"].append({"role": "user", "content": dialog_content})
        items.append(item)


results = parallel_inference(items, max_workers=20, max_retry=3, validator=validate_knowledge_output)
save_data = "your_save_path_here"  # 替换为你的保存路径
with open(save_data, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)