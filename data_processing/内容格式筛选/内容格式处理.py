import os
import json
import re
import copy
import html
import random
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher


# ========== 通用工具 ==========
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_level_result(output_root, level_id, level_name, keep, drop):
    level_dir = Path(output_root) / f"Level{level_id}_{level_name}"
    level_dir.mkdir(parents=True, exist_ok=True)

    save_json(keep, level_dir / "keep.json")
    save_json(drop, level_dir / "drop.json")
    save_json(
        {
            "level_id": level_id,
            "level_name": level_name,
            "keep_num": len(keep),
            "drop_num": len(drop),
            "total_num": len(keep) + len(drop),
        },
        level_dir / "stats.json"
    )

def make_drop_record(dialog_content, level_name, reason, original_text=None, processed_text=None, extra=None):
    rec = copy.deepcopy(dialog_content)
    rec["_drop_meta"] = {
        "level": level_name,
        "reason": reason,
    }
    if original_text is not None:
        rec["_drop_meta"]["original_text"] = original_text
    if processed_text is not None:
        rec["_drop_meta"]["processed_text"] = processed_text
    if extra is not None:
        rec["_drop_meta"]["extra"] = extra
    return rec

def run_level(dialogs, level_id, level_name, processor, output_root):
    keep, drop = {}, {}

    for dialog_id, dialog_content in tqdm(dialogs.items(), desc=f"Level {level_id} - {level_name}"):
        ok, result = processor(dialog_id, copy.deepcopy(dialog_content))
        if ok:
            keep[dialog_id] = result
        else:
            drop[dialog_id] = result

    save_level_result(output_root, level_id, level_name, keep, drop)
    return keep


# ========== 你原有的一些函数，修正版 ==========

# Level 1: 案例链接替换（这是“改写”，不是“筛掉”）
def 案例库():
    anli_path = "path_to_your_case_library.json"
    with open(anli_path, "r", encoding="utf-8") as f:
        case_library = json.load(f)
    case_ids = list(case_library.keys())

    return case_ids

def anli(text, case_id):
    pattern = rf'https?://[^\s"]*caseId={re.escape(case_id)}[^\s"]*'
    return re.sub(pattern, "[案例链接]", text)

# 思路：这个caseID必须在我们的案例库里才替换成[案例链接]，如果文本里存在caseId=xxx的链接，但是这个xxx不在我们的案例库里，删除整个对话
def case_selection_single(dialog_content):
    dialog_content = copy.deepcopy(dialog_content)
    if dialog_content.get("caseId"):
        for i in range(len(dialog_content["text"])):
            if "caseId=" in dialog_content["text"][i]:
                case_id_match = re.search(r'caseId=([A-Za-z0-9]+)', dialog_content["text"][i])
                if case_id_match:
                    case_id = case_id_match.group(1)
                    if case_id in 案例库():
                        dialog_content["text"][i] = anli(dialog_content["text"][i], case_id)
                    else:
                        return None
    return dialog_content

# Level 3: 无效话术删除
def useless_utt_delete(dialog):
    user_patterns = [
        "What can I do for you?",
        "I'm glad to be of service to you",
        "很高兴为您服务",
        "我是本次为您服务的",
        "欢迎使用IT人工支持",
        "请问有什么可以帮到您",
        "当前会话窗口即将关闭",
        "长时间没有收到您的回复",
        "会话将在1分钟后关闭",
        "会话将于1分钟后关闭",
        "窗口将在一分钟后关闭",
        "窗口将会在1分钟后关闭",
        "本次会话已经超时结束",
        "超时关闭",
        "长时间未响应",
        "避免其他用户等待",
    ]

    assistant_patterns = {
        "开头系统自动greeting": {
            "What can I do for you?",
            "I'm glad to be of service to you",
            "很高兴为您服务",
            "我是本次为您服务的",
        },
        "共性故障排队": {
            "当前求助用户较多",
        },
        "需要用户等待话术": {
            "您好，坐席正在快马加鞭为您查询中，请您耐心等候，谢谢",
            "欢迎回来",
            "正在帮您查询相关解决方案",
        },
        "查询解决方案后回复话术": {
            "提供给您的案例是否解决了您的问题",
            "请问还有其他可以帮到您的吗",
            "请问您这边还有其它的疑问吗",
            "请问您还有其它问题需要咨询吗",
            "当前求助人数较多",
            "请问还在线吗？",
        },
        "挂断会话": {
            "如没有其他问题",
            "没有其他疑问",
            "welink沟通群已经给您拉好了",
            "会话就先关闭了",
            "很荣幸为您服务",
            "感谢你的理解与支持！",
            "后续您有需要",
            "进行评价",
            "祝您工作愉快",
            "祝您生活愉快",
            "五星好评",
            "仓库小棉袄",
            "继续在welink沟通",
            "感谢您的咨询",
        },
        "系统超时自动关闭窗口": {
            "当前会话窗口即将关闭",
            "长时间没有收到您的回复",
            "会话将在1分钟后关闭",
            "会话将于1分钟后关闭",
            "窗口将在一分钟后关闭",
            "窗口将会在1分钟后关闭",
            "本次会话已经超时结束",
            "超时关闭",
            "长时间未响应",
            "避免其他用户等待",
        },
    }

    new_dialog = []
    for i in range(len(dialog)):
        flag = True

        if dialog[i].startswith("用户"):
            for pattern in user_patterns:
                if pattern in dialog[i]:
                    flag = False
                    break

        elif dialog[i].startswith("客服"):
            for pattern_type, patterns in assistant_patterns.items():
                if pattern_type == "查询解决方案后回复话术":
                    for pattern in patterns:
                        if pattern in dialog[i]:
                            flag = False
                            break
                else:
                    for pattern in patterns:
                        if pattern in dialog[i]:
                            flag = False
                            break
                if not flag:
                    break

        if flag:
            new_dialog.append(dialog[i])

    return new_dialog


# Level 4: 重复信息
def is_rewrite(a, b, sim_threshold=0.9, max_len_diff=3):
    if abs(len(a) - len(b)) > max_len_diff:
        return False

    sim = SequenceMatcher(None, a, b).ratio()
    if sim < sim_threshold:
        return False

    matcher = SequenceMatcher(None, a, b)
    diffs = [op for op in matcher.get_opcodes() if op[0] != "equal"]

    if len(diffs) <= 2:
        changed_chars = 0
        for tag, i1, i2, j1, j2 in diffs:
            changed_chars += max(i2 - i1, j2 - j1)
        if changed_chars <= 4:
            return True
    return False

def is_bilingual_text(text, min_cn_chars=10, min_en_words=5):
    cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
    en_words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(cn_chars) >= min_cn_chars and len(en_words) >= min_en_words

def repeat_utt_selection(dialog):
    if len(dialog) < 2:
        return dialog

    delete_ids = set()

    for utt_id in range(len(dialog) - 1):
        if dialog[utt_id + 1].startswith(dialog[utt_id]):
            delete_ids.add(utt_id)
        if dialog[utt_id].startswith(dialog[utt_id + 1]):
            delete_ids.add(utt_id + 1)

        if is_rewrite(dialog[utt_id], dialog[utt_id + 1]):
            delete_ids.add(utt_id)

    for utt_id in range(len(dialog)):
        if is_bilingual_text(dialog[utt_id]):
            delete_ids.add(utt_id)

    new_dialog = [dialog[i] for i in range(len(dialog)) if i not in delete_ids]
    return new_dialog


# Level 5: 敏感信息脱敏
def data_masking(dialog):
    dialog = copy.deepcopy(dialog)

    for idx in range(len(dialog)):
        if "工号" in dialog[idx]:
            dialog[idx] = re.sub(r"[A-Za-z]{2,}\d{3,}", "[工号]", dialog[idx])
            dialog[idx] = re.sub(r"\b\d{6,}\b", "[工号]", dialog[idx])
            dialog[idx] = re.sub(r"[A-Za-z]?[Ww][Xx]\d{6,7}(?!\d)", "[工号]", dialog[idx])

    if any("电话" in turn for turn in dialog):
        for idx in range(len(dialog)):
            dialog[idx] = re.sub(r"\d{11}", "[电话]", dialog[idx])

    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    if any("邮箱" in turn for turn in dialog):
        for idx in range(len(dialog)):
            dialog[idx] = re.sub(email_pattern, "[邮箱]", dialog[idx])

    return dialog


# Level 6: 标点修正
def typing_correction(dialog):
    dialog = copy.deepcopy(dialog)
    for idx in range(len(dialog)):
        if re.search(r"[。！？!?…]$", dialog[idx]):
            continue

        dialog[idx] = re.sub(r'[，、；：;:]+$', '', dialog[idx])

        question_patterns = [
            r'吗$', r'呢$', r'么$', r'嘛$', r'是否$', r'怎么$',
            r'为什么$', r'如何$', r'哪个$', r'几点$', r'多少$',
            r'能不能$', r'可不可以$', r'有没有$', r'是不是$'
        ]

        exclaim_patterns = [
            r'太好了$', r'真棒$', r'谢谢$', r'太感谢了$', r'辛苦了$', r'麻烦了$'
        ]

        if any(re.search(p, dialog[idx]) for p in question_patterns):
            dialog[idx] += '？'
        elif any(re.search(p, dialog[idx]) for p in exclaim_patterns):
            dialog[idx] += '！'
        else:
            dialog[idx] += '。'

    return dialog


# Level 7: 用户开头
def user_start(dialog):
    delete_ids = []
    for turn_id in range(len(dialog)):
        if dialog[turn_id].startswith("客服"):
            delete_ids.append(turn_id)
        else:
            break
    return [dialog[i] for i in range(len(dialog)) if i not in delete_ids]


# Level 8: 客服结尾
def assistant_end(dialog):
    delete_ids = []
    for turn_id in range(len(dialog) - 1, -1, -1):
        if dialog[turn_id].startswith("用户"):
            delete_ids.append(turn_id)
        else:
            break
    return [dialog[i] for i in range(len(dialog)) if i not in delete_ids]


# Level 9: 交替重组（修正版）
def interact_reformat(dialog):
    if not dialog:
        return []

    def get_role(utt):
        if utt.startswith("用户:"):
            return "用户"
        if utt.startswith("客服:"):
            return "客服"
        return None

    first_role = get_role(dialog[0])
    if first_role is None:
        return []

    merge_list = []
    current_role = first_role
    current_str = dialog[0].split(":", 1)[-1].strip()

    for i in range(1, len(dialog)):
        role = get_role(dialog[i])
        if role is None:
            continue

        content = dialog[i].split(":", 1)[-1].strip()

        if role == current_role:
            current_str += "\n" + content
        else:
            merge_list.append({current_role: current_str})
            current_role = role
            current_str = content

    merge_list.append({current_role: current_str})
    return merge_list


# Level 10: 中文用户
def is_english_sentence(text):
    text = text.replace("&nbsp", " ").replace("&nbsp;", "")
    text = html.unescape(text)
    return bool(re.fullmatch(r"[A-Za-z0-9\s,.!?;:'\"()\-&$]+", text))

def has_english_user(dialog):
    for turn in dialog:
        for role, content in turn.items():
            if role == "用户":
                first_line = re.sub(r'[。！？!?…]$', '', content.split("\n")[0])
                if is_english_sentence(first_line):
                    return True
    return False


# ========== 每个 level 的 processor ==========

def processor_level1(dialog_id, dialog_content):
    original_text = copy.deepcopy(dialog_content)
    new_content = case_selection_single(dialog_content)
    if new_content is None:
        drop_rec = make_drop_record(
            dialog_content,
            level_name="case_replace",
            reason="contains_unqualified_case_link",
            original_text=original_text
        )
        return False, drop_rec
    return True, new_content

def processor_level2(dialog_id, dialog_content):
    dialog = dialog_content["text"]
    hit_keys = []

    for key in ["http://", "https://", "edoc", "errorMsg", "style="]:
        if any(key in x for x in dialog):
            hit_keys.append(key)

    if hit_keys:
        drop_rec = make_drop_record(
            dialog_content,
            level_name="non_text",
            reason="contains_non_text_signal",
            original_text=dialog_content["text"],
            extra={"hit_keys": hit_keys}
        )
        return False, drop_rec

    return True, dialog_content

def make_transform_processor(fn, level_name):
    def processor(dialog_id, dialog_content):
        original_text = copy.deepcopy(dialog_content["text"])
        new_text = fn(copy.deepcopy(dialog_content["text"]))

        if not new_text:
            drop_rec = make_drop_record(
                dialog_content,
                level_name=level_name,
                reason="empty_after_transform",
                original_text=original_text,
                processed_text=new_text
            )
            return False, drop_rec

        dialog_content["text"] = new_text
        return True, dialog_content

    return processor

def processor_level10(dialog_id, dialog_content):
    dialog = dialog_content["text"]
    if has_english_user(dialog):
        drop_rec = make_drop_record(
            dialog_content,
            level_name="chinese_user",
            reason="contains_english_user_turn",
            original_text=dialog
        )
        return False, drop_rec
    return True, dialog_content

def processor_level11(dialog_id, dialog_content):
    dialog = dialog_content["text"]
    if len(dialog) < 2 and list(dialog[-1].values())[0].endswith("？"):
        drop_rec = make_drop_record(
            dialog_content,
            level_name="meaningful_end",
            reason="too_short_and_ends_with_question",
            original_text=dialog
        )
        return False, drop_rec
    return True, dialog_content


# ========== List convert to messages ==========
def list_to_messages(dialogue):
    converted = []
    for dialog_id, dialog_content in dialogue.items():
        dialog = dialog_content["text"]
        if not isinstance(dialog, list):
            continue

        messages = []
        for turn in dialog:
            for role, content in turn.items():
                if role == "用户":
                    messages.append({"role": "user", "content": content.strip().replace("\n", " ")})
                elif role == "客服":
                    messages.append({"role": "assistant", "content": content.strip().replace("\n", " ")})
        if len(messages) >= 2:
            converted.append({"dialog_id": dialog_id, "messages": messages})

    return converted


# ========== 主流程 ==========
if __name__ == "__main__":
    source_data_file = "Your_Source_Data_File"
    output_root = "pipeline_outputs"

    with open(source_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("### 全部样本数量 ###")
    print(len(data))

    dialogs = copy.deepcopy(data)

    dialogs = run_level(dialogs, 1, "case_replace", processor_level1, output_root)
    dialogs = run_level(dialogs, 2, "non_text", processor_level2, output_root)
    dialogs = run_level(dialogs, 3, "useless_utt_delete", make_transform_processor(useless_utt_delete, "useless_utt_delete"), output_root)
    dialogs = run_level(dialogs, 4, "repeat_utt_selection", make_transform_processor(repeat_utt_selection, "repeat_utt_selection"), output_root)
    dialogs = run_level(dialogs, 5, "data_masking", make_transform_processor(data_masking, "data_masking"), output_root)
    dialogs = run_level(dialogs, 6, "typing_correction", make_transform_processor(typing_correction, "typing_correction"), output_root)
    dialogs = run_level(dialogs, 7, "user_start", make_transform_processor(user_start, "user_start"), output_root)
    dialogs = run_level(dialogs, 8, "assistant_end", make_transform_processor(assistant_end, "assistant_end"), output_root)
    dialogs = run_level(dialogs, 9, "interact_reformat", make_transform_processor(interact_reformat, "interact_reformat"), output_root)
    dialogs = run_level(dialogs, 10, "chinese_user", processor_level10, output_root)
    dialogs = run_level(dialogs, 11, "meaningful_end", processor_level11, output_root)

    print("### 最终保留样本数量 ###")
    print(len(dialogs))

    save_json(dialogs, Path(output_root) / "final_keep.json")

    # 下面这个地方也建议防一下样本数不足
    random.seed(42)
    sample_num = min(10000, len(dialogs))
    raw_dialogs_idxs = random.sample(list(dialogs.keys()), sample_num)

    train_cut = int(sample_num * 0.8)
    val_cut = int(sample_num * 0.9)

    train_dialogs = {idx: dialogs[idx] for idx in raw_dialogs_idxs[:train_cut]}
    val_dialogs = {idx: dialogs[idx] for idx in raw_dialogs_idxs[train_cut:val_cut]}
    test_dialogs = {idx: dialogs[idx] for idx in raw_dialogs_idxs[val_cut:]}

    save_json(train_dialogs, Path(output_root) / "data" / "train.json")
    save_json(val_dialogs, Path(output_root) / "data" / "val.json")
    save_json(test_dialogs, Path(output_root) / "data" / "test.json")

    save_json(list_to_messages(train_dialogs), Path(output_root) / "data" / "train_messages.json")
    save_json(list_to_messages(val_dialogs), Path(output_root) / "data" / "val_messages.json")
    save_json(list_to_messages(test_dialogs), Path(output_root) / "data" / "test_messages.json")