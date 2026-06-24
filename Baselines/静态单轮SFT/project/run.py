from pipeline import Pipeline
from json_repair import repair_json
import json
import re
import time
import sys
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm


def save_json(new_data: list, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f2:
        json.dump(new_data, f2, ensure_ascii=False, indent=2)

    return None

def extract_case_id(turn):
    pattern2 = r'\[案例链接:\s*(KT\d+)\]'
    match = re.search(pattern2, turn)

    if not match:
        print("current wrong turn:", turn)
        return None

    return match.group(1)

def call_api(dialogue):
    url = "http://localhost:8338/chat"
    payload = {
        "dialogue": dialogue
    }

    response = requests.post(url, json=payload)

    return response.json()

if __name__ == "__main__":
    INPUT_PATH = "带caseID对话.json"
    MAX_DIALOGUES = 300
    # pipeline = Pipeline()
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        items = list(data.items())
        if MAX_DIALOGUES is not None:
            items = items[:MAX_DIALOGUES]

    dialogue_history_str_list = []
    golden_case_ids = []
    for dialog_id, dialogue_content in items:
        dialogue_history: List[str] = []
        for turn in dialogue_content["text"]:
            if "用户" in turn:
                dialogue_history.append(f"用户: {turn['用户']}")
            else:
                current_turn = f"客服: {turn['客服']}"
                if "案例链接:" in current_turn:
                    if extract_case_id(current_turn):
                        dialogue_history_str = "".join(dialogue_history)
                        dialogue_history_str_list.append(dialogue_history_str)
                        golden_case_ids.append(extract_case_id(current_turn))
                dialogue_history.append(current_turn)
    
    save_results = []
    for idx, dialogue_history_str in tqdm(enumerate(dialogue_history_str_list)):
        # result = pipeline.run(dialogue_history_str)
        result = call_api(dialogue_history_str)
        print("\n=====Result=====")
        print(result)
        save_results.append(result)
        print("\n=====Golden=====")
        print(golden_case_ids[idx])

    save_path = "result.json"
    save_json(save_results, save_path)