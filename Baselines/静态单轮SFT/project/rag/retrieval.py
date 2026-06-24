import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def zefeng_rag(dialog_history, query):
    payload = {
        "dialogue": dialog_history,
        "query": query,
    }
    cases = []
    url = "http://localhost:8088/process"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    # print(result)
    occur_id = []


    # for item in result["result"]:
    #     case = {}
    #     if item["case_id"] in result["retrieved_case_ids"]:
    #         if item["case_id"] not in occur_id:
    #             occur_id.append(item["case_id"])
    #             cases = cases + str(item["case_id"]) + "\n"
    #             cases = cases + str(item["pruned_context"]) + "\n"
    for item in result["cases"]:
        case = {}
        case['case_id'] = item["case_id"]
        case['title'] = item["title"]
        case['content'] = item["processed_content"]
        cases.append(case)
    return cases[:5]