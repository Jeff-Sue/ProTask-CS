import requests

def rag(query, chat_content, use_chat):
    payload = {
        "query": query,
        "max_doc_num": 5,
        "index": "full_document",
        "use_chat": use_chat,
        "chat_content": chat_content
    }

    url = "your_api_url"
    response = response.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    cases = []
    occur_id = []

    for item in result["es_response"]["hits"]["hits"]:
        case = {}
        if item["_source"]["caseId"] in result["case_ids"]:
            if item["_source"]["caseId"] not in occur_id:
                case["case_id"] = item["_source"]["caseId"]
                case["title"] = item["_source"]["title"]
                case["content"] = item["_source"]["content"]
                cases.append(case)
    
    return cases

    
