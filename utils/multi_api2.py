import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = "你是一个有帮助的助手。"

OPENAI_API_KEY = "EMPTY"

# 两个已经部署好的服务
API_CONFIGS = [
    {
        "name": "qwen3-32b",
        "base_url": "http://localhost:8851/v1",
        "model": "qwen3-32b"
    },
    {
        "name": "qwen3-32b",
        "base_url": "http://localhost:8850/v1",
        "model": "qwen3-32b"
    },
    {
        "name": "qwen3-32b",
        "base_url": "http://localhost:8849/v1",
        "model": "qwen3-32b"
    }
]

MAX_WORKERS_PER_SERVER = 16
MAX_RETRY = 3

# 每个线程缓存自己的 client，避免重复创建
_thread_local = threading.local()


def get_client(base_url):
    if not hasattr(_thread_local, "clients"):
        _thread_local.clients = {}

    if base_url not in _thread_local.clients:
        _thread_local.clients[base_url] = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=base_url
        )

    return _thread_local.clients[base_url]


def call_qwen_api(messages, base_url, model_name):
    """
    单次 API 调用
    """
    client = get_client(base_url)
    full_messages = messages
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=full_messages,
        temperature=0.0,
        max_tokens=200,
        top_p=0.5,
        presence_penalty=1.5,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    return chat_response.choices[0].message.content

def build_retry_messages(base_messages, bad_response, format_error):
    retry_messages = list(base_messages)
    retry_messages.append({"role": "assistant", "content": bad_response})
    retry_messages.append({
        "role": "user",
        "content": (
            "你刚才的输出不符合要求。"
            f"错误原因：{format_error}。"
            "请严格按原要求重新输出，且只输出合法结果，不要输出任何解释、前缀、后缀或 markdown 代码块。"
        )
    })
    return retry_messages
    

def process_one_sample(sample, api_config, validator=None, max_retry=MAX_RETRY):
    sample_id = sample.get("id")
    base_messages = sample.get("messages", [])
    current_messages = list(base_messages)
    last_error = None
    last_response = None
    parsed_response = None

    for attempt in range(1, max_retry + 1):
        try:
            response = call_qwen_api(
            current_messages,
            base_url=api_config["base_url"],
            model_name=api_config["model"]
        )
            last_response = response

            if validator is None:
                return {
                    "id": sample_id,
                    "messages": base_messages,
                    "response": response,
                    "parsed_response": None,
                    "success": True,
                    "error": None,
                    "retry_count": attempt - 1,
                    "format_valid": True
                }

            is_valid, parsed_or_error = validator(response)

            if is_valid:
                parsed_response = parsed_or_error
                return {
                    "id": sample_id,
                    "messages": base_messages,
                    "response": response,
                    "parsed_response": parsed_response,
                    "success": True,
                    "error": None,
                    "retry_count": attempt - 1,
                    "format_valid": True
                }

            last_error = parsed_or_error
            current_messages = build_retry_messages(
                base_messages,
                response,
                parsed_or_error
            )

        except Exception as e:
            last_error = str(e)

        time.sleep(1)

    return {
        "id": sample_id,
        "messages": base_messages,
        "response": last_response,
        "parsed_response": None,
        "success": False,
        "error": last_error,
        "retry_count": max_retry,
        "format_valid": False
    }

def parallel_inference_dual_servers(samples, max_workers_per_server=8, max_retry=3, validator=None):
    """
    两个服务同时推理：
    - 每个服务单独一个线程池
    - 每个线程池各 8 worker
    - 样本轮询分配到两个服务
    """
    results = [None] * len(samples)
    start_time = time.time()

    # 两个线程池
    executors = {
        config["name"]: ThreadPoolExecutor(max_workers=max_workers_per_server)
        for config in API_CONFIGS
    }

    future_to_idx = {}

    try:
        # 轮询分配样本给两个服务
        for idx, sample in enumerate(samples):
            api_config = API_CONFIGS[idx % len(API_CONFIGS)]
            executor = executors[api_config["name"]]

            future = executor.submit(process_one_sample, sample, api_config, validator, max_retry)
            future_to_idx[future] = idx

        success_count = 0
        fail_count = 0

        with tqdm(total=len(samples), desc="Processing", ncols=100) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "id": samples[idx].get("id"),
                        "messages": samples[idx].get("messages", []),
                        "response": None,
                        "parsed_response": None,
                        "success": False,
                        "error": f"Worker crashed: {str(e)}",
                        "retry_count": max_retry,
                        "format_valid": False
                    }

                results[idx] = result

                if result["success"]:
                    success_count += 1
                else:
                    fail_count += 1

                pbar.update(1)
                pbar.set_postfix(success=success_count, fail=fail_count)

    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n全部完成，总耗时: {elapsed:.2f} 秒")
    print(f"平均每条耗时: {elapsed / len(samples):.2f} 秒")
    print(f"总样本数: {len(samples)}")
    print(f"总成功数: {sum(1 for x in results if x and x['success'])}")
    print(f"总失败数: {sum(1 for x in results if x and not x['success'])}")

    return results


if __name__ == "__main__":
    samples = [
        {
            "id": 1,
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "我邮箱登录不上"}
            ]
        },
        {
            "id": 2,
            "messages": [
                {"role": "user", "content": "帮我介绍一下机器学习"}
            ]
        },
        {
            "id": 3,
            "messages": [
                {"role": "user", "content": "今天天气怎么样"}
            ]
        },
        {
            "id": 4,
            "messages": [
                {"role": "user", "content": "Python 怎么读取 json 文件？"}
            ]
        }
    ]

    results = parallel_inference_dual_servers(
        samples,
        max_workers_per_server=MAX_WORKERS_PER_SERVER
    )

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("结果已保存到 results.json")
