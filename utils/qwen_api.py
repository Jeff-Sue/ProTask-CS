import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


OPENAI_API_KEY = "your_api_key"
OPENAI_API_BASE = "https://api.openai.com/v1"
MODEL_NAME = "qwen3-30b-a3b"

MAX_WORKERS = 8
MAX_RETRY = 3


def get_client():
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )


def call_qwen_api(messages):
    client = get_client()

    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
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


def process_one_sample(sample, validator=None, max_retry=MAX_RETRY):
    sample_id = sample.get("id")
    base_messages = sample.get("messages", [])

    current_messages = list(base_messages)
    last_error = None
    last_response = None
    parsed_response = None

    for attempt in range(1, max_retry + 1):
        try:
            response = call_qwen_api(current_messages)
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


def parallel_inference(samples, max_workers=8, max_retry=3, validator=None):
    results = [None] * len(samples)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_one_sample, sample, validator, max_retry): idx
            for idx, sample in enumerate(samples)
        }

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
                pbar.update(1)

                if result["success"]:
                    success_count += 1
                else:
                    fail_count += 1

                pbar.set_postfix(success=success_count, fail=fail_count)

    elapsed = time.time() - start_time
    print(f"\n全部完成，总耗时: {elapsed:.2f} 秒")
    print(f"平均每条耗时: {elapsed / len(samples):.2f} 秒")

    return results