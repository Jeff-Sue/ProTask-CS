import re
from urllib import response
from openai import OpenAI
import json

def qwen3_30b_a3b(prompt, user_message):
    openai_api_key = "your_api_key"  # 替换成你的API密钥
    openai_api_base = "https://api.openai.com/v1"  # 替换成你的API地址


    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )

    chat_response = client.chat.completions.create(
        model="qwen3-30b-a3b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
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