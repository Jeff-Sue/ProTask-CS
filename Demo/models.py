import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Base Interfaces
# =========================
class BaseStrategyModel(ABC):
    @abstractmethod
    def predict(self, history: List[Dict], user_input: str) -> str:
        pass


class BaseResponseModel(ABC):
    @abstractmethod
    def generate(
        self,
        history: List[Dict],
        user_input: str,
        strategy: str,
        extra_context: Optional[Dict] = None,
    ) -> str:
        pass


# =========================
# Shared Qwen LoRA Base
# =========================
class QwenLoRABase:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str,
        use_fp16: bool = True,
        use_bf16: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        trust_remote_code: bool = True,
        device: str = "cuda",
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.trust_remote_code = trust_remote_code
        self.device = device

        if use_bf16:
            dtype = torch.bfloat16
        elif use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=None,
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.eval().to(device)

        # 兼容 Qwen chat 格式
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_end_id is None:
            self.im_end_id = self.tokenizer.eos_token_id

    def _chat_template_or_fallback(self, messages: List[Dict]) -> str:
        """
        优先使用 tokenizer.apply_chat_template；
        如果当前 tokenizer 不支持，则退回到手工拼接。
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return text
            except Exception:
                pass

        # fallback
        chunks = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            chunks.append(f"{role}: {content}")
        chunks.append("assistant:")
        return "\n".join(chunks)

    def _generate_text(self, prompt_text: str) -> str:
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.im_end_id,
        )

        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        # 只取新增部分
        gen_ids = outputs[0][input_ids.shape[1]:]
        pred_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        pred_text = self._extract_prediction(pred_text)
        return pred_text.strip()

    @staticmethod
    def _extract_prediction(text: str) -> str:
        """
        参考你原来的 extract_prediction 思路。
        尽量截断到特殊符号前，避免把后续垃圾内容带出来。
        """
        stop_markers = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
        ]
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0]

        return text.strip()


# =========================
# Strategy Model
# =========================
class QwenLoRAStrategyModel(QwenLoRABase, BaseStrategyModel):
    """
    策略模型输出:
    - 直接回复
    - 调RAG
    - 调子agent
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

    def build_messages(self, history: List[Dict], user_input: str) -> List[Dict]:
        history_text = self._history_to_text(history)

        system_prompt = (
            "你是一个多轮对话系统中的策略分类器。\n"
            "你的任务是根据对话历史和当前用户输入，"
            "从以下三种策略中选择且只能选择一种：\n"
            "1. 直接回复\n"
            "2. 调RAG\n"
            "3. 调子agent\n\n"
            "输出要求：\n"
            "只输出策略名称本身，不要解释，不要输出多余内容。\n"
            "合法输出只能是：直接回复 / 调RAG / 调子agent"
        )

        user_prompt = (
            f"对话历史：\n{history_text}\n\n"
            f"当前用户输入：\n用户：{user_input}\n\n"
            f"请输出策略："
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _history_to_text(self, history: List[Dict]) -> str:
        if not history:
            return "无"

        lines = []
        for turn in history:
            role = turn.get("role", "")
            if role == "user":
                lines.append(f"用户：{turn.get('content', '')}")
            elif role == "assistant":
                strategy = turn.get("strategy", "")
                if strategy:
                    lines.append(f"助手（策略={strategy}）：{turn.get('content', '')}")
                else:
                    lines.append(f"助手：{turn.get('content', '')}")
        return "\n".join(lines)

    def _normalize_strategy(self, text: str) -> str:
        text = text.strip()

        # 优先精确匹配
        if text in {"直接回复", "调RAG", "调子agent"}:
            return text

        # 再做包含匹配，增强鲁棒性
        if "调子agent" in text or "子agent" in text:
            return "调子agent"
        if "调RAG" in text or "RAG" in text:
            return "调RAG"
        if "直接回复" in text:
            return "直接回复"

        # 最后兜底
        return "直接回复"

    def predict(self, history: List[Dict], user_input: str) -> str:
        messages = self.build_messages(history, user_input)
        prompt_text = self._chat_template_or_fallback(messages)
        raw_output = self._generate_text(prompt_text)
        strategy = self._normalize_strategy(raw_output)
        return strategy


# =========================
# Response Model
# =========================
class QwenLoRAResponseModel(QwenLoRABase, BaseResponseModel):
    """
    回复模型输入：
    - 对话历史
    - 当前用户输入
    - 已预测策略
    - 可选 extra_context（如 RAG 结果 / 子agent结果）
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def build_messages(
        self,
        history: List[Dict],
        user_input: str,
        strategy: str,
        extra_context: Optional[Dict] = None,
    ) -> List[Dict]:
        extra_context = extra_context or {}
        history_messages = self._history_to_messages(history)

        rag_result = extra_context.get("rag_result", "")
        agent_result = extra_context.get("agent_result", "")

        system_prompt = (
            "你是一个智能客服对话助手。\n"
            "你需要根据给定的策略、对话历史、当前用户输入，以及可能提供的外部信息，"
            "生成自然、简洁、准确的回复。\n\n"
            "要求：\n"
            "1. 回复要贴合上下文。\n"
            "2. 不要暴露内部推理过程。\n"
            "3. 如果策略是调RAG，要优先利用提供的检索信息。\n"
            "4. 如果策略是调子agent，要优先结合子agent执行结果。\n"
            "5. 不要输出“策略：xxx”这类内部字段，只输出最终客服回复。"
        )

        extra_text = (
            f"当前策略：{strategy}\n\n"
            f"RAG结果：\n{rag_result if rag_result else '无'}\n\n"
            f"子agent结果：\n{agent_result if agent_result else '无'}"
        )

        current_user = (
            f"{extra_text}\n\n"
            f"当前用户输入：\n{user_input}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": current_user})
        return messages

    def _history_to_messages(self, history: List[Dict]) -> List[Dict]:
        messages = []
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": content})
        return messages

    def generate(
        self,
        history: List[Dict],
        user_input: str,
        strategy: str,
        extra_context: Optional[Dict] = None,
    ) -> str:
        messages = self.build_messages(
            history=history,
            user_input=user_input,
            strategy=strategy,
            extra_context=extra_context,
        )
        prompt_text = self._chat_template_or_fallback(messages)
        raw_output = self._generate_text(prompt_text)
        return self._clean_response(raw_output)

    @staticmethod
    def _clean_response(text: str) -> str:
        text = text.strip()

        # 去掉一些可能出现的前缀
        text = re.sub(r"^(回复|答复|assistant)\s*[:：]\s*", "", text, flags=re.I)

        return text.strip()