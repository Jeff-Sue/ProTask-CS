import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple


# =========================
# 1. Prompt Templates
# =========================
with open("prompts/detector_prompt.txt", "r", encoding="utf-8") as f:
    DETECTOR_PROMPT = f.read()

with open("prompts/time_judge.prompt.txt", "r", encoding="utf-8") as f:
    TIME_JUDGE_PROMPT = f.read()

with open("prompts/utility_judge.prompt.txt", "r", encoding="utf-8") as f:
    UTILITY_JUDGE_PROMPT = f.read()


# =========================
# 2. Data Structures
# =========================

@dataclass
class DetectorResult:
    has_case_usage: bool
    first_case_usage_turn: Optional[int]
    evidence_span: str
    reason: str


@dataclass
class TimingResult:
    problem_clear: bool
    missing_critical_info: bool
    recommendation_ready: bool
    timing_judgment: str
    reason: str


@dataclass
class UtilityResult:
    relevance: str
    solvability: str
    utility_judgment: str
    reason: str


# =========================
# 3. Core Evaluator
# =========================

class CaseRecommendationEvaluator:
    """
    输入完整对话，输出案例推荐是否准确。
    full_dialogue 格式固定为：
    [
        "用户：对话内容",
        "客服：对话内容",
        "用户：对话内容",
        "客服：对话内容",
        ...
    ]

    你只需要实现 _call_llm() 这个方法，就能接入你自己的模型服务。
    """

    def __init__(self, model_name: str = "your_model_name"):
        self.model_name = model_name

    # ---------- Public API ----------

    def evaluate(
        self,
        full_dialogue: List[str],
        retrieved_cases: str = "",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            full_dialogue: 完整对话，格式示例：
                [
                    "用户：你好，我这边登录异常",
                    "客服：您好，请问报什么错？",
                    "用户：提示权限不足",
                    "客服：建议您重新登录试试"
                ]
            retrieved_cases: 返回案例内容，可为空
            strict: True 表示严格版准确率；False 表示宽松版

        Returns:
            一个结构化字典，包含 detector、timing、utility 和最终准确性结论
        """
        self._validate_dialogue_list(full_dialogue)

        detector = self.detect_first_case_usage(full_dialogue)

        if not detector.has_case_usage or detector.first_case_usage_turn is None:
            return {
                "accurate": False,
                "mode": "strict" if strict else "relaxed",
                "detector": detector.__dict__,
                "timing": None,
                "utility": None,
                "summary": {
                    "has_case_usage": False,
                    "timing_pass": False,
                    "utility_pass": False,
                },
            }

        truncated_dialogue, first_response = self._truncate_dialogue_to_turn(
            full_dialogue=full_dialogue,
            turn_id=detector.first_case_usage_turn
        )

        timing = self.judge_timing(
            dialogue_until_first_case_usage=truncated_dialogue,
            first_case_usage_response=first_response,
            retrieved_cases=retrieved_cases
        )

        utility = self.judge_utility(
            dialogue_until_first_case_usage=truncated_dialogue,
            first_case_usage_response=first_response,
            retrieved_cases=retrieved_cases
        )

        timing_pass = timing.timing_judgment in {"acceptable", "borderline"}
        if strict:
            utility_pass = utility.utility_judgment == "useful"
        else:
            utility_pass = utility.utility_judgment in {"useful", "partially_useful"}

        accurate = detector.has_case_usage and timing_pass and utility_pass

        return {
            "accurate": accurate,
            "mode": "strict" if strict else "relaxed",
            "detector": detector.__dict__,
            "timing": timing.__dict__,
            "utility": utility.__dict__,
            "summary": {
                "has_case_usage": detector.has_case_usage,
                "timing_pass": timing_pass,
                "utility_pass": utility_pass,
            },
        }

    def detect_first_case_usage(self, full_dialogue: List[str]) -> DetectorResult:
        """
        检测是否存在首次案例使用轮。
        注意：这里约定 first_case_usage_turn 表示“第几轮客服回复”，从 1 开始计数。
        """
        dialogue_text = self._dialogue_list_to_text(full_dialogue)
        prompt = DETECTOR_PROMPT.format(full_dialogue=dialogue_text)
        result = self._call_and_parse_json(prompt)

        return DetectorResult(
            has_case_usage=bool(result.get("has_case_usage", False)),
            first_case_usage_turn=result.get("first_case_usage_turn"),
            evidence_span=str(result.get("evidence_span", "")),
            reason=str(result.get("reason", "")),
        )

    def judge_timing(
        self,
        dialogue_until_first_case_usage: str,
        first_case_usage_response: str,
        retrieved_cases: str = "",
    ) -> TimingResult:
        prompt = TIME_JUDGE_PROMPT.format(
            dialogue_until_first_case_usage=dialogue_until_first_case_usage,
            first_case_usage_response=first_case_usage_response,
            retrieved_cases=retrieved_cases or "无"
        )
        result = self._call_and_parse_json(prompt)

        return TimingResult(
            problem_clear=bool(result.get("problem_clear", False)),
            missing_critical_info=bool(result.get("missing_critical_info", False)),
            recommendation_ready=bool(result.get("recommendation_ready", False)),
            timing_judgment=str(result.get("timing_judgment", "too_early")),
            reason=str(result.get("reason", "")),
        )

    def judge_utility(
        self,
        dialogue_until_first_case_usage: str,
        first_case_usage_response: str,
        retrieved_cases: str = "",
    ) -> UtilityResult:
        prompt = UTILITY_JUDGE_PROMPT.format(
            dialogue_until_first_case_usage=dialogue_until_first_case_usage,
            first_case_usage_response=first_case_usage_response,
            retrieved_cases=retrieved_cases or "无"
        )
        result = self._call_and_parse_json(prompt)

        return UtilityResult(
            relevance=str(result.get("relevance", "low")),
            solvability=str(result.get("solvability", "weak")),
            utility_judgment=str(result.get("utility_judgment", "not_useful")),
            reason=str(result.get("reason", "")),
        )

    # ---------- Internal Helpers ----------

    def _truncate_dialogue_to_turn(self, full_dialogue: List[str], turn_id: int) -> Tuple[str, str]:
        """
        从完整对话中截断到指定“客服轮”，并返回：
        1. 截断后的对话文本
        2. 该轮客服原文（不带“客服：”前缀）

        例如：
        full_dialogue = [
            "用户：A",
            "客服：B",
            "用户：C",
            "客服：D"
        ]

        turn_id = 2
        返回：
        (
            "用户：A\n客服：B\n用户：C\n客服：D",
            "D"
        )
        """
        kept_lines: List[str] = []
        first_case_usage_response = ""
        agent_turn_count = 0

        for line in full_dialogue:
            if not isinstance(line, str):
                continue

            line = line.strip()
            if not line:
                continue

            kept_lines.append(line)

            if self._is_agent_line(line):
                agent_turn_count += 1
                if agent_turn_count == turn_id:
                    first_case_usage_response = self._strip_role_prefix(line)
                    break

        return "\n".join(kept_lines), first_case_usage_response

    def _dialogue_list_to_text(self, full_dialogue: List[str]) -> str:
        """
        将 List[str] 格式对话转为 prompt 使用的字符串。
        """
        cleaned_lines = []
        for line in full_dialogue:
            if not isinstance(line, str):
                continue
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _validate_dialogue_list(self, full_dialogue: List[str]) -> None:
        """
        对输入对话做基本校验。
        """
        if not isinstance(full_dialogue, list):
            raise TypeError("full_dialogue 必须是 List[str] 格式。")

        if len(full_dialogue) == 0:
            raise ValueError("full_dialogue 不能为空。")

        for i, line in enumerate(full_dialogue):
            if not isinstance(line, str):
                raise TypeError(f"full_dialogue[{i}] 不是字符串。")

            stripped = line.strip()
            if not stripped:
                continue

            if not (
                stripped.startswith("用户：") or stripped.startswith("用户:")
                or stripped.startswith("客服：") or stripped.startswith("客服:")
            ):
                raise ValueError(
                    f'full_dialogue[{i}] 格式不合法："{line}"。'
                    '每一项应以 "用户：" 或 "客服：" 开头。'
                )

    def _is_agent_line(self, line: str) -> bool:
        return line.startswith("客服：") or line.startswith("客服:")

    def _strip_role_prefix(self, line: str) -> str:
        """
        去掉 '用户：' / '客服：' 这类前缀，只保留内容。
        """
        return re.sub(r"^(用户|客服)[:：]\s*", "", line).strip()

    def _call_and_parse_json(self, prompt: str) -> Dict[str, Any]:
        raw = self._call_llm(prompt)
        return self._safe_parse_json(raw)

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """
        尽量鲁棒地解析 LLM 输出的 JSON
        """
        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        code_block_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except Exception:
                pass

        brace_match = re.search(r"(\{.*\})", text, flags=re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except Exception:
                pass

        raise ValueError(f"无法解析 LLM 输出为 JSON：\n{text}")

    def _call_llm(self, prompt: str) -> str:
        """
        这里改成你自己的模型调用逻辑。

        你可以接：
        - OpenAI SDK
        - vLLM OpenAI-compatible API
        - 内部 HTTP 服务
        - Qwen / DeepSeek / 自研服务
        """
        raise NotImplementedError("请实现 _call_llm()，接入你的模型服务。")


# =========================
# 4. Example: OpenAI-Compatible Implementation
# =========================

class OpenAICompatibleCaseRecommendationEvaluator(CaseRecommendationEvaluator):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "你是一个严格遵循输出格式的评测助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


# =========================
# 5. Batch Evaluation Utility
# =========================

def evaluate_dataset(
    evaluator: CaseRecommendationEvaluator,
    dataset: List[Dict[str, Any]],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    dataset 中每个元素格式示例：
    {
        "id": "sample_001",
        "full_dialogue": [
            "用户：...",
            "客服：...",
            "用户：...",
            "客服：..."
        ],
        "retrieved_cases": "..."
    }
    """
    results = []
    accurate_count = 0

    for item in dataset:
        sample_id = item.get("id", "")
        full_dialogue = item["full_dialogue"]
        retrieved_cases = item.get("retrieved_cases", "")

        result = evaluator.evaluate(
            full_dialogue=full_dialogue,
            retrieved_cases=retrieved_cases,
            strict=strict,
        )
        result["id"] = sample_id
        results.append(result)

        if result["accurate"]:
            accurate_count += 1

    total = len(results)
    accuracy = accurate_count / total if total > 0 else 0.0

    return {
        "mode": "strict" if strict else "relaxed",
        "total": total,
        "accurate_count": accurate_count,
        "accuracy": accuracy,
        "results": results,
    }


# =========================
# 6. Minimal Example
# =========================

if __name__ == "__main__":
    evaluator = OpenAICompatibleCaseRecommendationEvaluator(
        api_key="your_api_key",
        base_url="https://api.openai.com/v1",
        model_name="your_model_name",
        temperature=0.0,
        max_tokens=1024,
    )

    full_dialogue = [
        "用户：你好，我这边welink登录后一直报权限异常。",
        "客服：您好，请问您当前使用的是电脑端还是手机端？",
        "用户：电脑端。",
        "客服：好的，收到。",
        "用户：而且昨天还能用，今天突然不行了。",
        "客服：这种情况一般是权限未同步导致的，建议您先退出账号后重新登录试试。"
    ]

    retrieved_cases = """
案例1：
标题：WeLink 电脑端权限异常处理
内容：若用户突然出现权限异常，可先退出当前账号重新登录；若仍异常，检查组织权限同步状态。
""".strip()

    result = evaluator.evaluate(
        full_dialogue=full_dialogue,
        retrieved_cases=retrieved_cases,
        strict=True,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))