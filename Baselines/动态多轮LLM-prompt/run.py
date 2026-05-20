"""
动态多轮 LLM-prompt Runtime 环境

设计目标：
1. 将静态 prefix pipeline 改造成动态多轮交互环境；
2. 系统持续维护 dialogue_state，而不是每轮重新构造 gold prefix；
3. 使用 active_evidence 是否为空决定 retrieval 链路：
   - active_evidence 为空：Query -> Retrieval -> Trigger -> Policy -> Response
   - active_evidence 非空：Query -> Trigger -> Retrieval/Reuse/NoEvidence -> Policy -> Response
4. Query / Trigger / Policy / Response 均通过 prompt-based LLM 调用；
5. Retrieval 可替换为你已有的 rag(query, dialogue_history, use_chat)；
6. 每轮保存 query、trigger、retrieval、policy、response、latency 和 dialogue_state。

注意：
- 这是动态多轮 runtime，不再输出 1...N 的完整 trajectory；
- 每轮只输出当前 turn 的状态；
- 完整 trajectory 由多轮运行日志自然累积形成。
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ==================== 可选：接入你原工程里的 LLM / RAG ====================

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from utils.qwen_api import parallel_inference  # type: ignore
    from utils.scu_rag import rag  # type: ignore
except Exception:
    parallel_inference = None
    rag = None

# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== 枚举约束 ====================

ALLOWED_QUERY_MODES = {"none", "reuse", "refine", "rewrite"}

ALLOWED_EVIDENCE_MODES = {
    "no_evidence_needed",
    "retrieve_new_evidence",
    "reuse_existing_evidence",
}

ALLOWED_POLICY_LABELS = {
    "AskMissingSlot",
    "AskClarification",
    "ExplainedResponse",
    "CaseRecommendation",
    "Handoff",
    "ProcessAcknowledgement",
}

# ==================== 数据结构 ====================

@dataclass
class DialogueTurn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class RetrievalCase:
    case_id: str
    title: str
    content: str


@dataclass
class QueryState:
    query_mode: str = "none"
    query_text: str = ""
    reason: str = ""


@dataclass
class TriggerState:
    trigger: bool = False
    evidence_mode: str = "no_evidence_needed"
    reason: str = ""


@dataclass
class PolicyState:
    label: str = "ProcessAcknowledgement"
    reason: str = ""


@dataclass
class EvidenceState:
    """
    active_evidence 是当前已经被 Trigger 判断为可用的知识。
    本脚本只用 active_evidence 是否为空来决定动态链路。
    """
    active_evidence: List[RetrievalCase] = field(default_factory=list)
    evidence_source_query: str = ""
    last_retrieved_cases: List[RetrievalCase] = field(default_factory=list)


@dataclass
class DialogueState:
    dialogue_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    current_turn_id: int = 0
    query_state: QueryState = field(default_factory=QueryState)
    trigger_state: TriggerState = field(default_factory=TriggerState)
    policy_state: PolicyState = field(default_factory=PolicyState)
    evidence_state: EvidenceState = field(default_factory=EvidenceState)


@dataclass
class Latency:
    query_ms: int = 0
    retrieval_ms: int = 0
    trigger_ms: int = 0
    policy_ms: int = 0
    response_ms: int = 0
    total_ms: int = 0


@dataclass
class TurnResult:
    dialogue_id: str
    turn_id: int
    user_input: str
    query_state: QueryState
    retrieval_output: List[RetrievalCase]
    trigger_state: TriggerState
    policy_state: PolicyState
    response: str
    latency: Latency
    state_snapshot: Dict[str, Any]

# ==================== Prompt 管理 ====================

PROMPT_DIR = Path("prompts")
PROMPT_FILES = {
    "query": "dynamic_query.prompt",
    "trigger_use": "dynamic_trigger_use.prompt",
    "trigger_retrieval": "dynamic_trigger_retrieval.prompt",
    "policy": "dynamic_policy.prompt",
    "response": "dynamic_response.prompt",
}

DEFAULT_QUERY_PROMPT = """
你是一个 IT 客服动态多轮系统中的 Query 更新模块。
你的任务是根据当前完整对话历史、上一轮 query 状态和用户最新输入，输出当前轮 query 状态。

你必须输出 JSON，不要输出额外文本。

字段要求：
{
  "query_mode": "<none|reuse|refine|rewrite>",
  "query_text": "",
  "reason": ""
}

标签定义：
- none：当前用户输入不形成业务检索意图，例如寒暄、确认、感谢。
- reuse：当前用户只是补充信息或确认，核心问题不变，继续使用上一轮 query。
- refine：当前用户补充了新的约束、报错、系统名称或场景，需要细化上一轮 query。
- rewrite：当前用户提出了新问题或主题发生切换，需要重写 query。

约束：
- query_mode=none 时，query_text 必须为空字符串。
- query_mode!=none 时，query_text 必须是可用于检索的简洁查询。
""".strip()

DEFAULT_TRIGGER_USE_PROMPT = """
你是一个 IT 客服动态多轮系统中的 Knowledge-use Trigger 模块。
当前 active_evidence 为空，系统已经根据当前 query 进行了检索。
你的任务是判断 retrieved evidence 是否可用于当前轮回复。

你必须输出 JSON，不要输出额外文本。

字段要求：
{
  "trigger": true,
  "evidence_mode": "<no_evidence_needed|retrieve_new_evidence>",
  "reason": ""
}

判断规则：
- 如果 retrieved evidence 与当前问题匹配，并且足以支持客服当前回复，则 trigger=true, evidence_mode=retrieve_new_evidence。
- 如果 retrieved evidence 不匹配、过泛、证据不足，或者当前更应该追问澄清，则 trigger=false, evidence_mode=no_evidence_needed。

注意：
- 这里 Retrieval 已经发生，所以 evidence_mode=retrieve_new_evidence 表示“本轮新检索得到的 evidence 可被激活使用”。
- 不要输出 reuse_existing_evidence，因为当前 active_evidence 为空。
""".strip()

DEFAULT_TRIGGER_RETRIEVAL_PROMPT = """
你是一个 IT 客服动态多轮系统中的 Retrieval Trigger 模块。
当前系统已经存在 active_evidence。
你的任务是根据当前对话、当前 query 和已有 active_evidence，判断本轮是否需要知识，以及是否需要重新检索。

你必须输出 JSON，不要输出额外文本。

字段要求：
{
  "trigger": true,
  "evidence_mode": "<no_evidence_needed|reuse_existing_evidence|retrieve_new_evidence>",
  "reason": ""
}

判断规则：
- no_evidence_needed：当前轮只是确认、接收信息、流程告知、寒暄等，不需要使用案例知识。
- reuse_existing_evidence：当前问题仍被已有 active_evidence 覆盖，可以复用已有案例。
- retrieve_new_evidence：用户问题发生主题切换，或当前 query 明显超出已有 evidence 覆盖范围，需要重新检索。

约束：
- trigger=false 时，evidence_mode 必须是 no_evidence_needed。
- trigger=true 时，evidence_mode 只能是 reuse_existing_evidence 或 retrieve_new_evidence。
""".strip()

DEFAULT_POLICY_PROMPT = """
你是一个 IT 客服动态多轮系统中的 Policy 模块。
你的任务是根据当前对话历史、query 状态、trigger 状态和可用 evidence，判断当前客服回复应该采取哪一种服务动作。

你必须输出 JSON，不要输出额外文本。

字段要求：
{
  "label": "<AskMissingSlot|AskClarification|ExplainedResponse|CaseRecommendation|Handoff|ProcessAcknowledgement>",
  "reason": ""
}

标签定义：
- AskMissingSlot：缺少处理问题所需的关键客观信息，例如账号、系统名、报错截图、工号、设备环境。
- AskClarification：用户描述含糊，需要澄清具体问题、现象或意图。
- ExplainedResponse：可以直接解释原因、规则、概念或操作说明。
- CaseRecommendation：已有可用案例，应该给出具体解决方案或处理步骤。
- Handoff：当前问题需要人工、后台、工单或其他团队介入。
- ProcessAcknowledgement：确认收到、告知已记录、说明后续流程或等待处理。
""".strip()

DEFAULT_RESPONSE_PROMPT = """
你是一个 IT 客服回复生成模块。
你的任务是根据当前对话历史、query 状态、trigger 状态、policy 动作和可用 evidence，生成当前轮客服回复。

要求：
1. 回复自然、简洁、可执行；
2. 如果 policy 是 AskMissingSlot 或 AskClarification，优先提出必要问题，不要过早给方案；
3. 如果 policy 是 CaseRecommendation，应结合 evidence 给出具体步骤；
4. 如果 policy 是 ProcessAcknowledgement，应确认收到并说明后续处理；
5. 不要编造 evidence 中不存在的具体制度、链接或流程；
6. 只输出客服回复文本，不要输出 JSON。
""".strip()


def load_prompt(prompt_name: str, prompt_dir: Path = PROMPT_DIR) -> str:
    file_name = PROMPT_FILES[prompt_name]
    path = prompt_dir / file_name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    defaults = {
        "query": DEFAULT_QUERY_PROMPT,
        "trigger_use": DEFAULT_TRIGGER_USE_PROMPT,
        "trigger_retrieval": DEFAULT_TRIGGER_RETRIEVAL_PROMPT,
        "policy": DEFAULT_POLICY_PROMPT,
        "response": DEFAULT_RESPONSE_PROMPT,
    }
    logger.warning("Prompt file %s not found. Using built-in default prompt.", path)
    return defaults[prompt_name]


QUERY_PROMPT = load_prompt("query")
TRIGGER_USE_PROMPT = load_prompt("trigger_use")
TRIGGER_RETRIEVAL_PROMPT = load_prompt("trigger_retrieval")
POLICY_PROMPT = load_prompt("policy")
RESPONSE_PROMPT = load_prompt("response")

# ==================== 文本格式化 ====================


def format_dialogue_history(turns: Sequence[DialogueTurn]) -> str:
    if not turns:
        return "无"
    role_map = {"user": "用户", "assistant": "客服"}
    return "\n".join(f"{role_map.get(t.role, t.role)}：{t.text}" for t in turns)


def format_cases(cases: Sequence[RetrievalCase]) -> str:
    if not cases:
        return "无"
    parts = []
    for i, c in enumerate(cases, start=1):
        parts.append(
            f"{i}. case_id: {c.case_id}\n"
            f"   title: {c.title}\n"
            f"   content: {c.content}"
        )
    return "\n".join(parts)


def format_query_state(query_state: QueryState) -> str:
    return (
        f"query_mode: {query_state.query_mode}\n"
        f"query_text: {query_state.query_text}\n"
        f"reason: {query_state.reason}"
    )


def format_trigger_state(trigger_state: TriggerState) -> str:
    return (
        f"trigger: {str(trigger_state.trigger).lower()}\n"
        f"evidence_mode: {trigger_state.evidence_mode}\n"
        f"reason: {trigger_state.reason}"
    )


def format_policy_state(policy_state: PolicyState) -> str:
    return f"label: {policy_state.label}\nreason: {policy_state.reason}"

# ==================== LLM 调用层 ====================


def _extract_text_from_parallel_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ["content", "text", "response", "output"]:
            if key in result and isinstance(result[key], str):
                return result[key]
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ["content", "text", "response", "output"]:
                if key in first and isinstance(first[key], str):
                    return first[key]
    raise ValueError(f"Unrecognized LLM result format: {type(result)} -> {result}")


def call_llm_text(messages: List[Dict[str, str]]) -> str:
    """
    默认接入原工程 parallel_inference。
    如果你本地没有 parallel_inference，请替换这里。
    """
    if parallel_inference is None:
        raise RuntimeError(
            "parallel_inference is unavailable. "
            "Please replace call_llm_text with your own local/API LLM call."
        )

    item = {"messages": messages}
    result = parallel_inference([item])
    return _extract_text_from_parallel_result(result).strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            first = lines[0].strip().lower()
            if first in {"```", "```json"}:
                return "\n".join(lines[1:-1]).strip()
    return text


def call_llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    text = call_llm_text(messages)
    text = strip_code_fence(text)
    return json.loads(text)

# ==================== Retriever ====================


def default_retriever(query_text: str, dialogue_history: str) -> List[Dict[str, str]]:
    """
    默认调用原工程 rag。
    返回格式需要兼容：
    [
      {"case_id": "...", "title": "...", "content": "..."},
      ...
    ]
    """
    if rag is None:
        logger.warning("rag is unavailable. Return empty evidence.")
        return []
    return rag(query_text, dialogue_history, "true")


def normalize_cases(raw_cases: Sequence[Dict[str, Any]], top_k: int = 5) -> List[RetrievalCase]:
    cases: List[RetrievalCase] = []
    for case in list(raw_cases)[:top_k]:
        cases.append(
            RetrievalCase(
                case_id=str(case.get("case_id", case.get("caseId", ""))),
                title=str(case.get("title", "")),
                content=str(case.get("content", "")),
            )
        )
    return cases

# ==================== 输出解析与校验 ====================


def parse_query_state(raw: Dict[str, Any]) -> QueryState:
    state = QueryState(
        query_mode=str(raw.get("query_mode", "none")),
        query_text=str(raw.get("query_text", "") or ""),
        reason=str(raw.get("reason", "") or ""),
    )
    if state.query_mode not in ALLOWED_QUERY_MODES:
        raise ValueError(f"Invalid query_mode: {state.query_mode}")
    if state.query_mode == "none" and state.query_text.strip():
        raise ValueError("query_mode=none 时 query_text 必须为空")
    if state.query_mode != "none" and not state.query_text.strip():
        raise ValueError("query_mode 非 none 时 query_text 不能为空")
    return state


def parse_trigger_state(raw: Dict[str, Any], active_evidence_empty: bool) -> TriggerState:
    trigger = raw.get("trigger", False)
    if not isinstance(trigger, bool):
        raise ValueError(f"trigger must be bool, got: {trigger}")

    state = TriggerState(
        trigger=trigger,
        evidence_mode=str(raw.get("evidence_mode", "no_evidence_needed")),
        reason=str(raw.get("reason", "") or ""),
    )
    if state.evidence_mode not in ALLOWED_EVIDENCE_MODES:
        raise ValueError(f"Invalid evidence_mode: {state.evidence_mode}")
    if not state.trigger and state.evidence_mode != "no_evidence_needed":
        raise ValueError("trigger=False 时 evidence_mode 必须是 no_evidence_needed")
    if state.trigger and state.evidence_mode == "no_evidence_needed":
        raise ValueError("trigger=True 时 evidence_mode 不能是 no_evidence_needed")

    if active_evidence_empty and state.evidence_mode == "reuse_existing_evidence":
        raise ValueError("active_evidence 为空时不能输出 reuse_existing_evidence")

    return state


def parse_policy_state(raw: Dict[str, Any]) -> PolicyState:
    state = PolicyState(
        label=str(raw.get("label", "ProcessAcknowledgement")),
        reason=str(raw.get("reason", "") or ""),
    )
    if state.label not in ALLOWED_POLICY_LABELS:
        raise ValueError(f"Invalid policy label: {state.label}")
    return state

# ==================== 动态 Runtime ====================

class DynamicMultiTurnRuntime:
    def __init__(
        self,
        dialogue_id: str = "dynamic_dialogue",
        retriever: Callable[[str, str], List[Dict[str, str]]] = default_retriever,
        retrieval_top_k: int = 5,
        save_history_path: Optional[str] = None,
    ) -> None:
        self.state = DialogueState(dialogue_id=dialogue_id)
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.save_history_path = save_history_path
        self.turn_results: List[TurnResult] = []

    # ---------- 对外主入口 ----------

    def step(self, user_input: str) -> TurnResult:
        """
        动态多轮环境的一轮执行。
        输入用户新 utterance，输出当前轮系统回复和完整模块状态。
        """
        total_start = time.time()
        self.state.current_turn_id += 1
        turn_id = self.state.current_turn_id
        logger.info("Processing dynamic turn %s", turn_id)

        # 0. 追加用户输入
        self.state.turns.append(DialogueTurn(role="user", text=user_input))

        # 1. Query Update
        query_state, query_ms = self.run_query_update(user_input=user_input)
        self.state.query_state = query_state

        retrieval_ms = 0
        response_cases: List[RetrievalCase] = []

        # query_mode=none 时，不走检索，也不激活 evidence
        if query_state.query_mode == "none":
            trigger_state = TriggerState(
                trigger=False,
                evidence_mode="no_evidence_needed",
                reason="当前用户输入未形成业务检索意图。",
            )
            trigger_ms = 0
            self.state.trigger_state = trigger_state

        else:
            # 2. 根据 active_evidence 是否为空决定链路
            if not self.state.evidence_state.active_evidence:
                # Cold-start: Query -> Retrieval -> Trigger(use)
                response_cases, retrieval_ms = self.run_retrieval(query_state.query_text)
                self.state.evidence_state.last_retrieved_cases = response_cases

                trigger_state, trigger_ms = self.run_trigger_use(
                    retrieved_cases=response_cases,
                )
                self.state.trigger_state = trigger_state

                # Trigger 判断新检索 evidence 可用，则写入 active_evidence
                if trigger_state.trigger and trigger_state.evidence_mode == "retrieve_new_evidence":
                    self.state.evidence_state.active_evidence = response_cases
                    self.state.evidence_state.evidence_source_query = query_state.query_text

            else:
                # Stateful: Query -> Trigger(retrieval) -> Retrieval/Reuse/NoEvidence
                trigger_state, trigger_ms = self.run_trigger_retrieval()
                self.state.trigger_state = trigger_state

                if trigger_state.evidence_mode == "retrieve_new_evidence":
                    response_cases, retrieval_ms = self.run_retrieval(query_state.query_text)
                    self.state.evidence_state.last_retrieved_cases = response_cases
                    self.state.evidence_state.active_evidence = response_cases
                    self.state.evidence_state.evidence_source_query = query_state.query_text

                elif trigger_state.evidence_mode == "reuse_existing_evidence":
                    response_cases = list(self.state.evidence_state.active_evidence)
                    retrieval_ms = 0

                elif trigger_state.evidence_mode == "no_evidence_needed":
                    response_cases = []
                    retrieval_ms = 0

                else:
                    raise ValueError(f"Unexpected evidence_mode: {trigger_state.evidence_mode}")

        # 3. Policy
        policy_state, policy_ms = self.run_policy(response_cases=response_cases)
        self.state.policy_state = policy_state

        # 4. Response
        response, response_ms = self.run_response(response_cases=response_cases)

        # 5. 追加客服回复
        self.state.turns.append(DialogueTurn(role="assistant", text=response))

        total_ms = int((time.time() - total_start) * 1000)
        latency = Latency(
            query_ms=query_ms,
            retrieval_ms=retrieval_ms,
            trigger_ms=trigger_ms,
            policy_ms=policy_ms,
            response_ms=response_ms,
            total_ms=total_ms,
        )

        result = TurnResult(
            dialogue_id=self.state.dialogue_id,
            turn_id=turn_id,
            user_input=user_input,
            query_state=query_state,
            retrieval_output=response_cases,
            trigger_state=self.state.trigger_state,
            policy_state=policy_state,
            response=response,
            latency=latency,
            state_snapshot=self.snapshot_state(),
        )

        self.turn_results.append(result)
        self._autosave()
        return result

    # ---------- 各模块 ----------

    def run_query_update(self, user_input: str) -> Tuple[QueryState, int]:
        start = time.time()
        user_prompt = f"""
下面是当前动态多轮客服对话状态。

【完整对话历史】
{format_dialogue_history(self.state.turns)}

【上一轮 query 状态】
{format_query_state(self.state.query_state)}

【用户最新输入】
{user_input}

请输出当前轮 query 状态 JSON。
""".strip()

        messages = [
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm_json(messages)
        output = parse_query_state(raw)
        elapsed_ms = int((time.time() - start) * 1000)
        return output, elapsed_ms

    def run_retrieval(self, query_text: str) -> Tuple[List[RetrievalCase], int]:
        start = time.time()
        dialogue_history = format_dialogue_history(self.state.turns)
        raw_cases = self.retriever(query_text, dialogue_history)
        cases = normalize_cases(raw_cases, top_k=self.retrieval_top_k)
        elapsed_ms = int((time.time() - start) * 1000)
        return cases, elapsed_ms

    def run_trigger_use(self, retrieved_cases: List[RetrievalCase]) -> Tuple[TriggerState, int]:
        """
        active_evidence 为空时使用。
        此时 Retrieval 已经发生，Trigger 判断 retrieved evidence 是否可用。
        """
        start = time.time()
        user_prompt = f"""
下面是当前动态多轮客服对话状态。

【完整对话历史】
{format_dialogue_history(self.state.turns)}

【当前 query 状态】
{format_query_state(self.state.query_state)}

【当前新检索 retrieved evidence】
{format_cases(retrieved_cases)}

【当前 active evidence】
无

请判断 retrieved evidence 是否可用于当前轮客服回复，并输出 Trigger JSON。
""".strip()

        messages = [
            {"role": "system", "content": TRIGGER_USE_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm_json(messages)
        output = parse_trigger_state(raw, active_evidence_empty=True)
        elapsed_ms = int((time.time() - start) * 1000)
        return output, elapsed_ms

    def run_trigger_retrieval(self) -> Tuple[TriggerState, int]:
        """
        active_evidence 非空时使用。
        Trigger 基于 previous active_evidence 判断复用、重新检索或不使用 evidence。
        """
        start = time.time()
        user_prompt = f"""
下面是当前动态多轮客服对话状态。

【完整对话历史】
{format_dialogue_history(self.state.turns)}

【当前 query 状态】
{format_query_state(self.state.query_state)}

【已有 active evidence】
{format_cases(self.state.evidence_state.active_evidence)}

【active evidence 来源 query】
{self.state.evidence_state.evidence_source_query or "无"}

请判断当前轮是否需要使用知识、复用已有 evidence 或重新检索，并输出 Trigger JSON。
""".strip()

        messages = [
            {"role": "system", "content": TRIGGER_RETRIEVAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm_json(messages)
        output = parse_trigger_state(raw, active_evidence_empty=False)
        elapsed_ms = int((time.time() - start) * 1000)
        return output, elapsed_ms

    def run_policy(self, response_cases: List[RetrievalCase]) -> Tuple[PolicyState, int]:
        start = time.time()
        user_prompt = f"""
下面是当前动态多轮客服对话状态。

【完整对话历史】
{format_dialogue_history(self.state.turns)}

【当前 query 状态】
{format_query_state(self.state.query_state)}

【当前 trigger 状态】
{format_trigger_state(self.state.trigger_state)}

【本轮可用于决策的 evidence】
{format_cases(response_cases)}

请判断当前轮客服最应该采取的 policy，并输出 Policy JSON。
""".strip()

        messages = [
            {"role": "system", "content": POLICY_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm_json(messages)
        output = parse_policy_state(raw)
        elapsed_ms = int((time.time() - start) * 1000)
        return output, elapsed_ms

    def run_response(self, response_cases: List[RetrievalCase]) -> Tuple[str, int]:
        start = time.time()
        user_prompt = f"""
下面是当前动态多轮客服对话状态。

【完整对话历史】
{format_dialogue_history(self.state.turns)}

【当前 query 状态】
{format_query_state(self.state.query_state)}

【当前 trigger 状态】
{format_trigger_state(self.state.trigger_state)}

【当前 policy 状态】
{format_policy_state(self.state.policy_state)}

【本轮可用于回复的 evidence】
{format_cases(response_cases)}

请生成当前轮客服回复。
""".strip()

        messages = [
            {"role": "system", "content": RESPONSE_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = call_llm_text(messages).strip()
        elapsed_ms = int((time.time() - start) * 1000)
        return response, elapsed_ms

    # ---------- 状态与保存 ----------

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "dialogue_id": self.state.dialogue_id,
            "current_turn_id": self.state.current_turn_id,
            "dialogue_history": [asdict(t) for t in self.state.turns],
            "query_state": asdict(self.state.query_state),
            "trigger_state": asdict(self.state.trigger_state),
            "policy_state": asdict(self.state.policy_state),
            "evidence_state": asdict(self.state.evidence_state),
        }

    def reset(self, dialogue_id: Optional[str] = None) -> None:
        dialogue_id = dialogue_id or self.state.dialogue_id
        self.state = DialogueState(dialogue_id=dialogue_id)
        self.turn_results = []

    def save(self, output_path: str) -> None:
        data = [asdict(r) for r in self.turn_results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Dynamic runtime results saved to %s", output_path)

    def _autosave(self) -> None:
        if self.save_history_path:
            self.save(self.save_history_path)

# ==================== 命令行交互环境 ====================


def print_turn_result(result: TurnResult) -> None:
    print("\n" + "=" * 80)
    print(f"Turn {result.turn_id}")
    print("-" * 80)
    print(f"用户输入：{result.user_input}")
    print("\n[Query]")
    print(json.dumps(asdict(result.query_state), ensure_ascii=False, indent=2))
    print("\n[Trigger]")
    print(json.dumps(asdict(result.trigger_state), ensure_ascii=False, indent=2))
    print("\n[Policy]")
    print(json.dumps(asdict(result.policy_state), ensure_ascii=False, indent=2))
    print("\n[Evidence Used This Turn]")
    print(json.dumps([asdict(c) for c in result.retrieval_output], ensure_ascii=False, indent=2))
    print("\n[Response]")
    print(result.response)
    print("\n[Latency]")
    print(json.dumps(asdict(result.latency), ensure_ascii=False, indent=2))
    print("=" * 80 + "\n")


def interactive_main() -> None:
    runtime = DynamicMultiTurnRuntime(
        dialogue_id="interactive_dialogue",
        retrieval_top_k=5,
        save_history_path="dynamic_runtime_output.json",
    )

    print("动态多轮 IT 客服 Runtime 已启动。")
    print("输入用户消息后回车。输入 /exit 退出，/reset 重置对话，/state 查看状态。")

    while True:
        user_input = input("用户：").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            runtime.save("dynamic_runtime_output.json")
            print("已退出，并保存到 dynamic_runtime_output.json")
            break
        if user_input == "/reset":
            runtime.reset(dialogue_id="interactive_dialogue")
            print("对话已重置。")
            continue
        if user_input == "/state":
            print(json.dumps(runtime.snapshot_state(), ensure_ascii=False, indent=2))
            continue

        try:
            result = runtime.step(user_input)
            print_turn_result(result)
        except Exception as e:
            logger.exception("Turn failed")
            print(f"当前轮执行失败：{e}")


if __name__ == "__main__":
    interactive_main()