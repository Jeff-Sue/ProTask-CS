
"""静态多轮 LLM-prompt 框架
核心思路：
1. 将完整对话展开成多个 prefix 样本；
2. 每个 prefix 独立运行同一套链路；
3. Query / Trigger / Policy 输出 trajectory；
4. Retrieval / Response 只服务于当前 prefix 的最后一轮；
5. 记录每个 prefix, 每个模块的耗时。
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

# 如果你自己的工程里有 parallel_inference，就保留；没有就删掉并替换 call_llm_* 两个函数。
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from utils.qwen_api import parallel_inference  # type: ignore
    from utils.scu_rag import rag
except Exception:
    parallel_inference = None

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 数据结构定义 ====================

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

@dataclass
class Dialogue:
    """完整对话"""
    dialog_id: str
    turns: List[Dict[str, str]]  # [{"role": "user"|"assistant", "text": "..."}]

@dataclass
class PrefixSample:
    """一个 prefix 样本，对应一个客服目标轮"""
    dialog_id: str
    turn_id: int                      # 第几个客服目标轮
    history: List[Dict[str, str]]     # 不包含当前金标客服回复
    target_role: str                  # 固定 assistant
    gold_response: str                # 当前客服金标回复
    target_turn_index: int            # 在原始 dialogue.turns 中该 assistant 所在位置

@dataclass
class QueryTrajectoryStep:
    turn_id: int
    query_mode: str
    query_text: str
    reason: str = ""

@dataclass
class QueryTrajectoryOutput:
    trajectory: List[QueryTrajectoryStep] = field(default_factory=list)

@dataclass
class RetrievalCase:
    case_id: str
    title: str
    content: str

@dataclass
class RetrievalOutput:
    retrieved_cases: List[RetrievalCase] = field(default_factory=list)

@dataclass
class TriggerTrajectoryStep:
    turn_id: int
    trigger: bool
    evidence_mode: str
    reason: str = ""

@dataclass
class TriggerTrajectoryOutput:
    trajectory: List[TriggerTrajectoryStep] = field(default_factory=list)

@dataclass
class PolicyTrajectoryStep:
    turn_id: int
    label: str
    reason: str = ""

@dataclass
class PolicyTrajectoryOutput:
    trajectory: List[PolicyTrajectoryStep] = field(default_factory=list)

@dataclass
class ResponseOutput:
    response: str

@dataclass
class Latency:
    query_ms: int
    retrieval_ms: int
    trigger_ms: int
    policy_ms: int
    response_ms: int
    total_ms: int

@dataclass
class PrefixResult:
    dialog_id: str
    turn_id: int
    prefix_sample: PrefixSample
    query_output: QueryTrajectoryOutput
    retrieval_output: RetrievalOutput
    trigger_output: TriggerTrajectoryOutput
    policy_output: PolicyTrajectoryOutput
    response_output: ResponseOutput
    latency: Latency

# ==================== Prompt 管理 ====================

PROMPT_DIR = Path("prompts")
PROMPT_FILES = {
    "query": "query.prompt",
    "trigger": "trigger.prompt",
    "policy": "policy.prompt",
    "response": "response.prompt",
}

def load_prompt(prompt_name: str, prompt_dir: Path = PROMPT_DIR) -> str:
    file_name = PROMPT_FILES[prompt_name]
    path = prompt_dir / file_name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

QUERY_PROMPT = load_prompt("query")
TRIGGER_PROMPT = load_prompt("trigger")
POLICY_PROMPT = load_prompt("policy")
RESPONSE_PROMPT = load_prompt("response")

# ==================== 第1层：数据层 ====================

def load_dialogues(data_path: str) -> List[Dialogue]:
    """加载对话数据，支持 list 或 dict 两种常见格式。"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues: List[Dialogue] = []

    if isinstance(data, list):
        for item in data:
            dialogues.append(Dialogue(dialog_id=item["dialog_id"], turns=item["turns"]))
    elif isinstance(data, dict):
        for dialog_id, item in data.items():
            turns = item["turns"] if isinstance(item, dict) and "turns" in item else item
            dialogues.append(Dialogue(dialog_id=dialog_id, turns=turns))
    else:
        raise ValueError("Unsupported dialogue json format.")

    logger.info("Loaded %s dialogues", len(dialogues))
    return dialogues

# ==================== 第2层：Prefix构造层 ====================

def build_prefix_samples(dialogue: Dialogue) -> List[PrefixSample]:
    """将完整对话拆成多个 prefix 样本。每个 assistant 回复点对应一个样本。"""
    prefix_samples: List[PrefixSample] = []
    assistant_turn_id = 0

    for idx, turn in enumerate(dialogue.turns):
        if turn.get("role") != "assistant":
            continue

        assistant_turn_id += 1
        prefix_samples.append(
            PrefixSample(
                dialog_id=dialogue.dialog_id,
                turn_id=assistant_turn_id,
                history=dialogue.turns[:idx],
                target_role="assistant",
                gold_response=turn.get("text", ""),
                target_turn_index=idx,
            )
        )

    return prefix_samples

# ==================== 文本格式化函数 ====================

def format_history(turns: Sequence[Dict[str, str]]) -> str:
    if not turns:
        return "无"
    role_map = {"user": "用户", "assistant": "客服"}
    return "\n".join(
        f"{role_map.get(t.get('role', ''), t.get('role', 'unknown'))}：{t.get('text', '')}"
        for t in turns
    )


def format_retrieved_cases(retrieved_cases: Sequence[RetrievalCase]) -> str:
    if not retrieved_cases:
        return "无"
    parts = []
    for i, c in enumerate(retrieved_cases, start=1):
        parts.append(
            f"{i}. case_id: {c.case_id}\n"
            f"   title: {c.title}\n"
            f"   content: {c.content}"
        )
    return "\n".join(parts)


def format_query_trajectory(query_output: QueryTrajectoryOutput) -> str:
    if not query_output.trajectory:
        return "无"
    parts = []
    for step in query_output.trajectory:
        parts.append(
            f"turn {step.turn_id}:\n"
            f"- query_mode: {step.query_mode}\n"
            f"- query_text: {step.query_text}\n"
            f"- reason: {step.reason}"
        )
    return "\n\n".join(parts)


def format_trigger_trajectory(trigger_output: TriggerTrajectoryOutput) -> str:
    if not trigger_output.trajectory:
        return "无"
    parts = []
    for step in trigger_output.trajectory:
        parts.append(
            f"turn {step.turn_id}:\n"
            f"- trigger: {str(step.trigger).lower()}\n"
            f"- evidence_mode: {step.evidence_mode}\n"
            f"- reason: {step.reason}"
        )
    return "\n\n".join(parts)


def format_policy_trajectory(policy_output: PolicyTrajectoryOutput) -> str:
    if not policy_output.trajectory:
        return "无"
    parts = []
    for step in policy_output.trajectory:
        parts.append(
            f"turn {step.turn_id}:\n"
            f"- label: {step.label}\n"
            f"- reason: {step.reason}"
        )
    return "\n\n".join(parts)

# ==================== LLM 调用层 ====================

def _extract_text_from_parallel_result(result: Any) -> str:
    """尽量兼容常见 parallel_inference 返回结构。"""
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
    """你可以在这里替换成自己项目中的 API 调用逻辑。"""
    if parallel_inference is None:
        raise RuntimeError("parallel_inference is unavailable. Please replace call_llm_text with your own API call.")


    item = {}
    item["messages"] = messages

    result = parallel_inference([item])
    return _extract_text_from_parallel_result(result).strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def call_llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    text = call_llm_text(messages)
    text = strip_code_fence(text)
    return json.loads(text)

# ==================== 检索接口 ====================

def default_retriever(query_text: str, dialog_history: str) -> List[Dict[str, str]]:
    """默认空检索器。请替换成你的真实检索逻辑。"""
    evidence = rag(query_text, dialog_history, "true")
    logger.warning("Using default empty retriever for query: %s", query_text)
    return evidence

# ==================== 输出解析与校验 ====================

def _validate_turn_ids(trajectory: Sequence[Any], expected_turns: int, name: str) -> None:
    got = [getattr(x, "turn_id", None) for x in trajectory]
    expected = list(range(1, expected_turns + 1))
    if got != expected:
        raise ValueError(f"{name} turn_id invalid. expected={expected}, got={got}")


def parse_query_output(raw: Dict[str, Any], expected_turns: int) -> QueryTrajectoryOutput:
    traj_raw = raw.get("trajectory", [])
    traj = []
    for item in traj_raw:
        step = QueryTrajectoryStep(
            turn_id=int(item["turn_id"]),
            query_mode=item["query_mode"],
            query_text=item.get("query_text", "") or "",
            reason=item.get("reason", "") or "",
        )
        if step.query_mode not in ALLOWED_QUERY_MODES:
            raise ValueError(f"Invalid query_mode: {step.query_mode}")
        if step.query_mode == "none" and step.query_text != "":
            raise ValueError("query_mode=none 时 query_text 必须为空")
        if step.query_mode != "none" and not step.query_text.strip():
            raise ValueError("非 none 的 query_mode 必须有 query_text")
        traj.append(step)

    _validate_turn_ids(traj, expected_turns, "query trajectory")
    return QueryTrajectoryOutput(trajectory=traj)


def parse_trigger_output(raw: Dict[str, Any], expected_turns: int) -> TriggerTrajectoryOutput:
    traj_raw = raw.get("trajectory", [])
    traj = []
    for item in traj_raw:
        trigger = item["trigger"]
        if not isinstance(trigger, bool):
            raise ValueError(f"trigger must be bool, got: {trigger}")
        step = TriggerTrajectoryStep(
            turn_id=int(item["turn_id"]),
            trigger=trigger,
            evidence_mode=item["evidence_mode"],
            reason=item.get("reason", "") or "",
        )
        if step.evidence_mode not in ALLOWED_EVIDENCE_MODES:
            raise ValueError(f"Invalid evidence_mode: {step.evidence_mode}")
        if not step.trigger and step.evidence_mode != "no_evidence_needed":
            raise ValueError("trigger=False 时 evidence_mode 必须是 no_evidence_needed")
        if step.trigger and step.evidence_mode == "no_evidence_needed":
            raise ValueError("trigger=True 时 evidence_mode 不能是 no_evidence_needed")
        traj.append(step)

    _validate_turn_ids(traj, expected_turns, "trigger trajectory")
    return TriggerTrajectoryOutput(trajectory=traj)


def parse_policy_output(raw: Dict[str, Any], expected_turns: int) -> PolicyTrajectoryOutput:
    traj_raw = raw.get("trajectory", [])
    traj = []
    for item in traj_raw:
        step = PolicyTrajectoryStep(
            turn_id=int(item["turn_id"]),
            label=item["label"],
            reason=item.get("reason", "") or "",
        )
        if step.label not in ALLOWED_POLICY_LABELS:
            raise ValueError(f"Invalid policy label: {step.label}")
        traj.append(step)

    _validate_turn_ids(traj, expected_turns, "policy trajectory")
    return PolicyTrajectoryOutput(trajectory=traj)

# ==================== 第3层：单个Prefix的Pipeline层 ====================

def format_history_with_turn_ids(prefix_sample: PrefixSample) -> str:
    """
    按客服目标轮格式化 prefix 历史
    """
    history = prefix_sample.history
    blocks = []
    current_turn = 1
    current_block = []

    for turn in history:
        role = "用户" if turn["role"] == "user" else "客服"
        current_block.append(f"{role}：{turn['text']}")

        # 每遇到一个 assistant，说明这个客服目标轮结束
        if turn["role"] == "assistant":
            block_text = "\n".join(current_block)
            blocks.append(f"[Turn {current_turn} | 客服目标轮]\n{block_text}")
            current_turn += 1
            current_block = []

    # history 最后如果还有未闭合内容，说明这是当前待回复轮
    if current_block:
        block_text = "\n".join(current_block)
        blocks.append(f"[Turn {current_turn} | 当前待生成回复的客服目标轮]\n{block_text}")

    return "\n\n".join(blocks)


def build_turn_skeleton(turn_count: int, fields: dict[str, str]) -> str:
    """
    根据 turn_count 和字段模板，生成固定长度的 trajectory JSON 骨架文本
    fields: 例如
    {
    "query_mode": '"<none|reuse|refine|rewrite>"',
    "query_text": '""',
    "reason": '""'
    }
    """
    items = []
    for i in range(1, turn_count + 1):
        lines = [f' "turn_id": {i}']
        for k, v in fields.items():
            lines.append(f' "{k}": {v}')
        item = "{\n" + ",\n".join(lines) + "\n }"
        items.append(item)
    return '{\n "trajectory": [\n ' + ",\n ".join(items) + '\n ]\n}'


def run_query_model(prefix_sample: PrefixSample) -> Tuple[QueryTrajectoryOutput, int]:
    start_time = time.time()
    dialogue_prefix = format_history_with_turn_ids(prefix_sample)
    
    query_output_schema = build_turn_skeleton(
        turn_count=prefix_sample.turn_id,
        fields={
        "query_mode": '"<none|reuse|refine|rewrite>"',
        "query_text": '""',
        "reason": '""'
        }
    )

    user_prompt = f"""
下面是当前 prefix，对应的是“当前客服回复之前”的完整对话历史。

请你只对这个 prefix 中**已经形成的固定客服应答点**做标注，输出 query trajectory。

【重要约束】
1. 当前 prefix 的固定客服目标轮数为：{prefix_sample.turn_id}
2. 你必须且只能输出 turn_id = 1 到 turn_id = {prefix_sample.turn_id}
3. 不能新增任何 turn_id
4. 不能遗漏任何 turn_id
5. 不能只输出最后一个 turn_id
6. 不能脑补未来用户发言、未来客服回复或未来轮次
7. 你的任务是“对固定轮次做标注”，不是“预测未来对话”或“补全轨迹”

【特别提醒】
- 当 target_turn_count = 1 时，即使输入中还没有出现任何客服文本，也必须输出 turn_id = 1
- 当 target_turn_count = 2 时，必须完整输出 turn_id = 1 和 turn_id = 2，不能只输出 turn_id = 2
- 当 target_turn_count = N 时，你要重新标注从 1 到 N 的完整 trajectory，而不是只标最后一轮

【对话历史】
{dialogue_prefix}

**【请严格按照下面的 JSON 模板填写，不要改动字段名，不要新增字段，不要输出模板外内容，不要遗忘字段】**
{query_output_schema}

若输出 turn 数不等于 target_turn_count，答案视为无效；请自行修正后再输出。
""".strip()

    messages = [
        {"role": "system", "content": QUERY_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_llm_json(messages)
    output = parse_query_output(raw, expected_turns=prefix_sample.turn_id)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return output, elapsed_ms


def run_retrieval_model(
    prefix_sample: PrefixSample,
    query_output: QueryTrajectoryOutput,
    retriever: Callable[[str, int], List[Dict[str, str]]],
    top_k: int = 5,
) -> Tuple[RetrievalOutput, int]:
    """Retrieval 只针对当前 prefix 的最后一轮 query 执行。"""
    start_time = time.time()

    if not query_output.trajectory:
        raise ValueError("query_output.trajectory is empty")

    last_query = query_output.trajectory[-1]
    retrieved_cases: List[RetrievalCase] = []

    if last_query.query_mode != "none" and last_query.query_text.strip():
        dialogue_history = ""
        for message in prefix_sample.history:
            if message['role'] == "user":
                dialogue_history += f"用户: {message['text']}"
            elif message['role'] == "assistant":
                dialogue_history += f"客服: {message['text']}"
        raw_cases = retriever(last_query.query_text.strip(), dialogue_history)
        for case in raw_cases:
            retrieved_cases.append(
                RetrievalCase(
                    case_id=str(case.get("case_id", "")),
                    title=str(case.get("title", "")),
                    content=str(case.get("content", "")),
                )
            )

    elapsed_ms = int((time.time() - start_time) * 1000)
    return RetrievalOutput(retrieved_cases=retrieved_cases), elapsed_ms


def run_trigger_model(
    prefix_sample: PrefixSample,
    query_output: QueryTrajectoryOutput,
    retrieval_output: RetrievalOutput,
) -> Tuple[TriggerTrajectoryOutput, int]:
    start_time = time.time()

    dialogue_prefix = format_history_with_turn_ids(prefix_sample)
    query_trajectory = format_query_trajectory(query_output)
    retrieved_evidence = format_retrieved_cases(retrieval_output.retrieved_cases)

    trigger_output_schema = build_turn_skeleton(
        turn_count=prefix_sample.turn_id,
        fields={
            "trigger": '<true_or_false>',
            "evidence_mode": '"<no_evidence_needed|retrieve_new_evidence|reuse_existing_evidence>"',
            "reason": '""'
        }
    )
    user_prompt =f"""
下面是当前 prefix、该 prefix 对应的 query trajectory，以及当前 prefix 可用的候选 evidence。

请你只对这个 prefix 中**已经形成的固定客服应答点**做标注，输出 trigger trajectory。

【重要约束】
1. 当前 prefix 的固定客服目标轮数为：{prefix_sample.turn_id}
2. 你必须且只能输出 turn_id = 1 到 turn_id = {prefix_sample.turn_id}
3. 不能新增任何 turn_id
4. 不能遗漏任何 turn_id
5. 不能只输出最后一个 turn_id
6. 不能脑补未来用户发言、未来客服回复或未来知识链路变化
7. 你的任务是“对固定轮次做标注”，不是“推演知识链路会如何发展”

【特别提醒】
- 当 target_turn_count = 1 时，即使输入中还没有出现任何客服文本，也必须输出 turn_id = 1
- 当 target_turn_count = 2 时，必须完整输出 turn_id = 1 和 turn_id = 2，不能只输出 turn_id = 2
- 当 target_turn_count = N 时，你要重新标注从 1 到 N 的完整 trajectory，而不是只标最后一轮
- 少写任何一个 turn 或多写任何一个 turn 都是错误的

【对话历史】
{dialogue_prefix}

【query trajectory】
{query_trajectory}

【当前 prefix 可用的候选 evidence】
{retrieved_evidence}

**【请严格按照下面的 JSON 模板填写，不要改动字段名，不要新增字段，不要输出模板外内容，不要遗忘字段】**
{trigger_output_schema}
""".strip()

    messages = [
        {"role": "system", "content": TRIGGER_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_llm_json(messages)
    output = parse_trigger_output(raw, expected_turns=prefix_sample.turn_id)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return output, elapsed_ms


def run_policy_model(
    prefix_sample: PrefixSample,
    query_output: QueryTrajectoryOutput,
    retrieval_output: RetrievalOutput,
    trigger_output: TriggerTrajectoryOutput,
) -> Tuple[PolicyTrajectoryOutput, int]:
    start_time = time.time()

    dialogue_prefix = format_history_with_turn_ids(prefix_sample)
    query_trajectory = format_query_trajectory(query_output)
    trigger_trajectory = format_trigger_trajectory(trigger_output)
    retrieved_cases = format_retrieved_cases(retrieval_output.retrieved_cases)

    policy_output_schema = build_turn_skeleton(
        turn_count=prefix_sample.turn_id,
        fields={
            "label": '"<AskMissingSlot|AskClarification|ExplainedResponse|CaseRecommendation|Handoff|ProcessAcknowledgement>"',
            "reason": '""'
        }
    )

    user_prompt = f"""
下面是当前 prefix、该 prefix 对应的 query trajectory、trigger trajectory，以及当前 prefix 可用的返回案例内容。

请你只对这个 prefix 中**已经形成的固定客服应答点**做标注，输出 policy trajectory。

【重要约束】
1. 当前 prefix 的固定客服目标轮数为：{prefix_sample.turn_id}
2. 你必须且只能输出 turn_id = 1 到 turn_id = {prefix_sample.turn_id}
3. 不能新增任何 turn_id
4. 不能遗漏任何 turn_id
5. 不能只输出最后一个 turn_id
6. 不能脑补未来用户发言、未来客服回复或未来策略变化
7. 你的任务是“对固定轮次做标注”，不是“预测后续客服会怎么做”

【特别提醒】
- 当 target_turn_count = 1 时，即使输入中还没有出现任何客服文本，也必须输出 turn_id = 1
- 当 target_turn_count = 2 时，必须完整输出 turn_id = 1 和 turn_id = 2，不能只输出 turn_id = 2
- 当 target_turn_count = N 时，你要重新标注从 1 到 N 的完整 trajectory，而不是只标最后一轮
- 少写任何一个 turn 或多写任何一个 turn 都是错误的

【对话上下文】
{dialogue_prefix}

【query trajectory】
{query_trajectory}

【trigger trajectory】
{trigger_trajectory}

【返回案例】
{retrieved_cases}

**【请严格按照下面的 JSON 模板填写，不要改动字段名，不要新增字段，不要输出模板外内容，不要遗忘字段】**
{policy_output_schema}
""".strip()

    messages = [
        {"role": "system", "content": POLICY_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_llm_json(messages)
    output = parse_policy_output(raw, expected_turns=prefix_sample.turn_id)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return output, elapsed_ms


def run_response_model(
    prefix_sample: PrefixSample,
    query_output: QueryTrajectoryOutput,
    retrieval_output: RetrievalOutput,
    trigger_output: TriggerTrajectoryOutput,
    policy_output: PolicyTrajectoryOutput,
) -> Tuple[ResponseOutput, int]:
    """Response 只生成当前 prefix 最后一轮的回复。"""
    start_time = time.time()

    dialogue_prefix = format_history_with_turn_ids(prefix_sample)
    query_trajectory = format_query_trajectory(query_output)
    trigger_trajectory = format_trigger_trajectory(trigger_output)
    policy_trajectory = format_policy_trajectory(policy_output)
    retrieved_cases = format_retrieved_cases(retrieval_output.retrieved_cases)

    user_prompt = f"""
下面是当前 prefix、该 prefix 对应的 query trajectory、trigger trajectory、policy trajectory，以及当前 prefix 可用的返回案例内容。

请你只为 policy trajectory 最后一轮所对应的当前客服目标轮，生成一条自然、合理、符合策略的客服回复。

【历史对话】
{dialogue_prefix}

【query trajectory】
{query_trajectory}

【trigger trajectory】
{trigger_trajectory}

【policy trajectory】
{policy_trajectory}

【返回的案例内容】
{retrieved_cases}
""".strip()

    messages = [
        {"role": "system", "content": RESPONSE_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = call_llm_text(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return ResponseOutput(response=response), elapsed_ms


def run_single_prefix_pipeline(
    prefix_sample: PrefixSample,
    retriever: Callable[[str, int], List[Dict[str, str]]] = default_retriever,
    retrieval_top_k: int = 5,
) -> PrefixResult:
    logger.info("Processing prefix %s-%s", prefix_sample.dialog_id, prefix_sample.turn_id)
    total_start = time.time()

    # 1. Query trajectory
    query_output, query_ms = run_query_model(prefix_sample)

    # 2. Retrieval for last query only
    retrieval_output, retrieval_ms = run_retrieval_model(
        prefix_sample=prefix_sample,
        query_output=query_output,
        retriever=retriever,
        top_k=retrieval_top_k,
    )

    # 3. Trigger trajectory
    trigger_output, trigger_ms = run_trigger_model(
        prefix_sample=prefix_sample,
        query_output=query_output,
        retrieval_output=retrieval_output,
    )

    # 4. Policy trajectory
    policy_output, policy_ms = run_policy_model(
        prefix_sample=prefix_sample,
        query_output=query_output,
        retrieval_output=retrieval_output,
        trigger_output=trigger_output,
    )

    # 5. Current-turn response
    response_output, response_ms = run_response_model(
        prefix_sample=prefix_sample,
        query_output=query_output,
        retrieval_output=retrieval_output,
        trigger_output=trigger_output,
        policy_output=policy_output,
    )

    total_ms = int((time.time() - total_start) * 1000)

    return PrefixResult(
        dialog_id=prefix_sample.dialog_id,
        turn_id=prefix_sample.turn_id,
        prefix_sample=prefix_sample,
        query_output=query_output,
        retrieval_output=retrieval_output,
        trigger_output=trigger_output,
        policy_output=policy_output,
        response_output=response_output,
        latency=Latency(
            query_ms=query_ms,
            retrieval_ms=retrieval_ms,
            trigger_ms=trigger_ms,
            policy_ms=policy_ms,
            response_ms=response_ms,
            total_ms=total_ms,
        ),
    )

# ==================== 第4层：批处理与保存层 ====================

def run_dialogue(
    dialogue: Dialogue,
    retriever: Callable[[str, int], List[Dict[str, str]]] = default_retriever,
    retrieval_top_k: int = 5,
) -> List[PrefixResult]:
    logger.info("Processing dialogue %s", dialogue.dialog_id)
    prefix_samples = build_prefix_samples(dialogue)
    results = []
    for prefix_sample in prefix_samples:
        results.append(
            run_single_prefix_pipeline(
                prefix_sample=prefix_sample,
                retriever=retriever,
                retrieval_top_k=retrieval_top_k,
            )
        )
    logger.info("Dialogue %s completed with %s prefixes", dialogue.dialog_id, len(results))
    return results


def run_dataset(
    dialogues: List[Dialogue],
    output_path: str = "multi_turn_output.json",
    retriever: Callable[[str, int], List[Dict[str, str]]] = default_retriever,
    retrieval_top_k: int = 5,
) -> None:
    logger.info("Processing dataset with %s dialogues", len(dialogues))
    all_results: List[PrefixResult] = []
    total_start = time.time()

    for i, dialogue in enumerate(dialogues, 1):
        try:
            results = run_dialogue(
                dialogue=dialogue,
                retriever=retriever,
                retrieval_top_k=retrieval_top_k,
            )
            all_results.extend(results)
        except Exception as e:
            logger.exception("Dialogue failed: %s", getattr(dialogue, "dialog_id", None))
            continue  # 关键：失败后继续下一个 dialogue

    total_ms = int((time.time() - total_start) * 1000)
    logger.info(
        "Dataset processing completed in %sms, total %s prefix results",
        total_ms,
        len(all_results),
    )

    results_dict = [asdict(result) for result in all_results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s", output_path)

# ==================== 校验与辅助函数 ====================

def validate_prefix_samples(dialogue: Dialogue, prefix_samples: List[PrefixSample]) -> bool:
    """验证 prefix 构造是否正确。"""
    assistant_indices = [i for i, t in enumerate(dialogue.turns) if t.get("role") == "assistant"]

    if len(prefix_samples) != len(assistant_indices):
        logger.error(
            "Sample count mismatch: expected %s, got %s",
            len(assistant_indices),
            len(prefix_samples),
        )
        return False

    for sample, assistant_idx in zip(prefix_samples, assistant_indices):
        expected_turn_id = assistant_indices.index(assistant_idx) + 1
        if sample.turn_id != expected_turn_id:
            logger.error("Turn ID mismatch: expected %s, got %s", expected_turn_id, sample.turn_id)
            return False
        if sample.history != dialogue.turns[:assistant_idx]:
            logger.error("History mismatch at turn_id=%s", sample.turn_id)
            return False
        if sample.gold_response != dialogue.turns[assistant_idx].get("text", ""):
            logger.error("Gold response mismatch at turn_id=%s", sample.turn_id)
            return False
        if sample.target_turn_index != assistant_idx:
            logger.error("target_turn_index mismatch at turn_id=%s", sample.turn_id)
            return False

    return True

# ==================== 示例主函数 ====================

def main() -> None:
    data_path= "/home/aarc/CuhkszTeam/nas1/RUNTIME/data_processing/案例推荐筛选/data/10/带caseID对话.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = list(data.items())[:100]
        sample_dialogues = []
        for dialog_id, dialogue_content in tqdm(data):
            sample = {}
            sample["dialog_id"] = dialog_id
            reformat_turns = []
            for turn in dialogue_content["text"]:
                if "用户" in turn:
                    reformat_turns.append({"role": "user", "text": turn['用户']})
                elif "客服" in turn:
                    reformat_turns.append({"role": "assistant", "text": turn['客服']})
            sample["turns"] = reformat_turns
            sample_dialogues.append(sample)

    dialogues = [Dialogue(dialog_id=item["dialog_id"], turns=item["turns"]) for item in sample_dialogues]

    for dialogue in dialogues:
        assert validate_prefix_samples(dialogue, build_prefix_samples(dialogue)), f"Invalid prefix samples: {dialogue.dialog_id}"

    run_dataset(dialogues, output_path="scu_output.json")

if __name__ == "__main__":
    main()
