"""
多模型对话系统框架
流程：
  1. Query Model: 生成查询
  2. Retriever Model: 检索相关知识
  3. Trigger Model: 判断是否需要触发知识
  4. Policy Model: 判断对话动作
  5. Response Model: 生成回复
"""
from json_repair import repair_json
import json
import time
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.qwen_api import call_qwen_api
from utils.scu_rag import rag

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 数据结构定义 ====================
@dataclass
class DialogueInput:
    """对话输入"""
    current_turn: str
    dialogue_history: List[str]

@dataclass
class Case:
    """案例数据结构"""
    id: str
    title: str
    content: str

@dataclass
class QueryResult:
    """查询结果"""
    query: str

@dataclass
class RetrieverResult:
    """检索结果"""
    evidence: List[Case]

@dataclass
class TriggerResult:
    """触发决策结果"""
    should_trigger: bool
    confidence: float

@dataclass
class PolicyResult:
    """策略结果"""
    dialogue_action: str
    case_id: str

@dataclass
class ResponseResult:
    """回复结果"""
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
class PipelineOutput:
    """完整流程输出"""
    dialogue_input: DialogueInput
    golden_response: str
    query_result: QueryResult
    retriever_result: RetrieverResult
    trigger_result: TriggerResult
    policy_result: PolicyResult
    response_result: ResponseResult
    latency: Latency


# ==================== Prompt 定义 ====================
with open("prompts/query.prompt", "r", encoding="utf-8") as f:
    query_prompt = f.read()

with open("prompts/trigger.prompt", "r", encoding="utf-8") as f:
    trigger_prompt = f.read()

with open("prompts/policy.prompt", "r", encoding="utf-8") as f:
    policy_prompt = f.read()

with open("prompts/response.prompt", "r", encoding="utf-8") as f:
    response_prompt = f.read()


# ==================== 工具函数 ====================
def cases_to_string(cases: Optional[List[Case]]) -> str:
    """把案例列表格式化成 prompt 需要的字符串"""
    if not cases:
        return "无"

    parts = []
    for case in cases:
        parts.append(
            f"case_id: {case.id}\n"
            f"case_title: {case.title}\n"
            f"case_content: {case.content}\n"
        )
    return "\n".join(parts).strip()


def safe_json_loads(text: str, default: Optional[dict] = None) -> dict:
    """安全解析模型 JSON 输出"""
    if default is None:
        default = {}
    try:
        return json.loads(repair_json(text))
    except Exception:
        return default


def load_retrieval_cases(retrieval_path: str) -> List[List[Case]]:
    """预加载 topk 检索结果"""
    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cases = []
    for item in data:
        cur_cases = []
        retrieval_result = item.get("retrieval_result", {})
        for _, case in retrieval_result.items():
            cur_cases.append(
                Case(
                    id=str(case.get("case_id", "")),
                    title=str(case.get("case_title", "")),
                    content=str(case.get("content", ""))
                )
            )
        all_cases.append(cur_cases)
    return all_cases


# ==================== Model 调用函数 ====================
def call_query_model(dialogue_input: DialogueInput) -> Tuple[QueryResult, int]:
    """
    调用Query Model生成查询
    """
    logger.info("Step 1: Calling Query Model...")
    start_time = time.time()

    messages = [{
        "role": "user",
        "content": query_prompt.replace("{p_dialogue}", str(dialogue_input.dialogue_history))
    }]
    query = call_qwen_api(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)

    return QueryResult(query=query), elapsed_ms


def call_retriever_model(
    query: str,
    dialogue_history: List[str]
) -> Tuple[RetrieverResult, int]:
    """
    调用Retriever Model检索相关知识
    如果你之后想切回实时 RAG，可以启用这个函数。
    """
    logger.info("Step 2: Calling Retriever Model...")
    start_time = time.time()

    evidence = rag(query, "\n".join(dialogue_history), "true")
    cases = []
    for case in evidence:
        cases.append(
            Case(
                id=str(case.get("case_id", "")),
                title=str(case.get("title", "")),
                content=str(case.get("content", ""))
            )
        )

    elapsed_ms = int((time.time() - start_time) * 1000)
    return RetrieverResult(evidence=cases), elapsed_ms


def call_trigger_model(
    evidence: List[Case],
    dialogue_history: List[str]
) -> Tuple[TriggerResult, int]:
    """
    调用Trigger Model判断是否需要触发知识
    """
    start_time = time.time()
    logger.info("Step 3: Calling Trigger Model...")

    evidence_str = cases_to_string(evidence)
    messages = [{
        "role": "user",
        "content": trigger_prompt
            .replace("{messages}", str(dialogue_history))
            .replace("{retrieved_cases}", evidence_str)
    }]

    resp = safe_json_loads(call_qwen_api(messages), default={"knowledge_support": "检索+常识"})
    knowledge_support = resp.get("knowledge_support", "检索+常识")

    should_trigger = knowledge_support in ["检索+常识", "检索 + 常识"]
    confidence = 0.0
    elapsed_ms = int((time.time() - start_time) * 1000)

    return TriggerResult(should_trigger=should_trigger, confidence=confidence), elapsed_ms


def call_policy_model(
    dialogue_history: List[str],
    evidence: Optional[List[Case]] = None
) -> Tuple[PolicyResult, int]:
    """
    调用Policy Model判断对话动作
    """
    start_time = time.time()
    logger.info("Step 4: Calling Policy Model...")

    evidence_str = cases_to_string(evidence)

    messages = [{
        "role": "user",
        "content": policy_prompt
            .replace("{dialog_history}", str(dialogue_history))
            .replace("{retrieved_cases}", evidence_str)
    }]

    resp = safe_json_loads(call_qwen_api(messages), default={"label": "", "case_id": ""})
    dialogue_action = resp.get("label", "")
    case_id = resp.get("case_id", "")

    elapsed_ms = int((time.time() - start_time) * 1000)
    return PolicyResult(dialogue_action=dialogue_action, case_id=case_id), elapsed_ms


def call_response_model(
    dialogue_history: List[str],
    dialogue_action: str,
    evidence: Optional[List[Case]] = None
) -> Tuple[ResponseResult, int]:
    """
    调用Response Model生成回复
    """
    start_time = time.time()
    logger.info("Step 5: Calling Response Model...")

    # response 阶段默认只传 1 个案例更稳
    if evidence and len(evidence) > 0:
        evidence_for_prompt = [evidence[0]]
    else:
        evidence_for_prompt = None

    evidence_str = cases_to_string(evidence_for_prompt)

    messages = [{
        "role": "user",
        "content": response_prompt
            .replace("{dialog_history}", str(dialogue_history))
            .replace("{retrieved_cases}", evidence_str)
            .replace("{dialogue_act}", dialogue_action)
    }]
    response = call_qwen_api(messages)

    elapsed_ms = int((time.time() - start_time) * 1000)
    return ResponseResult(response=response), elapsed_ms


# ==================== 单条对话处理（多进程 worker） ====================
def process_single_dialogue(args) -> Tuple[int, Optional[PipelineOutput]]:
    """
    多进程 worker：
    输入 (idx, dialogue_input, golden_response, preloaded_cases)
    返回 (idx, pipeline_output)
    """
    idx, dialogue_input, golden_response, preloaded_cases = args
    logger.info(f"Processing dialogue {idx + 1}: {dialogue_input.current_turn}")
    start_time = time.time()

    try:
        # Step 1: Query
        query_result, query_ms = call_query_model(dialogue_input)
        logger.info(f"[{idx + 1}] Query generated: {query_result.query}")

        # Step 2: Retriever
        # 这里使用预加载好的检索结果，而不是在线 rag
        retriever_result = RetrieverResult(evidence=preloaded_cases)
        retrieval_ms = 0
        logger.info(f"[{idx + 1}] Retrieved {len(preloaded_cases)} evidences")

        # Step 3: Trigger
        trigger_result, trigger_ms = call_trigger_model(
            retriever_result.evidence,
            dialogue_input.dialogue_history
        )
        logger.info(f"[{idx + 1}] Trigger: {trigger_result.should_trigger}")

        # Step 4: Policy
        if trigger_result.should_trigger:
            policy_result, policy_ms = call_policy_model(
                dialogue_input.dialogue_history,
                evidence=retriever_result.evidence
            )
        else:
            policy_result, policy_ms = call_policy_model(
                dialogue_input.dialogue_history,
                evidence=None
            )
        logger.info(f"[{idx + 1}] Policy action: {policy_result.dialogue_action}")

        # Step 5: Response
        if trigger_result.should_trigger:
            evidence_for_response = retriever_result.evidence
            if policy_result.case_id:
                matched_cases = [c for c in retriever_result.evidence if c.id == policy_result.case_id]
                if matched_cases:
                    evidence_for_response = matched_cases
        else:
            evidence_for_response = None

        response_result, response_ms = call_response_model(
            dialogue_input.dialogue_history,
            policy_result.dialogue_action,
            evidence=evidence_for_response
        )
        logger.info(f"[{idx + 1}] Response generated")

        total_ms = int((time.time() - start_time) * 1000)

        pipeline_output = PipelineOutput(
            dialogue_input=dialogue_input,
            golden_response=golden_response,
            query_result=query_result,
            retriever_result=retriever_result,
            trigger_result=trigger_result,
            policy_result=policy_result,
            response_result=response_result,
            latency=Latency(
                query_ms=query_ms,
                retrieval_ms=retrieval_ms,
                trigger_ms=trigger_ms,
                policy_ms=policy_ms,
                response_ms=response_ms,
                total_ms=total_ms,
            ),
        )
        return idx, pipeline_output

    except Exception as e:
        logger.error(f"[{idx + 1}] Error in dialogue processing: {e}", exc_info=True)
        return idx, None


# ==================== 主流程编排（多进程） ====================
def process_dialogue(
    dialogue_inputs: List[DialogueInput],
    golden_responses: List[str],
    retrieval_cases: List[List[Case]],
    max_workers: int = 4
) -> List[PipelineOutput]:
    """
    多进程处理对话主流程
    """
    logger.info(f"Processing {len(dialogue_inputs)} dialogues with {max_workers} processes")
    total_start_time = time.time()

    tasks = [
        (i, dialogue_inputs[i], golden_responses[i], retrieval_cases[i])
        for i in range(len(dialogue_inputs))
    ]

    results_map: Dict[int, PipelineOutput] = {}

    # 用 spawn 更稳，尤其是涉及外部 SDK / 网络请求时
    mp_context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        future_to_idx = {
            executor.submit(process_single_dialogue, task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result_idx, output = future.result()
                if output is not None:
                    results_map[result_idx] = output
                    logger.info(f"Dialogue {result_idx + 1} finished")
                else:
                    logger.warning(f"Dialogue {result_idx + 1} returned None")
            except Exception as e:
                logger.error(f"Future failed at dialogue {idx + 1}: {e}", exc_info=True)

    # 按原始顺序还原
    results = [results_map[i] for i in sorted(results_map.keys())]

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"All {len(results)} dialogues processed in {total_elapsed_time:.2f}s")
    return results


# ==================== 输出和保存 ====================
def save_result(outputs: List[PipelineOutput], output_path: str = "results/phase1_test3.json") -> None:
    """
    保存流程输出结果
    """
    result_list = []
    for i, output in enumerate(outputs, 1):
        result_dict = {
            "dialogue_id": i,
            "dialogue_input": asdict(output.dialogue_input),
            # "golden_response": output.golden_response,
            "query": asdict(output.query_result),
            "retriever": {
                "evidence": [asdict(case) for case in output.retriever_result.evidence]
            },
            "trigger": asdict(output.trigger_result),
            "policy": asdict(output.policy_result),
            "response": asdict(output.response_result),
            "latency": asdict(output.latency)
        }
        result_list.append(result_dict)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    logger.info(f"Result saved to {output_path}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    data_path = "/home/aarc/CuhkszTeam/nas1/RUNTIME/Eval/Phase1_data/new_test2.json"
    retrieval_path = "/home/aarc/CuhkszTeam/nas1/RUNTIME/Eval/Phase1_data/topk_case_save.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogue_inputs = []
    golden_responses = []

    for item in data:
        dialogue_history = []
        for turn in item["text"]:
            if turn["role"] == "用户":
                dialogue_history.append(f"用户: {turn['content']}")
            else:
                dialogue_history.append(f"客服: {turn['content']}")

        dialogue_inputs.append(
            DialogueInput(
                current_turn="current_turn",
                dialogue_history=dialogue_history.copy()
            )
        )
        golden_responses.append("current_turn")

    retrieval_cases = load_retrieval_cases(retrieval_path)

    assert len(dialogue_inputs) == len(golden_responses) == len(retrieval_cases), \
        "dialogue_inputs / golden_responses / retrieval_cases 长度不一致"

    output = process_dialogue(
        dialogue_inputs=dialogue_inputs,
        golden_responses=golden_responses,
        retrieval_cases=retrieval_cases,
        max_workers=4
    )

    save_result(output, output_path="results/phase1_test3.json")


if __name__ == "__main__":
    main()
