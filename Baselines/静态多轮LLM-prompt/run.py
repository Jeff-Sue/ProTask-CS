"""
静态多轮LLM-prompt框架
核心思路：将完整对话展开成多个prefix样本，每个prefix独立运行静态单轮链路
"""

import json
import time
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.multi_api import parallel_inference

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================

@dataclass
class Dialogue:
    """完整对话"""
    dialog_id: str
    turns: List[Dict[str, str]]  # [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}]


@dataclass
class PrefixSample:
    """Prefix样本"""
    dialog_id: str
    turn_id: int
    history: List[Dict[str, str]]
    target_role: str
    gold_response: str


@dataclass
class QueryOutput:
    """Query模型输出"""
    query_mode: str  # "none" | "reuse" | "refine" | "rewrite"
    query_text: str


@dataclass
class RetrievalOutput:
    """Retrieval输出"""
    retrieved_cases: List[Dict[str, str]]  # [{"case_id": "...", "title": "...", "content": "..."}]


@dataclass
class TriggerOutput:
    """Trigger输出"""
    trigger: bool
    evidence_mode: str  # "no_evidence_needed" | "reuse_existing_evidence" | "retrieve_new_evidence"


@dataclass
class PolicyOutput:
    """Policy输出"""
    action: str  # "ask_clarification" | "direct_reply" | "recommend_case" | "transfer" | "close"


@dataclass
class ResponseOutput:
    """Response输出"""
    response: str


@dataclass
class Latency:
    """时长统计"""
    query_ms: int
    retrieval_ms: int
    trigger_ms: int
    policy_ms: int
    response_ms: int
    total_ms: int


@dataclass
class PrefixResult:
    """单个Prefix的完整结果"""
    dialog_id: str
    turn_id: int
    prefix_sample: PrefixSample

    query_output: QueryOutput
    retrieval_output: RetrievalOutput
    trigger_output: TriggerOutput
    policy_output: PolicyOutput
    response_output: ResponseOutput

    latency: Latency


# ==================== Prompt 定义 ====================
# TODO: 根据实际需求填充prompt
with open("prompts/query.prompt", 'r', encoding='utf-8') as f:
    query_prompt = f.read()

with open("prompts/retrieval.prompt", 'r', encoding='utf-8') as f:
    retrieval_prompt = f.read()

with open("prompts/trigger.prompt", 'r', encoding='utf-8') as f:
    trigger_prompt = f.read()

with open("prompts/policy.prompt", 'r', encoding='utf-8') as f:
    policy_prompt = f.read()

with open("prompts/response.prompt", 'r', encoding='utf-8') as f:
    response_prompt = f.read()


# ==================== 第1层：数据层 ====================

def load_dialogues(data_path: str) -> List[Dialogue]:
    """
    加载对话数据

    Args:
        data_path: 数据文件路径

    Returns:
        List[Dialogue]: 对话列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dialogues = []
    for item in data:
        dialogue = Dialogue(
            dialog_id=item['dialog_id'],
            turns=item['turns']
        )
        dialogues.append(dialogue)

    logger.info(f"Loaded {len(dialogues)} dialogues")
    return dialogues


# ==================== 第2层：Prefix构造层 ====================

def build_prefix_samples(dialogue: Dialogue) -> List[PrefixSample]:
    """
    将完整对话拆成多个prefix样本

    Args:
        dialogue: 完整对话

    Returns:
        List[PrefixSample]: Prefix样本列表
    """
    prefix_samples = []

    # 遍历所有assistant轮（客服回复点）
    for i, turn in enumerate(dialogue.turns):
        if turn['role'] == 'assistant':
            # 当前prefix包含到这个assistant轮为止的历史
            history = dialogue.turns[:i]  # 不包含当前assistant回复
            gold_response = turn['text']

            prefix_sample = PrefixSample(
                dialog_id=dialogue.dialog_id,
                turn_id=len(prefix_samples) + 1,  # turn_id从1开始
                history=history,
                target_role='assistant',
                gold_response=gold_response
            )
            prefix_samples.append(prefix_sample)

    return prefix_samples


# ==================== 第3层：单个Prefix的Pipeline层 ====================

def run_query_model(prefix_sample: PrefixSample) -> tuple[QueryOutput, int]:
    """
    运行Query模型

    Args:
        prefix_sample: Prefix样本

    Returns:
        tuple[QueryOutput, int]: (输出结果, 耗时ms)
    """
    start_time = time.time()

    query_user_prompt_template = """
    下面是当前客服回复之前的完整对话历史（prefix）。

    请你基于这段 prefix，判断当前轮理论上应采用的 query_mode，并给出当前轮应使用的 query_text。

    【对话历史】
    {dialogue_prefix}
    """
    messages = [
            {"role": "system", "content": query_prompt},
            {"role": "user", "content": query_user_prompt_template.format(dialogue_prefix=prefix_sample)}
        ]
    elapsed_ms = int((time.time() - start_time) * 1000)

    output = QueryOutput(query_mode=query_mode, query_text=query_text)
    return output, elapsed_ms


def run_retrieval_model(prefix_sample: PrefixSample, query_output: QueryOutput) -> tuple[RetrievalOutput, int]:
    """
    运行Retrieval模型

    Args:
        prefix_sample: Prefix样本
        query_output: Query输出

    Returns:
        tuple[RetrievalOutput, int]: (输出结果, 耗时ms)
    """
    start_time = time.time()

    trigger_user_prompt_template = """
    下面是当前客服回复之前的完整对话历史（prefix）、当前轮 query 结果，以及当前轮候选 evidence。

    请你判断当前轮是否应该进入知识链路，并输出 trigger 与 evidence_mode。

    【对话历史】
    {dialogue_prefix}

    【当前轮 query 结果】
    {query_output}

    【当前轮候选 evidence】
    {retrieved_evidence}
    """

    retrieved_cases = []
    if query_output.query_mode != "none" and query_output.query_text:
        # TODO: 基于query_text进行检索
        # 这里应该调用检索逻辑
        pass

    elapsed_ms = int((time.time() - start_time) * 1000)

    output = RetrievalOutput(retrieved_cases=retrieved_cases)
    return output, elapsed_ms


def run_trigger_model(
    prefix_sample: PrefixSample,
    query_output: QueryOutput,
    retrieval_output: RetrievalOutput
) -> tuple[TriggerOutput, int]:
    """
    运行Trigger模型

    Args:
        prefix_sample: Prefix样本
        query_output: Query输出
        retrieval_output: Retrieval输出

    Returns:
        tuple[TriggerOutput, int]: (输出结果, 耗时ms)
    """
    start_time = time.time()

    # TODO: 构造prompt并调用API
    trigger = False  # 示例
    evidence_mode = "no_evidence_needed"  # 示例

    elapsed_ms = int((time.time() - start_time) * 1000)

    output = TriggerOutput(trigger=trigger, evidence_mode=evidence_mode)
    return output, elapsed_ms


def run_policy_model(
    prefix_sample: PrefixSample,
    query_output: QueryOutput,
    retrieval_output: RetrievalOutput,
    trigger_output: TriggerOutput
) -> tuple[PolicyOutput, int]:
    """
    运行Policy模型

    Args:
        prefix_sample: Prefix样本
        query_output: Query输出
        retrieval_output: Retrieval输出
        trigger_output: Trigger输出

    Returns:
        tuple[PolicyOutput, int]: (输出结果, 耗时ms)
    """
    start_time = time.time()

    # TODO: 构造prompt并调用API
    action = "direct_reply"  # 示例

    elapsed_ms = int((time.time() - start_time) * 1000)

    output = PolicyOutput(action=action)
    return output, elapsed_ms


def run_response_model(
    prefix_sample: PrefixSample,
    retrieval_output: RetrievalOutput,
    policy_output: PolicyOutput
) -> tuple[ResponseOutput, int]:
    """
    运行Response模型

    Args:
        prefix_sample: Prefix样本
        retrieval_output: Retrieval输出
        policy_output: Policy输出

    Returns:
        tuple[ResponseOutput, int]: (输出结果, 耗时ms)
    """
    start_time = time.time()

    # TODO: 构造prompt并调用API
    response = ""  # 示例

    elapsed_ms = int((time.time() - start_time) * 1000)

    output = ResponseOutput(response=response)
    return output, elapsed_ms


def run_single_prefix_pipeline(prefix_sample: PrefixSample) -> PrefixResult:
    """
    运行单个Prefix的完整pipeline

    Args:
        prefix_sample: Prefix样本

    Returns:
        PrefixResult: 完整结果
    """
    logger.info(f"Processing prefix {prefix_sample.dialog_id}-{prefix_sample.turn_id}")

    total_start = time.time()

    # 1. Query
    query_output, query_ms = run_query_model(prefix_sample)

    # 2. Retrieval
    retrieval_output, retrieval_ms = run_retrieval_model(prefix_sample, query_output)

    # 3. Trigger
    trigger_output, trigger_ms = run_trigger_model(prefix_sample, query_output, retrieval_output)

    # 4. Policy
    policy_output, policy_ms = run_policy_model(prefix_sample, query_output, retrieval_output, trigger_output)

    # 5. Response
    response_output, response_ms = run_response_model(prefix_sample, retrieval_output, policy_output)

    total_ms = int((time.time() - total_start) * 1000)

    latency = Latency(
        query_ms=query_ms,
        retrieval_ms=retrieval_ms,
        trigger_ms=trigger_ms,
        policy_ms=policy_ms,
        response_ms=response_ms,
        total_ms=total_ms
    )

    result = PrefixResult(
        dialog_id=prefix_sample.dialog_id,
        turn_id=prefix_sample.turn_id,
        prefix_sample=prefix_sample,
        query_output=query_output,
        retrieval_output=retrieval_output,
        trigger_output=trigger_output,
        policy_output=policy_output,
        response_output=response_output,
        latency=latency
    )

    logger.info(f"Prefix {prefix_sample.dialog_id}-{prefix_sample.turn_id} completed in {total_ms}ms")
    return result


# ==================== 第4层：批处理与保存层 ====================

def run_dialogue(dialogue: Dialogue) -> List[PrefixResult]:
    """
    处理单条完整对话

    Args:
        dialogue: 完整对话

    Returns:
        List[PrefixResult]: 该对话所有prefix的结果
    """
    logger.info(f"Processing dialogue {dialogue.dialog_id}")

    # 1. 构造prefix样本
    prefix_samples = build_prefix_samples(dialogue)

    # 2. 对每个prefix运行pipeline
    results = []
    for prefix_sample in prefix_samples:
        result = run_single_prefix_pipeline(prefix_sample)
        results.append(result)

    logger.info(f"Dialogue {dialogue.dialog_id} completed with {len(results)} prefixes")
    return results


def run_dataset(dialogues: List[Dialogue], output_path: str = "multi_turn_output.json") -> None:
    """
    处理整个数据集

    Args:
        dialogues: 对话列表
        output_path: 输出文件路径
    """
    logger.info(f"Processing dataset with {len(dialogues)} dialogues")

    all_results = []
    total_start = time.time()

    for dialogue in dialogues:
        results = run_dialogue(dialogue)
        all_results.extend(results)

    total_ms = int((time.time() - total_start) * 1000)
    logger.info(f"Dataset processing completed in {total_ms}ms, total {len(all_results)} prefix results")

    # 保存结果
    results_dict = [asdict(result) for result in all_results]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")


# ==================== 工具函数 ====================

def validate_prefix_samples(dialogue: Dialogue, prefix_samples: List[PrefixSample]) -> bool:
    """
    验证prefix样本构造是否正确

    Args:
        dialogue: 原始对话
        prefix_samples: 构造的prefix样本

    Returns:
        bool: 是否正确
    """
    # 检查样本数量
    assistant_count = sum(1 for turn in dialogue.turns if turn['role'] == 'assistant')
    if len(prefix_samples) != assistant_count:
        logger.error(f"Sample count mismatch: expected {assistant_count}, got {len(prefix_samples)}")
        return False

    # 检查每个样本
    for i, sample in enumerate(prefix_samples):
        if sample.turn_id != i + 1:
            logger.error(f"Turn ID mismatch at sample {i}: expected {i+1}, got {sample.turn_id}")
            return False

        # 检查history长度
        expected_history_len = sum(1 for turn in dialogue.turns[:dialogue.turns.index(next(t for t in dialogue.turns if t['role'] == 'assistant' and dialogue.turns.index(t) == len(sample.history)))] if turn['role'] == 'assistant')
        if len(sample.history) != expected_history_len:
            logger.error(f"History length mismatch at sample {i}")
            return False

    return True


# ==================== 主函数 ====================

def main():
    """主函数 - 演示流程"""

    # 示例对话数据
    sample_dialogues = [
        {
            "dialog_id": "dialog_001",
            "turns": [
                {"role": "user", "text": "我的电脑开机很慢"},
                {"role": "assistant", "text": "您好，请问您的电脑型号是什么？"},
                {"role": "user", "text": "是联想ThinkPad T14"},
                {"role": "assistant", "text": "我建议您清理一下系统垃圾文件，或者检查硬盘空间。"},
                {"role": "user", "text": "好的，我试试"},
                {"role": "assistant", "text": "如果还有问题，请随时联系我们。"}
            ]
        },
        {
            "dialog_id": "dialog_002",
            "turns": [
                {"role": "user", "text": "我想升级我的软件"},
                {"role": "assistant", "text": "请问您想升级哪个软件？"}
            ]
        }
    ]

    # 转换为Dialogue对象
    dialogues = []
    for item in sample_dialogues:
        dialogue = Dialogue(
            dialog_id=item['dialog_id'],
            turns=item['turns']
        )
        dialogues.append(dialogue)

    # 处理数据集
    run_dataset(dialogues, "multi_turn_output.json")


if __name__ == "__main__":
    main()