"""
多模型对话系统框架
流程：
  1. Query Model: 生成查询
  2. Retriever Model: 检索相关知识
  3. Trigger Model: 判断是否需要触发知识
  4. Policy Model: 判断对话动作
  5. Response Model: 生成回复
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
class DialogueInput:
    """对话输入"""
    current_turn: str  # 当前轮对话
    dialogue_history: List[str]  # 历史对话列表


@dataclass
class QueryResult:
    """查询结果"""
    query: str  # 生成的查询


@dataclass
class RetrieverResult:
    """检索结果"""
    evidence: List[str]  # 检索到的证据列表


@dataclass
class TriggerResult:
    """触发决策结果"""
    should_trigger: bool  # 是否需要触发知识
    confidence: float  # 置信度


@dataclass
class PolicyResult:
    """策略结果"""
    dialogue_action: str  # 对话动作


@dataclass
class ResponseResult:
    """回复结果"""
    response: str  # 生成的回复


@dataclass
class PipelineOutput:
    """完整流程输出"""
    dialogue_input: DialogueInput  # 原始对话输入
    golden_response: str  # 黄金标准回复
    query_result: QueryResult
    retriever_result: RetrieverResult
    trigger_result: TriggerResult
    policy_result: PolicyResult
    response_result: ResponseResult
    times: Dict[str, float]  # 各模块执行时间


# ==================== Prompt 定义 ====================
query_prompt = """"""
retriever_prompt = """"""
trigger_prompt = """"""
policy_prompt = """"""
response_prompt = """"""


# ==================== Model 调用函数 ====================
def call_query_model(dialogue_input: DialogueInput) -> QueryResult:
    """
    调用Query Model生成查询
    
    Args:
        dialogue_input: 对话输入
    
    Returns:
        QueryResult: 查询结果
    """
    logger.info("Step 1: Calling Query Model...")
    
    # TODO: 构造prompt并调用API
    query = ""
    
    return QueryResult(query=query)


def call_retriever_model(
    query: str,
    dialogue_history: List[str]
) -> RetrieverResult:
    """
    调用Retriever Model检索相关知识
    
    Args:
        query: 生成的查询
        dialogue_history: 对话历史
    
    Returns:
        RetrieverResult: 检索结果
    """
    logger.info("Step 2: Calling Retriever Model...")
    
    # TODO: 构造prompt并调用API
    evidence: List[str] = []
    
    return RetrieverResult(evidence=evidence)


def call_trigger_model(
    evidence: List[str],
    dialogue_history: List[str]
) -> TriggerResult:
    """
    调用Trigger Model判断是否需要触发知识
    
    Args:
        evidence: 检索到的证据
        dialogue_history: 对话历史
    
    Returns:
        TriggerResult: 触发决策结果
    """
    logger.info("Step 3: Calling Trigger Model...")
    
    # TODO: 构造prompt并调用API
    # 返回值应包含：should_trigger (bool) 和 confidence (float)
    should_trigger = False
    confidence = 0.0
    
    return TriggerResult(should_trigger=should_trigger, confidence=confidence)


def call_policy_model(
    dialogue_history: List[str],
    evidence: Optional[List[str]] = None
) -> PolicyResult:
    """
    调用Policy Model判断对话动作
    
    Args:
        dialogue_history: 对话历史
        evidence: 可选的检索证据（如果触发了知识）
    
    Returns:
        PolicyResult: 策略结果
    """
    logger.info("Step 4: Calling Policy Model...")
    
    # TODO: 根据是否有evidence构造不同的prompt并调用API
    # 情况1: 有evidence -> 把evidence和历史对话输入
    # 情况2: 无evidence -> 只把历史对话输入
    dialogue_action = ""
    
    return PolicyResult(dialogue_action=dialogue_action)


def call_response_model(
    dialogue_history: List[str],
    dialogue_action: str,
    evidence: Optional[List[str]] = None
) -> ResponseResult:
    """
    调用Response Model生成回复
    
    Args:
        dialogue_history: 对话历史
        dialogue_action: 对话动作
        evidence: 可选的检索证据
    
    Returns:
        ResponseResult: 回复结果
    """
    logger.info("Step 5: Calling Response Model...")
    
    # TODO: 构造prompt并调用API
    response = ""
    
    return ResponseResult(response=response)


# ==================== 主流程编排 ====================
def process_dialogues(dialogue_inputs: List[DialogueInput], golden_responses: List[str]) -> List[PipelineOutput]:
    """
    处理多个对话的主流程
    
    Args:
        dialogue_inputs: 对话输入列表
        golden_responses: 黄金标准回复列表
    
    Returns:
        List[PipelineOutput]: 完整的流程输出列表
    """
    logger.info(f"Processing {len(dialogue_inputs)} dialogues")
    total_start_time = time.time()
    
    results = []
    
    for i, dialogue_input in enumerate(dialogue_inputs, 1):
        logger.info(f"Processing dialogue {i}/{len(dialogue_inputs)}: {dialogue_input.current_turn}")
        start_time = time.time()
        
        try:
            times = {}
            
            # Step 1: 调用Query Model
            start_time = time.time()
            query_result = call_query_model(dialogue_input)
            times['query'] = time.time() - start_time
            logger.info(f"Query generated: {query_result.query}")
            
            # Step 2: 调用Retriever Model
            start_time = time.time()
            retriever_result = call_retriever_model(
                query_result.query,
                dialogue_input.dialogue_history
            )
            times['retriever'] = time.time() - start_time
            logger.info(f"Retrieved {len(retriever_result.evidence)} evidences")
            
            # Step 3: 调用Trigger Model
            start_time = time.time()
            trigger_result = call_trigger_model(
                retriever_result.evidence,
                dialogue_input.dialogue_history
            )
            times['trigger'] = time.time() - start_time
            logger.info(f"Trigger decision: {trigger_result.should_trigger} (confidence: {trigger_result.confidence})")
            
            # Step 4: 调用Policy Model（根据trigger结果决定是否传入evidence）
            start_time = time.time()
            if trigger_result.should_trigger:
                logger.info("Triggering knowledge is needed, passing evidence to Policy Model")
                policy_result = call_policy_model(
                    dialogue_input.dialogue_history,
                    evidence=retriever_result.evidence
                )
            else:
                logger.info("No knowledge trigger needed, calling Policy Model without evidence")
                policy_result = call_policy_model(
                    dialogue_input.dialogue_history,
                    evidence=None
                )
            times['policy'] = time.time() - start_time
            logger.info(f"Dialogue action: {policy_result.dialogue_action}")
            
            # Step 5: 调用Response Model
            start_time = time.time()
            evidence_for_response = retriever_result.evidence if trigger_result.should_trigger else None
            response_result = call_response_model(
                dialogue_input.dialogue_history,
                policy_result.dialogue_action,
                evidence=evidence_for_response
            )
            times['response'] = time.time() - start_time
            logger.info(f"Response generated: {response_result.response}")
            
            # 组装完整输出
            pipeline_output = PipelineOutput(
                dialogue_input=dialogue_input,
                golden_response=golden_responses[i-1] if i-1 < len(golden_responses) else "",
                query_result=query_result,
                retriever_result=retriever_result,
                trigger_result=trigger_result,
                policy_result=policy_result,
                response_result=response_result,
                times=times
            )
            
            results.append(pipeline_output)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Dialogue {i} completed in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in dialogue {i} processing: {e}", exc_info=True)
            # 可以选择跳过或抛出异常，这里选择跳过并记录
            continue
    
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"All {len(dialogue_inputs)} dialogues processed in {total_elapsed_time:.2f}s")
    
    # 累加所有模块的时间
    cumulative_times = {}
    for output in results:
        for module, t in output.times.items():
            cumulative_times[module] = cumulative_times.get(module, 0) + t
    
    # 保存汇总记录
    summary = {
        "total_dialogues": len(results),
        "total_time": total_elapsed_time,
        "cumulative_times": cumulative_times,
        "average_times": {module: t / len(results) for module, t in cumulative_times.items()} if results else {}
    }
    
    with open("time_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Time summary saved to time_summary.json")
    
    return results


# ==================== 输出和保存 ====================
def save_results(outputs: List[PipelineOutput], output_path: str = "output.json") -> None:
    """
    保存多个流程输出结果
    
    Args:
        outputs: 流程输出列表
        output_path: 输出文件路径
    """
    results_list = []
    for i, output in enumerate(outputs, 1):
        result_dict = {
            'dialogue_id': i,
            'dialogue_input': asdict(output.dialogue_input),
            'golden_response': output.golden_response,
            'query': asdict(output.query_result),
            'retriever': asdict(output.retriever_result),
            'trigger': asdict(output.trigger_result),
            'policy': asdict(output.policy_result),
            'response': asdict(output.response_result),
            'times': output.times,
        }
        results_list.append(result_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def print_results(outputs: List[PipelineOutput]) -> None:
    """
    打印多个流程输出结果
    
    Args:
        outputs: 流程输出列表
    """
    print("\n" + "=" * 80)
    print(f"对话流程处理结果 - 共处理 {len(outputs)} 个对话")
    print("=" * 80)
    
    for i, output in enumerate(outputs, 1):
        print(f"\n--- 对话 {i} ---")
        print(f"[查询] Query: {output.query_result.query}")
        print(f"[检索] 检索到 {len(output.retriever_result.evidence)} 条证据")
        for j, evidence in enumerate(output.retriever_result.evidence, 1):
            print(f"  {j}. {evidence}")
        print(f"[触发] 是否触发知识: {output.trigger_result.should_trigger}")
        print(f"       置信度: {output.trigger_result.confidence}")
        print(f"[策略] 对话动作: {output.policy_result.dialogue_action}")
        print(f"[回复] 生成回复: {output.response_result.response}")
        print(f"[时间] 各模块耗时: {output.times}")
        total_time = sum(output.times.values())
        print(f"       总耗时: {total_time:.2f}s")
    
    print("\n" + "=" * 80)


# ==================== 主函数 ====================
def main():
    """主函数 - 演示流程"""
    # 示例输入 - 多个对话
    dialogue_inputs = [
        DialogueInput(
            current_turn="用户最新的问题1",
            dialogue_history=["用户提问1", "助手回答1", "用户提问2", "助手回答2"]
        ),
        DialogueInput(
            current_turn="用户最新的问题2",
            dialogue_history=["用户提问A", "助手回答A", "用户提问B", "助手回答B"]
        ),
        DialogueInput(
            current_turn="用户最新的问题3",
            dialogue_history=["用户提问X", "助手回答X"]
        )
    ]
    
    # 处理多个对话
    outputs = process_dialogues(dialogue_inputs)
    
    # 输出结果
    print_results(outputs)
    save_results(outputs)


if __name__ == "__main__":
    main()

