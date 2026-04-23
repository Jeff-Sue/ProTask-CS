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
from typing import Dict, List, Optional, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.qwen_api import call_qwen_api
from utils.scu_rag import rag

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
class Case:
    """案例数据结构"""
    id: str
    title: str
    content: str

@dataclass
class QueryResult:
    """查询结果"""
    query: str  # 生成的查询

@dataclass
class RetrieverResult:
    """检索结果"""
    evidence: List[Case]  # 检索到的证据列表


@dataclass
class TriggerResult:
    """触发决策结果"""
    should_trigger: bool  # 是否需要触发知识
    confidence: float  # 置信度

@dataclass
class PolicyResult:
    """策略结果"""
    dialogue_action: str  # 对话动作
    case_id: str # 调用的案例ID

@dataclass
class ResponseResult:
    """回复结果"""
    response: str  # 生成的回复

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
    start_time = time.time()
    # TODO: 构造prompt并调用API
    messages = [{"role": "user", "content": query_prompt.replace("{p_dialogue}", str(dialogue_input))}]
    query = call_qwen_api(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return QueryResult(query=query), elapsed_ms

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
    start_time = time.time()
    evidence = rag(query, "\n".join(dialogue_history), "true")
    evidence_str = ""
    for case in evidence:
        evidence_str += f"case_id: {str(case.get('case_id', ''))}\n"
        evidence_str += f"title: {str(case.get('title', ''))}\n"
        evidence_str += f"content: {str(case.get('content', ''))}\n\n"
    elapsed_ms = int((time.time() - start_time) * 1000)

    return RetrieverResult(evidence=evidence_str), elapsed_ms

def call_trigger_model(
    evidence: List[Case],
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
    start_time = time.time()
    logger.info("Step 3: Calling Trigger Model...")
    
    evidence_str = ""
    for case in evidence:
        evidence_str += f"case_id: {case.id}\ncase_title: {case.title}\ncase_content: {case.content}\n\n"

    messages = [{"role": "user", "content": trigger_prompt.replace("{messages}", str(dialogue_history)).replace("{retrieved_cases}", evidence_str)}]
    try:
        policy_mode = json.loads(repair_json(call_qwen_api(messages)))["knowledge_support"]
    except:
        logger.info("policy_mode生成错误")
        policy_mode = "检索+常识"
    if policy_mode in ["检索+常识", "检索 + 常识"]:
        should_trigger = True
    else:
        should_trigger = False
    confidence = 0.0
    elapsed_ms = int((time.time() - start_time) * 1000)

    return TriggerResult(should_trigger=should_trigger, confidence=confidence), elapsed_ms

def call_policy_model(
    dialogue_history: List[str],
    evidence: Optional[List[Case]] = None 
) -> PolicyResult:
    """
    调用Policy Model判断对话动作
   
    Args:
        dialogue_history: 对话历史
        evidence: 可选的检索证据（如果触发了知识）
   
    Returns:
        PolicyResult: 策略结果
    """
    start_time = time.time()
    logger.info("Step 4: Calling Policy Model...")
   
    # TODO: 根据是否有evidence构造不同的prompt并调用API
    # 情况1: 有evidence -> 把evidence和历史对话输入
    # 情况2: 无evidence -> 只把历史对话输入
    messages = [{"role": "user", "content": policy_prompt.replace("{dialog_history}", str(dialogue_history)).replace("{retrieved_cases}", str(evidence))}]
    resp = json.loads(repair_json(call_qwen_api(messages)))
    dialogue_action = resp.get("label", "")
    case_id = resp.get("case_id", "")
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return PolicyResult(dialogue_action=dialogue_action, case_id=case_id), elapsed_ms

def call_response_model(
    dialogue_history: List[str],
    dialogue_action: str,
    evidence: Optional[List[Case]] = None 
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
    start_time = time.time()
    logger.info("Step 5: Calling Response Model...")
    print("evidence: ", evidence)
    if evidence == "无" or not evidence:
        evidence = "无"
    else:
        evidence = f"case_id: {evidence[0].id}\ncase_title: {evidence[0].title}\ncase_content: {evidence[0].content}\n\n"
    print("evidence:", evidence)
    # TODO: 构造prompt并调用API
    messages = [{"role": "user", "content": response_prompt.replace("{dialog_history}", str(dialogue_history)).replace("{retrieved_cases}", evidence).replace("{dialogue_act}", dialogue_action)}]
    response = call_qwen_api(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return ResponseResult(response=response), elapsed_ms


# ==================== 主流程编排 ====================
def process_dialogue(dialogue_inputs: List[DialogueInput], golden_responses: List[str]) -> List[PipelineOutput]:
    """
    处理对话的主流程
   
    Args:
        dialogue_input: 对话输入
   
    Returns:
        PipelineOutput: 完整的流程输出
    """
    logger.info(f"Processing {len(dialogue_inputs)} dialogues")
    total_start_time = time.time()
    results = []
    for i, dialogue_input in enumerate(dialogue_inputs, 1):
        logger.info(f"Processing dialogue {i}/{len(dialogue_inputs)}: {dialogue_input.current_turn}")
        start_time = time.time()
        try:
            # Step 1: 调用Query Model
            query_result, query_ms = call_query_model(dialogue_input.dialogue_history)
            logger.info(f"Query generated: {query_result.query}")
        
            # Step 2: 调用Retriever Model
            # retriever_result, retrieval_ms = call_retriever_model(
            #     query_result.query,
            #     dialogue_input.dialogue_history
            # )
            # logger.info(f"Retrieved top-10 evidences")
            logger.info("Step 2: Calling Retriever Model...")
            with open("/home/aarc/CuhkszTeam/nas1/RUNTIME/Eval/Phase1_data/topk_case_save.json", "r", encoding="utf-8") as f_r:
                data = json.load(f_r)
                cases = data[i-1]
                new_cases = []
                for case_rank, case in cases["retrieval_result"].items():
                    new_cases.append(Case(id=case["case_id"], title=case["case_title"], content=case["content"]))
            retriever_result = RetrieverResult(evidence=new_cases)
            retrieval_ms = 0
            logger.info(f"Retrieved top-10 evidences")
        
            # Step 3: 调用Trigger Model
            trigger_result, trigger_ms = call_trigger_model(
                retriever_result.evidence,
                dialogue_input.dialogue_history
            )
            logger.info(f"Trigger decision: {trigger_result.should_trigger} (confidence: {trigger_result.confidence})")
        
            # Step 4: 调用Policy Model（根据trigger结果决定是否传入evidence）
            if trigger_result.should_trigger:
                logger.info("Triggering knowledge is needed, passing evidence to Policy Model")
                policy_result, policy_ms = call_policy_model(
                    dialogue_input.dialogue_history,
                    evidence=retriever_result.evidence
                )
            else:
                logger.info("No knowledge trigger needed, calling Policy Model without evidence")
                policy_result, policy_ms = call_policy_model(
                    dialogue_input.dialogue_history,
                    evidence="无"
                )
            logger.info(f"Dialogue action: {policy_result.dialogue_action}")
        
            # Step 5: 调用Response Model
            if trigger_result.should_trigger:
                evidence_for_response = retriever_result.evidence
            else:
                evidence_for_response = "无"
            
            if trigger_result.should_trigger and policy_result.case_id:
                evidence_for_response = [case for case in retriever_result.evidence if case.id == policy_result.case_id]
            
            response_result, response_ms = call_response_model(
                dialogue_input.dialogue_history,
                policy_result.dialogue_action,
                evidence=evidence_for_response
            )
            logger.info(f"Response generated: {response_result.response}")

            total_ms = int((time.time() - start_time) * 1000)        
            # 组装完整输出
            pipeline_output = PipelineOutput(
                dialogue_input=dialogue_input,
                golden_response=golden_responses[i-1],
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
            results.append(pipeline_output)
            elapsed_time = time.time() - start_time
            logger.info(f"Dialogue {i} completed in {elapsed_time:.2f}s")  
        
        except Exception as e:
            logger.error(f"Error in dialogue processing: {e}", exc_info=True)
            continue
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"All {len(dialogue_inputs)} dialogues processed in {total_elapsed_time:.2f}s")

    return results

# ==================== 输出和保存 ====================
def save_result(outputs: List[PipelineOutput], output_path: str = "results/phase1_test3.json") -> None:
    """
    保存流程输出结果
   
    Args:
        output: 流程输出
        output_path: 输出文件路径
    """
    result_list = []
    for i, output in enumerate(outputs, 1):
        result_dict = {
            "dialogue_id": i,
            'dialogue_input': asdict(output.dialogue_input),
            # 'golden_response': output.golden_response,
            'query': asdict(output.query_result),
            'retriever': asdict(output.retriever_result),
            'trigger': asdict(output.trigger_result),
            'policy': asdict(output.policy_result),
            'response': asdict(output.response_result),
            'lantency': asdict(output.latency)
        }
        result_list.append(result_dict)
   
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
   
    logger.info(f"Result saved to {output_path}")

# ==================== 主函数 ====================
def main():
    """主函数 - 演示流程"""
    # 示例输入
    data_path= "/home/aarc/CuhkszTeam/nas1/RUNTIME/Eval/Phase1_data/new_test2.json"
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
            dialogue_inputs.append(DialogueInput(
                current_turn="current_turn",
                dialogue_history=dialogue_history.copy()
            ))
            golden_responses.append("current_turn")

    # 处理对话
    output = process_dialogue(dialogue_inputs, golden_responses)
   
    # 输出结果
    # print_result(output)
    save_result(output)

if __name__ == "__main__":
    main()
