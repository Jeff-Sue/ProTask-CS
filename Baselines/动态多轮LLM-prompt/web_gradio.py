import html
import json
import uuid
from typing import Any, Dict, List

import gradio as gr

from dynamic_multiturn_runtime_v3 import DynamicMultiTurnRuntime


# ==================== Runtime ====================

def new_runtime() -> DynamicMultiTurnRuntime:
    session_id = f"gradio_{uuid.uuid4().hex[:8]}"
    return DynamicMultiTurnRuntime(
        dialogue_id=session_id,
        retrieval_top_k=5,
        save_history_path=f"saves/gradio_runtime_{session_id}.json",
        verbose=False,
    )


# ==================== Data Utils ====================

def dataclass_to_dict(obj: Any) -> Any:
    if isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def short_case_list(cases: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, str]]:
    output = []
    for c in cases[:max_items]:
        output.append({
            "case_id": str(c.get("case_id", "")),
            "title": str(c.get("title", "")),
        })
    return output


def build_state_sections(result) -> Dict[str, Any]:
    evidence_state = result.state_snapshot.get("evidence_state", {})

    evidence_used = dataclass_to_dict(result.retrieval_output)
    active_evidence = dataclass_to_dict(evidence_state.get("active_evidence", []))
    last_retrieved = dataclass_to_dict(evidence_state.get("last_retrieved_cases", []))

    return {
        "turn_id": result.turn_id,
        "query": dataclass_to_dict(result.query_state),
        "trigger": dataclass_to_dict(result.trigger_state),
        "policy": dataclass_to_dict(result.policy_state),
        "evidence": {
            "evidence_used_this_turn": evidence_used,
            "active_evidence": active_evidence,
            "last_retrieved_cases": last_retrieved,
            "evidence_source_query": evidence_state.get("evidence_source_query", ""),
        },
        "latency": dataclass_to_dict(result.latency),
    }


# ==================== Render ====================

def render_chat(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return """
        <div class="empty-chat">
            <div class="empty-chat-icon">✦</div>
            <div class="empty-chat-title">欢迎使用华为 IT 动态多轮客服助手</div>
            <div class="empty-chat-subtitle">
                请输入一个 IT 问题开始对话，例如：VPN 登录提示账号无权限
            </div>
        </div>
        """

    bubbles = []
    for msg in chat_history:
        role = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            bubbles.append(f"""
            <div class="msg-row user-row">
                <div class="msg-avatar user-avatar">U</div>
                <div class="bubble user-bubble">
                    <div class="msg-meta">用户</div>
                    <div class="msg-content">{content}</div>
                </div>
            </div>
            """)
        else:
            bubbles.append(f"""
            <div class="msg-row assistant-row">
                <div class="msg-avatar assistant-avatar">AI</div>
                <div class="bubble assistant-bubble">
                    <div class="msg-meta">客服助手</div>
                    <div class="msg-content">{content}</div>
                </div>
            </div>
            """)

    return f"""
    <div class="chat-scroll">
        {''.join(bubbles)}
    </div>
    """


def badge(text: Any, kind: str = "default") -> str:
    text = html.escape(str(text))
    return f'<span class="badge badge-{kind}">{text}</span>'


def render_list_block(items: List[Dict[str, Any]], empty_text: str = "无") -> str:
    if not items:
        return f'<div class="mini-empty">{html.escape(empty_text)}</div>'

    cards = []
    for item in items[:3]:
        case_id = html.escape(str(item.get("case_id", "")))
        title = html.escape(str(item.get("title", "")))
        cards.append(f"""
        <div class="mini-item">
            <div class="mini-id">{case_id or "未命名案例"}</div>
            <div class="mini-title">{title or "无标题"}</div>
        </div>
        """)
    return "".join(cards)


def render_state_cards(state: Dict[str, Any]) -> str:
    if not state:
        return """
        <div class="state-empty">
            <div class="state-empty-title">当前暂无状态</div>
            <div class="state-empty-desc">
                当你发送第一条消息后，这里会展示当前轮的 Query、Trigger、Policy、Evidence 和 Latency。
            </div>
        </div>
        """

    query = state.get("query", {})
    trigger = state.get("trigger", {})
    policy = state.get("policy", {})
    evidence = state.get("evidence", {})
    latency = state.get("latency", {})

    query_mode = query.get("query_mode", "")
    trigger_flag = trigger.get("trigger", "")
    evidence_mode = trigger.get("evidence_mode", "")
    policy_label = policy.get("label", "")

    trigger_kind = "success" if trigger_flag is True else "muted"
    evidence_kind = {
        "retrieve_new_evidence": "warning",
        "reuse_existing_evidence": "info",
        "no_evidence_needed": "muted",
    }.get(evidence_mode, "default")

    policy_zh = {
        "AskMissingSlot": "请求补充信息",
        "AskClarification": "请求澄清",
        "ExplainedResponse": "解释性回复",
        "CaseRecommendation": "案例推荐",
        "Handoff": "转交处理",
        "ProcessAcknowledgement": "流程确认",
    }.get(policy_label, policy_label)

    active_evidence = evidence.get("active_evidence", [])
    used_evidence = evidence.get("evidence_used_this_turn", [])
    last_retrieved = evidence.get("last_retrieved_cases", [])

    active_short = short_case_list(active_evidence)
    used_short = short_case_list(used_evidence)
    last_short = short_case_list(last_retrieved)

    total_ms = latency.get("total_ms", 0)

    return f"""
    <div class="state-panel">

        <div class="section-card">
            <div class="section-head">
                <div class="section-index">01</div>
                <div class="section-title-wrap">
                    <div class="section-title">Query Update</div>
                    <div class="section-subtitle">本轮 query 改写与状态更新</div>
                </div>
            </div>
            <div class="kv-line">
                <span class="kv-key">Mode</span>
                <span class="kv-value">{badge(query_mode or "unknown", "info")}</span>
            </div>
            <div class="text-block">
                <div class="text-label">Query</div>
                <div class="text-value">{html.escape(str(query.get("query_text", ""))) or "无"}</div>
            </div>
            <div class="reason-box">{html.escape(str(query.get("reason", "")) or "无")}</div>
        </div>

        <div class="section-card">
            <div class="section-head">
                <div class="section-index">02</div>
                <div class="section-title-wrap">
                    <div class="section-title">Trigger Decision</div>
                    <div class="section-subtitle">是否需要触发 retrieval 与 evidence 策略</div>
                </div>
            </div>
            <div class="dual-grid">
                <div class="info-chip">
                    <div class="info-chip-label">Trigger</div>
                    <div class="info-chip-value">{badge(trigger_flag, trigger_kind)}</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">Evidence Mode</div>
                    <div class="info-chip-value">{badge(evidence_mode or "unknown", evidence_kind)}</div>
                </div>
            </div>
            <div class="reason-box">{html.escape(str(trigger.get("reason", "")) or "无")}</div>
        </div>

        <div class="section-card">
            <div class="section-head">
                <div class="section-index">03</div>
                <div class="section-title-wrap">
                    <div class="section-title">Policy Decision</div>
                    <div class="section-subtitle">本轮策略动作与决策原因</div>
                </div>
            </div>
            <div class="kv-line">
                <span class="kv-key">Policy</span>
                <span class="kv-value">{badge(policy_label or "unknown", "success")}</span>
            </div>
            <div class="text-block">
                <div class="text-label">中文动作</div>
                <div class="text-value">{html.escape(str(policy_zh or "无"))}</div>
            </div>
            <div class="reason-box">{html.escape(str(policy.get("reason", "")) or "无")}</div>
        </div>

        <div class="section-card">
            <div class="section-head">
                <div class="section-index">04</div>
                <div class="section-title-wrap">
                    <div class="section-title">Evidence State</div>
                    <div class="section-subtitle">本轮使用、缓存中保留、最近检索到的 evidence</div>
                </div>
            </div>

            <div class="text-block">
                <div class="text-label">Source Query</div>
                <div class="text-value">{html.escape(str(evidence.get("evidence_source_query", ""))) or "无"}</div>
            </div>

            <div class="mini-section">
                <div class="mini-section-title">本轮使用 Evidence</div>
                {render_list_block(used_short, "本轮未使用 evidence")}
            </div>

            <div class="mini-section">
                <div class="mini-section-title">Active Evidence</div>
                {render_list_block(active_short, "当前没有 active evidence")}
            </div>

            <div class="mini-section">
                <div class="mini-section-title">Last Retrieved Cases</div>
                {render_list_block(last_short, "最近没有检索结果")}
            </div>
        </div>

        <div class="section-card">
            <div class="section-head">
                <div class="section-index">05</div>
                <div class="section-title-wrap">
                    <div class="section-title">Latency</div>
                    <div class="section-subtitle">本轮链路耗时拆解</div>
                </div>
            </div>

            <div class="latency-total">
                <span class="latency-total-num">{total_ms}</span>
                <span class="latency-total-unit">ms</span>
            </div>

            <div class="latency-grid">
                <div class="latency-item"><span>Query</span><b>{latency.get("query_ms", 0)} ms</b></div>
                <div class="latency-item"><span>Retrieval</span><b>{latency.get("retrieval_ms", 0)} ms</b></div>
                <div class="latency-item"><span>Trigger</span><b>{latency.get("trigger_ms", 0)} ms</b></div>
                <div class="latency-item"><span>Policy</span><b>{latency.get("policy_ms", 0)} ms</b></div>
                <div class="latency-item"><span>Response</span><b>{latency.get("response_ms", 0)} ms</b></div>
            </div>
        </div>

    </div>
    """


def render_full_json(state: Dict[str, Any]) -> Dict[str, Any]:
    return state or {}


# ==================== Main Interaction ====================

def chat(user_input: str, chat_history: List[Dict[str, str]], runtime: DynamicMultiTurnRuntime):
    if runtime is None:
        runtime = new_runtime()

    if chat_history is None:
        chat_history = []

    if not user_input or not user_input.strip():
        return (
            "",
            render_chat(chat_history),
            runtime,
            chat_history,
            render_state_cards({}),
            {},
        )

    user_input = user_input.strip()
    result = runtime.step(user_input)

    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result.response},
    ]

    state = build_state_sections(result)

    return (
        "",
        render_chat(chat_history),
        runtime,
        chat_history,
        render_state_cards(state),
        render_full_json(state),
    )


def reset_chat():
    runtime = new_runtime()
    chat_history = []
    state = {}

    return (
        render_chat(chat_history),
        runtime,
        chat_history,
        render_state_cards(state),
        render_full_json(state),
    )


def runtime_turns_to_chat_history(runtime: DynamicMultiTurnRuntime) -> List[Dict[str, str]]:
    """
    将 runtime.state.turns 转换为前端 HTML 渲染使用的 chat_history。
    """
    return [
        {"role": turn.role, "content": turn.text}
        for turn in runtime.state.turns
    ]


def load_prefix_from_text(
    prefix_text: str,
    runtime: DynamicMultiTurnRuntime,
):
    """
    从网页文本框加载历史 prefix。

    输入格式示例：
    用户: 我的 VPN 登录不了
    客服: 请问有什么具体报错？
    用户: 提示账号无权限
    """
    if runtime is None:
        runtime = new_runtime()

    if not prefix_text or not prefix_text.strip():
        status_html = """
        <div class="prefix-status prefix-error">
            请输入历史对话。格式示例：用户: ... / 客服: ...
        </div>
        """
        return (
            render_chat([]),
            runtime,
            [],
            render_state_cards({}),
            {},
            status_html,
        )

    try:
        runtime.load_dialogue_prefix_text(prefix_text, reset_state=True)
        chat_history = runtime_turns_to_chat_history(runtime)

        status_html = f"""
        <div class="prefix-status prefix-success">
            已加载历史对话：共 {len(runtime.state.turns)} 条消息，
            已完成客服轮数 {runtime.state.current_turn_id}。
            现在可以在左侧输入下一轮用户问题，观察系统回复。
        </div>
        """

        return (
            render_chat(chat_history),
            runtime,
            chat_history,
            render_state_cards({}),
            {},
            status_html,
        )

    except Exception as e:
        status_html = f"""
        <div class="prefix-status prefix-error">
            加载失败：{html.escape(str(e))}
        </div>
        """
        return (
            render_chat([]),
            runtime,
            [],
            render_state_cards({}),
            {},
            status_html,
        )



# ==================== CSS ====================

CUSTOM_CSS = """
:root {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --panel-soft: #fafbfc;
    --line: #e6eaf0;
    --line-soft: #eef2f6;
    --text: #0f1728;
    --text-2: #344054;
    --text-3: #667085;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --green: #16a34a;
    --green-soft: #effdf3;
    --amber: #d97706;
    --amber-soft: #fffbeb;
    --shadow: 0 10px 30px rgba(15, 23, 40, 0.06);
    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;
    --radius-sm: 12px;
}

html, body {
    background: var(--bg) !important;
}

body {
    color: var(--text);
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-top: 20px !important;
    padding-bottom: 28px !important;
}

#hero {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 26px 30px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
}

.hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}

.hero-title-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.hero-badge {
    width: fit-content;
    padding: 6px 10px;
    background: var(--blue-soft);
    color: var(--blue);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.2;
}

.hero-desc {
    font-size: 14px;
    color: var(--text-3);
    line-height: 1.7;
    max-width: 780px;
}

.hero-side {
    min-width: 220px;
    display: flex;
    justify-content: flex-end;
}

.hero-metrics {
    background: #f8fafc;
    border: 1px solid var(--line-soft);
    border-radius: 18px;
    padding: 12px 14px;
    min-width: 200px;
}

.hero-metrics-title {
    font-size: 12px;
    color: var(--text-3);
    margin-bottom: 8px;
}

.hero-metrics-item {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--text-2);
    margin-top: 4px;
}

.main-card {
    background: transparent;
}

.chat-shell, .state-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 24px;
    box-shadow: var(--shadow);
    overflow: hidden;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 22px;
    border-bottom: 1px solid var(--line-soft);
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
}

.panel-title-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.panel-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
}

.panel-subtitle {
    font-size: 13px;
    color: var(--text-3);
}

.panel-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: #22c55e;
    box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.12);
    margin-right: 6px;
}

.chat-box {
    height: 640px;
    overflow-y: auto;
    background: linear-gradient(180deg, #fcfdff 0%, #f8fbff 100%);
    padding: 22px;
}

.chat-scroll {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.empty-chat {
    height: 580px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-3);
    text-align: center;
    padding: 0 40px;
}

.empty-chat-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: var(--blue-soft);
    color: var(--blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 16px;
    font-weight: 700;
}

.empty-chat-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 10px;
}

.empty-chat-subtitle {
    font-size: 14px;
    line-height: 1.8;
    max-width: 520px;
}

.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    width: 100%;
}

.user-row {
    justify-content: flex-end;
}

.assistant-row {
    justify-content: flex-start;
}

.user-row .bubble {
    order: 1;
}

.user-row .msg-avatar {
    order: 2;
}

.assistant-row .bubble {
    order: 2;
}

.assistant-row .msg-avatar {
    order: 1;
}

.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 800;
    flex-shrink: 0;
    margin-top: 2px;
}

.user-avatar {
    background: #dbeafe;
    color: #1d4ed8;
}

.assistant-avatar {
    background: #e8ecf3;
    color: #334155;
}

.bubble {
    max-width: 76%;
    border-radius: 18px;
    padding: 12px 14px;
    line-height: 1.7;
    font-size: 14px;
    word-break: break-word;
}

.user-bubble {
    background: linear-gradient(180deg, #2b6ef3 0%, #2563eb 100%);
    color: white;
    border-bottom-right-radius: 8px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
}

.assistant-bubble {
    background: #ffffff;
    color: var(--text);
    border: 1px solid var(--line-soft);
    border-bottom-left-radius: 8px;
}

.msg-meta {
    font-size: 12px;
    opacity: 0.78;
    margin-bottom: 4px;
    font-weight: 600;
}

.msg-content {
    white-space: pre-wrap;
}

.input-area {
    padding: 16px 18px 18px 18px;
    border-top: 1px solid var(--line-soft);
    background: #ffffff;
}

.state-html {
    height: 640px;
    overflow-y: auto;
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    padding: 18px;
}

.state-empty {
    min-height: 560px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    color: var(--text-3);
    padding: 0 24px;
}

.state-empty-title {
    font-size: 20px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 10px;
}

.state-empty-desc {
    font-size: 14px;
    line-height: 1.8;
}

.state-panel {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.section-card {
    border: 1px solid var(--line-soft);
    border: 18px;
    background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
    padding: 16px;
}

.section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 14px;
}

.section-index {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: #f2f6ff;
    color: var(--blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
    flex-shrink: 0;
}

.section-title-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.section-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--text);
}

.section-subtitle {
    font-size: 12px;
    color: var(--text-3);
}

.kv-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 10px 0;
}

.kv-key {
    font-size: 13px;
    color: var(--text-3);
}

.kv-value {
    color: var(--text);
}

.text-block {
    margin-top: 12px;
    padding: 12px 13px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid var(--line-soft);
}

.text-label {
    font-size: 12px;
    color: var(--text-3);
    margin-bottom: 6px;
}

.text-value {
    font-size: 14px;
    color: var(--text);
    line-height: 1.6;
    word-break: break-word;
}

.reason-box {
    margin-top: 12px;
    padding: 12px 13px;
    border-radius: 14px;
    background: #fcfcfd;
    border: 1px dashed #d8e0ea;
    color: #475467;
    font-size: 13px;
    line-height: 1.7;
    word-break: break-word;
}

.dual-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.info-chip {
    padding: 12px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid var(--line-soft);
}

.info-chip-label {
    font-size: 12px;
    color: var(--text-3);
    margin-bottom: 8px;
}

.info-chip-value {
    font-size: 14px;
    color: var(--text);
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    border 1px solid transparent;
}

.badge-default {
    background: #f3f4f6;
    color: #475467;
}

.badge-info {
    background: #eff6ff;
    color: #1d4ed8;
    border-color: #dbeafe;
}

.badge-success {
    background: #effdf3;
    color: #15803d;
    border-color: #c7f0d5;
}

.badge-warning {
    background: #fff7ed;
    color: #c2410c;
    border-color: #fed7aa;
}

.badge-muted {
    background: #f3f4f6;
    color: #667085;
    border-color: #e5e7eb;
}

.mini-section {
    margin-top: 14px;
}

.mini-section-title {
    font-size: 12px;
    color: var(--text-3);
    margin-bottom: 8px;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.mini-item {
    padding: 10px 12px;
    border: 1px solid var(--line-soft);
    background: #fbfcfe;
    border-radius: 12px;
    margin-bottom: 8px;
}

.mini-id {
    font-size: 12px;
    color: var(--blue);
    font-weight: 700;
    margin-bottom: 4px;
}

.mini-title {
    font-size: 13px;
    color: var(--text-2);
    line-height: 1.5;
}

.mini-empty {
    padding: 10px 12px;
    border-radius: 12px;
    background: #f8fafc;
    color: var(--text-3);
    font-size: 13px;
    border: 1px dashed #d8e0ea;
}

.latency-total {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin: 4px 0 14px 0;
}

.latency-total-num {
    font-size: 34px;
    font-weight: 800;
    color: var(--blue);
    line-height: 1;
}

.latency-total-unit {
    font-size: 14px;
    color: var(--text-3);
    font-weight: 700;
}

.latency-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.latency-item {
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 12px;
    background: #f8fafc;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.latency-item span {
    font-size: 12px;
    color: var(--text-3);
}

.latency-item b {
    font-size: 14px;
    color: var(--text);
}

button.primary,
button[variant="primary"] {
    border-radius: 14px !important;
}


.prefix-status {
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 10px;
}

.prefix-success {
    background: #effdf3;
    border: 1px solid #abefc6;
    color: #027a48;
}

.prefix-error {
    background: #fff1f3;
    border: 1px solid #fecdd6;
    color: #be123c;
}

.prefix-help {
    color: var(--text-3);
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 10px;
}


footer {
    display: none !important;
}

@media (max-width: 1100px) {
    .hero-top {
        flex-direction: column;
        align-items: flex-start;
    }

    .hero-side {
        width: 100%;
        justify-content: flex-start;
    }

    .bubble {
        max-width: 88%;
    }

    .dual-grid,
    .latency-grid {
        grid-template-columns: 1fr;
    }
}
"""


# ==================== Gradio App ====================

with gr.Blocks(
    title="华为 IT 动态多轮客服助手",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
    ),
) as demo:
    runtime_state = gr.State(value=new_runtime())
    chat_history_state = gr.State(value=[])

    gr.HTML("""
    <div id="hero">
        <div class="hero-top">
            <div class="hero-title-wrap">
                <div class="hero-badge">Huawei IT Support Assistant</div>
                <div class="hero-title">华为 IT 动态多轮客服助手</div>
                <div class="hero-desc">
                    面向多轮 IT 支持场景的动态客服系统，支持 Query Update、Evidence Trigger、Policy Decision 与 Response Generation 的可解释闭环。
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-metrics">
                    <div class="hero-metrics-title">系统能力</div>
                    <div class="hero-metrics-item"><span>多轮状态跟踪</span><b>Enabled</b></div>
                    <div class="hero-metrics-item"><span>动态证据复用</span><b>Enabled</b></div>
                    <div class="hero-metrics-item"><span>策略可解释性</span><b>Enabled</b></div>
                </div>
            </div>
        </div>
    </div>
    """)

    with gr.Row(elem_classes=["main-card"]):
        with gr.Column(scale=7, min_width=680):
            with gr.Group(elem_classes=["chat-shell"]):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-title-wrap">
                        <div class="panel-title">对话面板</div>
                        <div class="panel-subtitle">与系统进行多轮交互，左侧展示完整对话过程</div>
                    </div>
                    <div class="panel-dot"></div>
                </div>
                """)

                chat_display = gr.HTML(
                    value=render_chat([]),
                    elem_classes=["chat-box"],
                )

                with gr.Group(elem_classes=["input-area"]):
                    with gr.Row():
                        user_box = gr.Textbox(
                            placeholder="请输入用户问题，例如：VPN 登录提示账号无权限",
                            show_label=False,
                            scale=6,
                            lines=3,
                        )
                        send_btn = gr.Button("发送", scale=1, variant="primary")

                    reset_btn = gr.Button("清空并重新开始")

                    with gr.Accordion("加载历史对话 Prefix", open=False):
                        gr.HTML("""
                        <div class="prefix-help">
                            输入已有历史对话，系统会将其作为 dialogue history 加载。
                            格式示例：<br>
                            用户: 我的 VPN 登录不了<br>
                            客服: 请问有什么具体报错？<br>
                            用户: 提示账号无权限
                        </div>
                        """)
                        prefix_box = gr.Textbox(
                            placeholder=(
                                "用户: 我的 VPN 登录不了\n"
                                "客服: 请问有什么具体报错？\n"
                                "用户: 提示账号无权限"
                            ),
                            show_label=False,
                            lines=8,
                        )
                        load_prefix_btn = gr.Button("加载为历史对话", variant="secondary")
                        prefix_status = gr.HTML(value="")

        with gr.Column(scale=5, min_width=520):
            with gr.Group(elem_classes=["state-shell"]):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-title-wrap">
                        <div class="panel-title">当前轮状态</div>
                        <div class="panel-subtitle">系统内部决策链路与证据状态展示</div>
                    </div>
                </div>
                """)

                state_cards = gr.HTML(
                    value=render_state_cards({}),
                    elem_classes=["state-html"],
                )

                with gr.Accordion("查看完整 JSON 状态", open=False):
                    state_json = gr.JSON(value={}, label=None)

    send_btn.click(
        fn=chat,
        inputs=[user_box, chat_history_state, runtime_state],
        outputs=[
            user_box,
            chat_display,
            runtime_state,
            chat_history_state,
            state_cards,
            state_json,
        ],
    )

    user_box.submit(
        fn=chat,
        inputs=[user_box, chat_history_state, runtime_state],
        outputs=[
            user_box,
            chat_display,
            runtime_state,
            chat_history_state,
            state_cards,
            state_json,
        ],
    )

    reset_btn.click(
        fn=reset_chat,
        inputs=[],
        outputs=[
            chat_display,
            runtime_state,
            chat_history_state,
            state_cards,
            state_json,
        ],
    )

    load_prefix_btn.click(
        fn=load_prefix_from_text,
        inputs=[prefix_box, runtime_state],
        outputs=[
            chat_display,
            runtime_state,
            chat_history_state,
            state_cards,
            state_json,
            prefix_status,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="10.67.43.9",
        server_port=7860,
        share=False,
    )
