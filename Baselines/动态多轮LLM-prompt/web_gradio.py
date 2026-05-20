import html
import json
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr

from run import DynamicMultiTurnRuntime


# ==================== Runtime ====================

def new_runtime() -> DynamicMultiTurnRuntime:
    session_id = f"gradio_{uuid.uuid4().hex[:8]}"
    return DynamicMultiTurnRuntime(
        dialogue_id=session_id,
        retrieval_top_k=5,
        save_history_path=f"gradio_runtime_{session_id}.json",
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


def safe_get(d: Dict[str, Any], path: List[str], default: Any = "") -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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

    state = {
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

    return state


# ==================== HTML Render ====================

def render_chat(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return """
        <div class="empty-chat">
            <div class="empty-title">欢迎使用动态多轮 IT 客服系统</div>
            <div class="empty-subtitle">请输入一个 IT 问题开始对话，例如：VPN 登录提示账号无权限</div>
        </div>
        """

    bubbles = []
    for msg in chat_history:
        role = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            bubbles.append(f"""
            <div class="msg-row user-row">
                <div class="bubble user-bubble">
                    <div class="role-label">用户</div>
                    <div class="msg-content">{content}</div>
                </div>
            </div>
            """)
        else:
            bubbles.append(f"""
            <div class="msg-row bot-row">
                <div class="bubble bot-bubble">
                    <div class="role-label">客服</div>
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


def render_state_cards(state: Dict[str, Any]) -> str:
    if not state:
        return """
        <div class="state-empty">
            当前还没有状态。发送一条用户消息后，将在这里显示 Query、Trigger、Policy、Evidence 和耗时。
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

        <div class="state-card">
            <div class="card-title">1. Query Update</div>
            <div class="card-line">Mode: {badge(query_mode, "info")}</div>
            <div class="card-line"><b>Query:</b> {html.escape(str(query.get("query_text", ""))) or "无"}</div>
            <div class="reason">{html.escape(str(query.get("reason", "")))}</div>
        </div>

        <div class="state-card">
            <div class="card-title">2. Trigger Decision</div>
            <div class="card-line">Trigger: {badge(trigger_flag, trigger_kind)}</div>
            <div class="card-line">Evidence Mode: {badge(evidence_mode, evidence_kind)}</div>
            <div class="reason">{html.escape(str(trigger.get("reason", "")))}</div>
        </div>

        <div class="state-card">
            <div class="card-title">3. Policy Decision</div>
            <div class="card-line">Policy: {badge(policy_label, "success")}</div>
            <div class="card-line"><b>中文动作:</b> {html.escape(str(policy_zh))}</div>
            <div class="reason">{html.escape(str(policy.get("reason", "")))}</div>
        </div>

        <div class="state-card">
            <div class="card-title">4. Evidence State</div>
            <div class="card-line"><b>Source Query:</b> {html.escape(str(evidence.get("evidence_source_query", ""))) or "无"}</div>
            <div class="evidence-block">
                <div class="evidence-title">本轮使用 evidence</div>
                <pre>{html.escape(json.dumps(used_short, ensure_ascii=False, indent=2))}</pre>
            </div>
            <div class="evidence-block">
                <div class="evidence-title">Active Evidence</div>
                <pre>{html.escape(json.dumps(active_short, ensure_ascii=False, indent=2))}</pre>
            </div>
            <div class="evidence-block">
                <div class="evidence-title">Last Retrieved Cases</div>
                <pre>{html.escape(json.dumps(last_short, ensure_ascii=False, indent=2))}</pre>
            </div>
        </div>

        <div class="state-card">
            <div class="card-title">5. Latency</div>
            <div class="latency-total">{total_ms} ms</div>
            <div class="latency-grid">
                <span>Query</span><b>{latency.get("query_ms", 0)} ms</b>
                <span>Retrieval</span><b>{latency.get("retrieval_ms", 0)} ms</b>
                <span>Trigger</span><b>{latency.get("trigger_ms", 0)} ms</b>
                <span>Policy</span><b>{latency.get("policy_ms", 0)} ms</b>
                <span>Response</span><b>{latency.get("response_ms", 0)} ms</b>
            </div>
        </div>

    </div>
    """


def render_full_json(state: Dict[str, Any]) -> Dict[str, Any]:
    return state or {}


# ==================== Main Interaction ====================

def chat(
    user_input: str,
    chat_history: List[Dict[str, str]],
    runtime: DynamicMultiTurnRuntime,
):
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


def fill_example(example_text: str):
    return example_text


# ==================== CSS ====================

CUSTOM_CSS = """
body {
    background: #f6f7fb;
}

.gradio-container {
    max-width: 1600px !important;
}

#main-title {
    padding: 18px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, #182848 0%, #4b6cb7 100%);
    color: white;
    margin-bottom: 16px;
}

#main-title h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
}

#main-title p {
    margin: 6px 0 0 0;
    opacity: 0.9;
    font-size: 14px;
}

.chat-box {
    height: 650px;
    overflow-y: auto;
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(20, 30, 60, 0.06);
}

.chat-scroll {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.empty-chat {
    height: 610px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #667085;
    text-align: center;
}

.empty-title {
    font-size: 22px;
    font-weight: 700;
    color: #344054;
    margin-bottom: 10px;
}

.empty-subtitle {
    font-size: 14px;
}

.msg-row {
    display: flex;
    width: 100%;
}

.user-row {
    justify-content: flex-end;
}

.bot-row {
    justify-content: flex-start;
}

.bubble {
    max-width: 78%;
    border-radius: 18px;
    padding: 12px 14px;
    line-height: 1.55;
    font-size: 15px;
    word-break: break-word;
}

.user-bubble {
    background: #3867ff;
    color: white;
    border-bottom-right-radius: 6px;
}

.bot-bubble {
    background: #f2f4f7;
    color: #1d2939;
    border-bottom-left-radius: 6px;
}

.role-label {
    font-size: 12px;
    opacity: 0.75;
    margin-bottom: 4px;
}

.msg-content {
    white-space: pre-wrap;
}

.state-html {
    height: 650px;
    overflow-y: auto;
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(20, 30, 60, 0.06);
}

.state-empty {
    color: #667085;
    padding: 18px;
    line-height: 1.6;
}

.state-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.state-card {
    border: 1px solid #eaecf0;
    border-radius: 14px;
    padding: 12px;
    background: #fcfcfd;
}

.card-title {
    font-weight: 700;
    color: #101828;
    margin-bottom: 8px;
    font-size: 15px;
}

.card-line {
    font-size: 13px;
    color: #344054;
    margin: 6px 0;
}

.reason {
    margin-top: 8px;
    padding: 8px;
    border-radius: 10px;
    background: #f8fafc;
    color: #475467;
    font-size: 13px;
    line-height: 1.45;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid transparent;
}

.badge-default {
    background: #f2f4f7;
    color: #344054;
}

.badge-info {
    background: #eef4ff;
    color: #3538cd;
    border-color: #c7d7fe;
}

.badge-success {
    background: #ecfdf3;
    color: #027a48;
    border-color: #abefc6;
}

.badge-warning {
    background: #fffaeb;
    color: #b54708;
    border-color: #fedf89;
}

.badge-muted {
    background: #f2f4f7;
    color: #667085;
    border-color: #eaecf0;
}

.evidence-block {
    margin-top: 8px;
}

.evidence-title {
    font-size: 12px;
    color: #667085;
    margin-bottom: 4px;
}

pre {
    background: #0b1020;
    color: #d1e7ff;
    padding: 8px;
    border-radius: 10px;
    font-size: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
}

.latency-total {
    font-size: 24px;
    font-weight: 800;
    color: #175cd3;
    margin-bottom: 10px;
}

.latency-grid {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 6px 12px;
    font-size: 13px;
    color: #475467;
}

.example-row button {
    font-size: 13px !important;
}
"""


# ==================== Gradio App ====================

with gr.Blocks(
    title="华为 IT 动态多轮客服助手",
    css=CUSTOM_CSS,
) as demo:
    runtime_state = gr.State(value=new_runtime())
    chat_history_state = gr.State(value=[])

    gr.HTML("""
    <div id="main-title">
        <h1>华为 IT 动态多轮客服助手</h1>
        <p>Query Update · Evidence Trigger · Policy Decision · Response Generation 可解释闭环</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3, min_width=620):
            chat_display = gr.HTML(
                value=render_chat([]),
                elem_classes=["chat-box"],
            )

            with gr.Row():
                user_box = gr.Textbox(
                    placeholder="请输入用户问题，例如：VPN 登录提示账号无权限",
                    label="用户输入",
                    scale=6,
                    lines=2,
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")

            with gr.Row(elem_classes=["example-row"]):
                ex1 = gr.Button("VPN 登录不了")
                ex2 = gr.Button("提示账号无权限")
                ex3 = gr.Button("我已经申请过权限了")
                ex4 = gr.Button("工号是 12345，申请单号是 REQ888")
                ex5 = gr.Button("另外我手机邮箱也收不到验证码")

            reset_btn = gr.Button("清空并重新开始")

        with gr.Column(scale=2, min_width=520):
            gr.Markdown("## 当前轮状态")
            state_cards = gr.HTML(
                value=render_state_cards({}),
                elem_classes=["state-html"],
            )

            with gr.Accordion("完整 JSON 状态", open=False):
                state_json = gr.JSON(value={}, label="Full State")

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

    ex1.click(fn=lambda: "VPN 登录不了", inputs=[], outputs=[user_box])
    ex2.click(fn=lambda: "提示账号无权限", inputs=[], outputs=[user_box])
    ex3.click(fn=lambda: "我已经申请过权限了", inputs=[], outputs=[user_box])
    ex4.click(fn=lambda: "工号是 12345，申请单号是 REQ888", inputs=[], outputs=[user_box])
    ex5.click(fn=lambda: "另外我手机邮箱也收不到验证码", inputs=[], outputs=[user_box])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )