import uuid
import gradio as gr

from run import DynamicMultiTurnRuntime


def new_runtime():
    session_id = f"gradio_{uuid.uuid4().hex[:8]}"
    return DynamicMultiTurnRuntime(
        dialogue_id=session_id,
        retrieval_top_k=5,
        save_history_path=f"gradio_runtime_{session_id}.json",
        verbose=False,
    )


def dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def build_state_sections(result):
    evidence_state = result.state_snapshot.get("evidence_state", {})

    query_section = {
        "turn_id": result.turn_id,
        "query_state": dataclass_to_dict(result.query_state),
    }

    trigger_section = {
        "trigger_state": dataclass_to_dict(result.trigger_state),
    }

    policy_section = {
        "policy_state": dataclass_to_dict(result.policy_state),
    }

    evidence_section = {
        "evidence_used_this_turn": dataclass_to_dict(result.retrieval_output),
        "active_evidence": dataclass_to_dict(evidence_state.get("active_evidence", [])),
        "last_retrieved_cases": dataclass_to_dict(evidence_state.get("last_retrieved_cases", [])),
        "evidence_source_query": evidence_state.get("evidence_source_query", ""),
    }

    latency_section = {
        "latency": dataclass_to_dict(result.latency),
    }

    return (
        query_section,
        trigger_section,
        policy_section,
        evidence_section,
        latency_section,
    )


def append_chat_history(chat_history, user_input, assistant_response):
    if chat_history is None:
        chat_history = []

    return chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response},
    ]


def chat(user_input, chat_history, runtime):
    if runtime is None:
        runtime = new_runtime()

    if chat_history is None:
        chat_history = []

    if not user_input or not user_input.strip():
        return "", chat_history, runtime, {}, {}, {}, {}, {}

    user_input = user_input.strip()
    result = runtime.step(user_input)

    chat_history = append_chat_history(
        chat_history=chat_history,
        user_input=user_input,
        assistant_response=result.response,
    )

    query_section, trigger_section, policy_section, evidence_section, latency_section = build_state_sections(result)

    return (
        "",
        chat_history,
        runtime,
        query_section,
        trigger_section,
        policy_section,
        evidence_section,
        latency_section,
    )


def reset_chat():
    runtime = new_runtime()
    return [], runtime, {}, {}, {}, {}, {}


with gr.Blocks(title="动态多轮 IT 客服系统") as demo:
    runtime_state = gr.State(value=new_runtime())

    gr.Markdown("# 动态多轮 IT 客服系统")

    with gr.Row():
        # 左侧聊天区
        with gr.Column(scale=2, min_width=520):
            chatbot = gr.Chatbot(
                label="对话",
                height=620,
            )

            with gr.Row():
                user_box = gr.Textbox(
                    placeholder="请输入用户问题，例如：VPN 登录提示账号无权限",
                    label="用户输入",
                    scale=5,
                    lines=2,
                )
                send_btn = gr.Button("发送", scale=1)

            reset_btn = gr.Button("重置对话")

        # 右侧状态区
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("## 当前轮状态")

            query_json = gr.JSON(label="Query", value={})
            trigger_json = gr.JSON(label="Trigger", value={})
            policy_json = gr.JSON(label="Policy", value={})
            evidence_json = gr.JSON(label="Evidence", value={})
            latency_json = gr.JSON(label="Latency", value={})

    send_btn.click(
        fn=chat,
        inputs=[user_box, chatbot, runtime_state],
        outputs=[
            user_box,
            chatbot,
            runtime_state,
            query_json,
            trigger_json,
            policy_json,
            evidence_json,
            latency_json,
        ],
    )

    user_box.submit(
        fn=chat,
        inputs=[user_box, chatbot, runtime_state],
        outputs=[
            user_box,
            chatbot,
            runtime_state,
            query_json,
            trigger_json,
            policy_json,
            evidence_json,
            latency_json,
        ],
    )

    reset_btn.click(
        fn=reset_chat,
        inputs=[],
        outputs=[
            chatbot,
            runtime_state,
            query_json,
            trigger_json,
            policy_json,
            evidence_json,
            latency_json,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
