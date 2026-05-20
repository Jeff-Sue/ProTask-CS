import uuid
import gradio as gr

from dynamic_multiturn_runtime import DynamicMultiTurnRuntime


def new_runtime():
    session_id = f"gradio_{uuid.uuid4().hex[:8]}"
    return DynamicMultiTurnRuntime(
        dialogue_id=session_id,
        retrieval_top_k=5,
        save_history_path=f"gradio_runtime_{session_id}.json",
        verbose=False,
    )


def build_state_panel(result):
    """把当前轮所有状态整理成右侧 JSON 面板。"""
    evidence_state = result.state_snapshot.get("evidence_state", {})

    return {
        "turn_id": result.turn_id,
        "query_state": result.query_state.__dict__,
        "trigger_state": result.trigger_state.__dict__,
        "policy_state": result.policy_state.__dict__,
        "evidence_used_this_turn": [
            case.__dict__ for case in result.retrieval_output
        ],
        "active_evidence": evidence_state.get("active_evidence", []),
        "last_retrieved_cases": evidence_state.get("last_retrieved_cases", []),
        "evidence_source_query": evidence_state.get("evidence_source_query", ""),
        "latency": result.latency.__dict__,
    }


def chat(user_input, chat_history, runtime):
    if runtime is None:
        runtime = new_runtime()

    if not user_input or not user_input.strip():
        return "", chat_history, runtime, {}

    result = runtime.step(user_input.strip())

    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result.response},
    ]

    state_panel = build_state_panel(result)

    return "", chat_history, runtime, state_panel


def reset_chat():
    runtime = new_runtime()
    return [], runtime, {}


with gr.Blocks(title="动态多轮 IT 客服系统") as demo:
    runtime_state = gr.State(value=new_runtime())

    gr.Markdown("# 动态多轮 IT 客服系统")

    with gr.Row():
        # 左侧：正常对话区
        with gr.Column(scale=2, min_width=520):
            chatbot = gr.Chatbot(
                label="对话",
                type="messages",
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

        # 右侧：状态面板
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("## 当前轮状态")
            state_json = gr.JSON(
                label="Query / Trigger / Policy / Evidence / Latency",
                value={},
            )

    send_btn.click(
        fn=chat,
        inputs=[user_box, chatbot, runtime_state],
        outputs=[user_box, chatbot, runtime_state, state_json],
    )

    user_box.submit(
        fn=chat,
        inputs=[user_box, chatbot, runtime_state],
        outputs=[user_box, chatbot, runtime_state, state_json],
    )

    reset_btn.click(
        fn=reset_chat,
        inputs=[],
        outputs=[chatbot, runtime_state, state_json],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )