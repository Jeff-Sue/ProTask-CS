import uuid
import inspect
import gradio as gr

from dynamic_multiturn_runtime import DynamicMultiTurnRuntime


def chatbot_supports_type() -> bool:
    return "type" in inspect.signature(gr.Chatbot).parameters


USE_MESSAGES_FORMAT = chatbot_supports_type()


def new_runtime():
    session_id = f"gradio_{uuid.uuid4().hex[:8]}"
    return DynamicMultiTurnRuntime(
        dialogue_id=session_id,
        retrieval_top_k=5,
        save_history_path=f"gradio_runtime_{session_id}.json",
        verbose=False,
    )


def dataclass_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def build_state_panel(result):
    evidence_state = result.state_snapshot.get("evidence_state", {})

    return {
        "turn_id": result.turn_id,
        "query_state": dataclass_to_dict(result.query_state),
        "trigger_state": dataclass_to_dict(result.trigger_state),
        "policy_state": dataclass_to_dict(result.policy_state),
        "evidence_used_this_turn": [
            dataclass_to_dict(case) for case in result.retrieval_output
        ],
        "active_evidence": evidence_state.get("active_evidence", []),
        "last_retrieved_cases": evidence_state.get("last_retrieved_cases", []),
        "evidence_source_query": evidence_state.get("evidence_source_query", ""),
        "latency": dataclass_to_dict(result.latency),
    }


def append_chat_history(chat_history, user_input, assistant_response):
    if chat_history is None:
        chat_history = []

    if USE_MESSAGES_FORMAT:
        return chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]

    return chat_history + [
        [user_input, assistant_response]
    ]


def chat(user_input, chat_history, runtime):
    if runtime is None:
        runtime = new_runtime()

    if chat_history is None:
        chat_history = []

    if not user_input or not user_input.strip():
        return "", chat_history, runtime, {}

    result = runtime.step(user_input.strip())

    chat_history = append_chat_history(
        chat_history=chat_history,
        user_input=user_input.strip(),
        assistant_response=result.response,
    )

    state_panel = build_state_panel(result)

    return "", chat_history, runtime, state_panel


def reset_chat():
    runtime = new_runtime()
    return [], runtime, {}


with gr.Blocks(title="动态多轮 IT 客服系统") as demo:
    runtime_state = gr.State(value=new_runtime())

    gr.Markdown("# 动态多轮 IT 客服系统")

    with gr.Row():
        with gr.Column(scale=2, min_width=520):
            if USE_MESSAGES_FORMAT:
                chatbot = gr.Chatbot(
                    label="对话",
                    height=620,
                    type="messages",
                )
            else:
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
    print(f"Gradio Chatbot format: {'messages' if USE_MESSAGES_FORMAT else 'tuples'}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )