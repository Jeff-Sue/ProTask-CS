import gradio as gr
from dynamic_multiturn_runtime import DynamicMultiTurnRuntime

runtime = DynamicMultiTurnRuntime(
    dialogue_id="web_demo",
    retrieval_top_k=5,
    save_history_path="web_runtime_output.json",
    verbose=False,
)

def chat_fn(message, history):
    result = runtime.step(message)
    return result.response

demo = gr.ChatInterface(
    fn=chat_fn,
    title="动态多轮 IT 客服系统",
    description="基于 Query / Trigger / Evidence / Policy / Response 的动态多轮 RAG 客服 Demo",
)

demo.launch(server_name="0.0.0.0", server_port=7860)