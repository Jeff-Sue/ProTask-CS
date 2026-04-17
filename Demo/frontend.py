import json
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Dialogue Comparison Demo", layout="wide")
st.title("多轮对话对比评测页面")

# =========================
# Session State Init
# =========================
if "session_id" not in st.session_state:
    resp = requests.post(f"{BACKEND_URL}/create_session")
    st.session_state.session_id = resp.json()["session_id"]

if "gt_messages" not in st.session_state:
    st.session_state.gt_messages = []

if "api_messages" not in st.session_state:
    st.session_state.api_messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "agent_last_result" not in st.session_state:
    st.session_state.agent_last_result = None


# =========================
# Helpers
# =========================
def parse_ground_truth(text: str):
    """
    支持输入 JSON list:
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    """
    data = json.loads(text)
    assert isinstance(data, list), "Ground truth 必须是一个 list"
    for item in data:
        assert "role" in item and "content" in item, "每项都必须含 role 和 content"
    return data


def render_chat_column(title, messages, show_strategy=False, show_step=False, last_result=None):
    st.subheader(title)

    for msg in messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                if show_strategy and msg.get("strategy"):
                    st.markdown(f"**历史策略：** `{msg['strategy']}`")
                st.markdown(msg["content"])

    if show_step and last_result:
        st.markdown("---")
        st.markdown("### 当前轮 Two-Stage 过程")
        st.markdown("**Step 1：根据历史对话生成策略**")
        st.code(last_result.get("strategy", ""), language=None)

        extra_context = last_result.get("extra_context", {})
        if extra_context:
            st.markdown("**中间上下文**")
            st.json(extra_context)

        st.markdown("**Step 2：根据 策略 + 历史对话 生成回复**")
        st.info(last_result.get("response", ""))


def mock_api_reply(user_input: str, history):
    """
    你后面可以替换成真实 API
    """
    return f"【API回复】这是对“{user_input}”的模拟API回答。"


# =========================
# Top Control Panel
# =========================
with st.expander("Ground Truth 配置", expanded=True):
    st.markdown("请在下面输入 ground truth 的多轮对话 JSON 列表。")

    default_gt = json.dumps([
        {"role": "user", "content": "我电脑连不上VPN"},
        {"role": "assistant", "content": "请问报错信息是什么？"},
        {"role": "user", "content": "提示691错误"},
        {"role": "assistant", "content": "好的，691通常和认证失败有关，请先确认账号密码是否正确。"}
    ], ensure_ascii=False, indent=2)

    gt_text = st.text_area(
        "Ground Truth 对话列表",
        value=default_gt,
        height=250
    )

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("加载 Ground Truth"):
            try:
                st.session_state.gt_messages = parse_ground_truth(gt_text)
                st.success("Ground Truth 已加载。")
            except Exception as e:
                st.error(f"Ground Truth 解析失败：{e}")

    with col_btn2:
        if st.button("清空全部会话"):
            st.session_state.gt_messages = []
            st.session_state.api_messages = []
            st.session_state.agent_messages = []
            st.session_state.agent_last_result = None

            resp = requests.post(f"{BACKEND_URL}/create_session")
            st.session_state.session_id = resp.json()["session_id"]

            st.success("已清空并创建新会话。")

st.caption(f"当前 Two-Stage Session ID: {st.session_state.session_id}")

# =========================
# Three Columns Comparison
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    render_chat_column(
        title="Ground Truth",
        messages=st.session_state.gt_messages,
        show_strategy=False,
        show_step=False
    )

with col2:
    render_chat_column(
        title="API 回复",
        messages=st.session_state.api_messages,
        show_strategy=False,
        show_step=False
    )

with col3:
    render_chat_column(
        title="Two-Stage Agent",
        messages=st.session_state.agent_messages,
        show_strategy=True,
        show_step=True,
        last_result=st.session_state.agent_last_result
    )

# =========================
# Bottom Input Area
# =========================
st.markdown("---")
st.subheader("当前轮输入")

user_input = st.text_input("请输入当前这一轮用户消息", value="")

col_send1, col_send2 = st.columns([1, 5])

with col_send1:
    send_clicked = st.button("发送")

if send_clicked and user_input.strip():
    # 1) Ground truth: 这里只追加当前用户轮，不自动生成 GT assistant
    st.session_state.gt_messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    # 2) API 回复
    st.session_state.api_messages.append({
        "role": "user",
        "content": user_input.strip()
    })
    api_reply = mock_api_reply(user_input.strip(), st.session_state.api_messages)
    st.session_state.api_messages.append({
        "role": "assistant",
        "content": api_reply
    })

    # 3) Two-stage agent
    st.session_state.agent_messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    resp = requests.post(
        f"{BACKEND_URL}/chat",
        json={
            "session_id": st.session_state.session_id,
            "user_input": user_input.strip(),
        },
        timeout=120,
    )
    result = resp.json()
    st.session_state.agent_last_result = result

    st.session_state.agent_messages.append({
        "role": "assistant",
        "strategy": result["strategy"],
        "content": result["response"]
    })

    st.rerun()