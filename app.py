import os
from datetime import datetime

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st

from utils.config_handler import rag_conf

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="智扫通 · 扫地机器人智能客服",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"

QUICK_QUESTIONS = [
    ("efficiency", "最近清扫效率下降，怎么排查？", ":material/trending_down:"),
    ("report", "给我生成这个月的使用报告", ":material/description:"),
    ("recharge", "机器人回充失败怎么办？", ":material/battery_alert:"),
    ("consumables", "滤网和边刷多久更换一次？", ":material/autorenew:"),
]

FEATURES = [
    (":material/menu_book:", "专业知识库", "选购、保养、故障排查，都有据可查。"),
    (":material/description:", "个性化报告", "读取使用记录，生成专属月度总结。"),
    (":material/partly_cloudy_day:", "环境感知", "结合城市天气，给出更合适的清扫建议。"),
]


@st.cache_resource(show_spinner=False)
def get_agent():
    """按需加载 Agent，避免阻塞首屏渲染。"""
    from agent.react_agent import ReactAgent

    return ReactAgent()


def queue_prompt(prompt: str) -> None:
    """快捷提问按钮的回调：把问题交给主流程处理。"""
    st.session_state["pending_prompt"] = prompt


def clear_chat() -> None:
    """清空当前会话。"""
    st.session_state["message"] = []
    st.session_state["pending_prompt"] = None
    st.session_state["awaiting_reply"] = None
    st.session_state["toast_message"] = "已开始新的对话"


def render_history() -> None:
    for message in st.session_state["message"]:
        role = message["role"]
        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

        with st.chat_message(role, avatar=avatar):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

            meta = message.get("meta") or message.get("time")
            if meta:
                st.caption(meta)


def remember_user_message(prompt: str) -> None:
    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt,
            "time": datetime.now().strftime("%H:%M"),
        }
    )
    st.session_state["awaiting_reply"] = prompt


def remember_failed_answer(reply_time: str, content: str) -> None:
    st.session_state["message"].append(
        {
            "role": "assistant",
            "content": content,
            "time": reply_time,
            "meta": f"{reply_time} · 回答未完成",
            "error": True,
        }
    )


def render_welcome() -> None:
    st.subheader("您好，我是智扫通 👋")
    st.caption("懂产品，也懂您的使用场景。检索知识库、分析环境与使用记录，给您可执行的建议。")
    st.space("small")

    feature_cols = st.columns(3)
    for col, (icon, title, description) in zip(feature_cols, FEATURES):
        with col.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {title}")
            st.caption(description)

    st.space("medium")
    st.markdown("#### 试试这样问")
    question_cols = st.columns(2)
    for index, (key, question, icon) in enumerate(QUICK_QUESTIONS):
        with question_cols[index % 2]:
            st.button(
                question,
                key=f"welcome_{key}",
                icon=icon,
                on_click=queue_prompt,
                args=(question,),
                width="stretch",
            )


def handle_prompt(prompt: str) -> None:
    """流式输出智能客服的回答。"""
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status_slot = st.empty()
        status = status_slot.status("正在连接智能客服…", expanded=True)
        with status:
            status_caption = st.empty()
            status_caption.caption("正在准备模型、知识库与工具，请稍候片刻。")

        reply_time = datetime.now().strftime("%H:%M")

        try:
            agent = get_agent()
            status.update(label="正在理解您的问题…", state="running", expanded=True)
            res_stream = agent.execute_stream(prompt)
            first_event = next(res_stream)
        except StopIteration:
            first_event = None
            res_stream = iter(())
        except Exception as exc:
            from utils.logger_handler import logger

            logger.exception("智能客服执行失败")
            status.update(label="回答失败", state="error", expanded=True)
            st.error("抱歉，智能客服暂时无法回答，请稍后重试。")
            st.caption("请检查网络、DashScope API Key 与向量库配置。")
            with st.expander("查看错误详情", icon=":material/bug_report:"):
                st.code(str(exc))
            remember_failed_answer(reply_time, "抱歉，智能客服暂时无法回答，请稍后重试。")
            return

        status.update(label="正在检索资料并组织回答…", state="running", expanded=True)
        answer_slot = st.empty()
        answer_text = ""

        def apply_event(event: tuple[str, str]) -> None:
            nonlocal answer_text
            kind, text = event
            if kind == "reset":
                # 这一轮模型其实要调用工具，刚流出的思考文字需要撤回
                answer_text = ""
                answer_slot.empty()
                status_caption.caption("正在调用工具并检索知识库，请稍候…")
                status.update(label="正在调用工具并检索资料…", state="running", expanded=True)
                return
            if not answer_text:
                status_caption.caption("资料已就绪，正在生成回答…")
                status.update(label="正在生成回答…", state="running", expanded=False)
            answer_text += text
            answer_slot.markdown(answer_text + "▍")

        try:
            if first_event:
                apply_event(first_event)
            for event in res_stream:
                apply_event(event)
        except Exception as exc:
            from utils.logger_handler import logger

            logger.exception("智能客服流式输出失败")
            status.update(label="回答中断", state="error", expanded=True)
            st.error("回答过程中出现异常，请稍后重试。")
            with st.expander("查看错误详情", icon=":material/bug_report:"):
                st.code(str(exc))
            remember_failed_answer(reply_time, "抱歉，回答过程中出现异常，请稍后重试。")
            return

        answer = (answer_text or "").strip()
        if not answer:
            answer = "抱歉，我暂时没有找到合适的答案，请换个说法再试一次。"
        answer_slot.markdown(answer)

        status_slot.empty()
        reply_time = datetime.now().strftime("%H:%M")
        st.caption(f"{reply_time} · 由 RAG 知识库与 ReAct Agent 生成")

    st.session_state["message"].append(
        {
            "role": "assistant",
            "content": answer,
            "time": reply_time,
            "meta": f"{reply_time} · 由 RAG 知识库与 ReAct Agent 生成",
        }
    )


# ---------------- 会话状态 ----------------
st.session_state.setdefault("message", [])
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("awaiting_reply", None)

# ---------------- 视觉样式 ----------------
st.html(
    """
    <style>
    .zst-hero {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        border-radius: 1.25rem;
        background: linear-gradient(120deg, #312e81 0%, #4f46e5 48%, #0891b2 100%);
        box-shadow: 0 18px 40px -20px rgba(79, 70, 229, 0.65);
        color: #ffffff;
    }
    .zst-hero__icon {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 4.5rem;
        height: 4.5rem;
        border-radius: 1.1rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
    }
    .zst-bot {
        position: relative;
        display: block;
        width: 2.5rem;
        height: 1.9rem;
        border: 2px solid #ffffff;
        border-radius: 0.55rem;
    }
    .zst-bot::before,
    .zst-bot::after {
        content: "";
        position: absolute;
        top: 0.5rem;
        width: 0.2rem;
        height: 0.6rem;
        border-radius: 0.2rem;
        background: rgba(255, 255, 255, 0.85);
    }
    .zst-bot::before { left: -0.42rem; }
    .zst-bot::after { right: -0.42rem; }
    .zst-bot__antenna {
        position: absolute;
        left: 50%;
        top: -0.55rem;
        width: 2px;
        height: 0.55rem;
        border-radius: 1px;
        background: rgba(255, 255, 255, 0.9);
        transform: translateX(-50%);
    }
    .zst-bot__antenna::after {
        content: "";
        position: absolute;
        top: -0.3rem;
        left: 50%;
        width: 0.32rem;
        height: 0.32rem;
        border-radius: 50%;
        background: #ffffff;
        transform: translateX(-50%);
    }
    .zst-bot__eyes {
        position: absolute;
        top: 0.5rem;
        left: 0;
        right: 0;
        display: flex;
        justify-content: center;
        gap: 0.55rem;
    }
    .zst-bot__eyes i {
        width: 0.34rem;
        height: 0.34rem;
        border-radius: 50%;
        background: #ffffff;
    }
    .zst-bot__mouth {
        position: absolute;
        bottom: 0.42rem;
        left: 50%;
        width: 0.75rem;
        height: 2px;
        border-radius: 2px;
        background: rgba(255, 255, 255, 0.9);
        transform: translateX(-50%);
    }
    .zst-hero__kicker {
        display: inline-block;
        margin-bottom: 0.2rem;
        color: #c7d2fe;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
    }
    .zst-hero h1 {
        margin: 0.1rem 0 0.45rem;
        color: #ffffff;
        font-size: clamp(1.45rem, 2.4vw, 2.05rem);
        font-weight: 700;
        line-height: 1.25;
    }
    .zst-hero p {
        margin: 0;
        max-width: 56rem;
        color: rgba(255, 255, 255, 0.86);
        font-size: 0.95rem;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] :is(h1, h2, h3, h4) {
        margin: 0.8rem 0 0.35rem;
        font-size: 1.05rem;
        line-height: 1.5;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] p {
        line-height: 1.75;
    }
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        margin: 0.2rem 0 0.5rem;
    }
    @media (max-width: 640px) {
        .zst-hero { gap: 1rem; padding: 1.25rem; }
        .zst-hero__icon { width: 3.25rem; height: 3.25rem; }
        .zst-hero__icon .zst-bot { transform: scale(0.72); }
    }
    </style>

    <div class="zst-hero">
        <div class="zst-hero__icon" aria-hidden="true">
            <span class="zst-bot">
                <span class="zst-bot__antenna"></span>
                <span class="zst-bot__eyes"><i></i><i></i></span>
                <span class="zst-bot__mouth"></span>
            </span>
        </div>
        <div>
            <span class="zst-hero__kicker">RAG · REACT AGENT</span>
            <h1>智扫通 · 扫地机器人智能客服</h1>
            <p>从选购到保养，从故障排查到个性化使用报告。告诉我您遇到的问题，我来查资料、找答案。</p>
        </div>
    </div>
    """
)

with st.container(horizontal=True, gap="small"):
    st.badge("服务在线", icon=":material/check_circle:", color="green")
    st.badge("RAG 知识库", icon=":material/menu_book:", color="blue")
    st.badge("ReAct 工具调用", icon=":material/psychology:", color="violet")

if toast_message := st.session_state.pop("toast_message", None):
    st.toast(toast_message, icon=":material/add_comment:")

# ---------------- 输入区 ----------------
prompt = st.session_state.pop("pending_prompt", None)
typed_prompt = st.chat_input(
    "描述您遇到的扫地机器人问题…",
    key="chat_input",
    submit_mode="disable",
)
prompt = prompt or typed_prompt

if prompt:
    remember_user_message(prompt)
    st.rerun()

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.markdown("### :material/smart_toy: 智扫通")
    st.caption("扫地机器人 · AI 智能客服")

    with st.container(horizontal=True, gap="small"):
        st.badge("在线", icon=":material/bolt:", color="green")
        st.badge("RAG", icon=":material/database:", color="blue")

    st.space("medium")
    st.markdown("**快捷提问**")
    st.caption("点击即可发送")
    for key, question, icon in QUICK_QUESTIONS:
        st.button(
            question,
            key=f"quick_{key}",
            icon=icon,
            on_click=queue_prompt,
            args=(question,),
            width="stretch",
        )

    st.space("medium")
    with st.container(border=True):
        turns = sum(1 for item in st.session_state["message"] if item["role"] == "user")
        st.metric("对话轮数", turns)
        st.caption("本次会话的提问次数")

    st.button(
        "开始新对话",
        icon=":material/add_comment:",
        type="primary",
        on_click=clear_chat,
        width="stretch",
    )

    st.space("small")
    with st.expander("能力与工具", icon=":material/tune:"):
        st.markdown(
            "- **知识库检索**：选购、保养、故障排查\n"
            "- **环境感知**：结合城市天气给出建议\n"
            "- **使用报告**：读取记录生成月度总结\n"
            "- **多轮思考**：按需调用工具再作答"
        )

    with st.expander("关于智扫通", icon=":material/info:"):
        st.caption("基于本地扫地机器人知识库构建，回答仅供参考。")
        st.markdown("`提问` → `ReAct Agent` → `工具调用` → `RAG 知识库` → `回答`")

    st.space("medium")
    st.caption(f"模型 · {rag_conf['chat_model_name']}")

# ---------------- 主对话区 ----------------
# 欢迎区每轮都渲染，有对话时用 CSS 隐藏：避免生成过程中旧内容以半透明残影留在页面上
with st.container(key="welcome"):
    render_welcome()

if st.session_state["message"]:
    st.html("<style>.st-key-welcome { display: none !important; }</style>")
    render_history()

awaiting_reply = st.session_state.pop("awaiting_reply", None)
if awaiting_reply:
    handle_prompt(awaiting_reply)
