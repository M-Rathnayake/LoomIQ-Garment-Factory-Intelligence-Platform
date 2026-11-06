"""AI Assistant: ask questions about the factory in plain language."""

import anthropic
import streamlit as st

st.set_page_config(page_title="AI Assistant | LoomIQ", page_icon="🤖", layout="wide")

from loomiq.assistant.llm_assistant import FactoryAssistant, build_context, offline_answer  # noqa: E402
from loomiq.dashboard import session, theme  # noqa: E402
from loomiq.dashboard.live import plan_for_view  # noqa: E402
from loomiq.models.explain import explain_line  # noqa: E402
from loomiq.models.predict import prepare_features  # noqa: E402

theme.apply_theme()
session.require_setup()
session.sidebar_controls()
sim = session.get_simulator()
view = session.get_view()

theme.page_header(
    "AI Assistant",
    "Ask about the factory in plain English. Answers are grounded in the live line status, "
    "the model's SHAP reasons and the recommended plan, not guesses.",
    "🤖",
)

if "assistant" not in st.session_state:
    st.session_state.assistant = FactoryAssistant()
    st.session_state.chat = []
assistant: FactoryAssistant = st.session_state.assistant

if assistant.is_online:
    st.badge(f"Connected to Claude ({assistant.model})", icon=":material/bolt:", color="green")
else:
    st.info(
        "**Offline mode:** answers come from simple rules over the same factory data. "
        "Add `ANTHROPIC_API_KEY=...` to `.env` and restart the app for full answers.",
        icon=":material/cloud_off:",
    )


def factory_briefing() -> tuple[str, dict, object]:
    """Current state of the factory: briefing text, SHAP drivers per line, and the plan."""
    explainer = session.load_explainer()
    bundle = session.load_bundle()
    features = prepare_features(bundle, view)
    drivers = {
        line_id: explain_line(explainer, features.iloc[[i]], top_n=3)
        for i, line_id in enumerate(view["line_id"])
    }
    plan = None if sim.is_finished else plan_for_view(view)
    context = build_context(
        view, drivers, plan, hour=sim.current_hour, hours_per_shift=sim.hours_per_shift,
        date=sim.date, dhu=st.session_state.get("dhu"), dhu_by_line=st.session_state.get("dhu_by_line"),
    )
    return context, drivers, plan


def answer(question: str) -> str:
    context, drivers, plan = factory_briefing()
    if assistant.is_online:
        try:
            return assistant.ask(question, context, st.session_state.chat)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
            # Auth, rate-limit, server or network problems: fall back instead of crashing
            st.warning(f"Claude is unavailable ({type(error).__name__}), so this answer uses offline mode.")
    return offline_answer(question, view, drivers, plan)


# --- Suggested questions ---
worst = view.sort_values("projected_share").iloc[0]["line_id"]
suggestions = [
    "Which lines need attention right now?",
    f"Why is {worst} behind?",
    "What does the recommended plan do?",
]
columns = st.columns(len(suggestions))
pressed = [col.button(q, icon=":material/chat_bubble:", width="stretch") for col, q in zip(columns, suggestions)]
clicked = next((q for q, was_pressed in zip(suggestions, pressed) if was_pressed), None)

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about your lines...") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Reading the factory floor..."):
            reply = answer(question)
        st.markdown(reply)
    st.session_state.chat += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]

if st.session_state.chat and st.button("Clear chat", icon=":material/delete:"):
    st.session_state.chat = []
    st.rerun()

with st.expander("What the assistant sees (factory briefing)", icon=":material/visibility:"):
    st.code(factory_briefing()[0], language="text")
