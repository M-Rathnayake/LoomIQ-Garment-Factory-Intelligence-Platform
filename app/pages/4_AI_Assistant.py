"""AI Assistant: chat with the factory assistant about line performance."""

import streamlit as st

st.set_page_config(page_title="AI Assistant | LoomIQ", page_icon="🤖", layout="wide")

st.title("🤖 AI Assistant")
st.write(
    "Ask questions in plain English, like *\"Why is Line 4 behind?\"*. "
    "The assistant answers using live predictions, SHAP explanations and "
    "recommendations, not guesses."
)

# TODO: Build context with loomiq.assistant.llm_assistant.build_context.
# TODO: Keep chat history in st.session_state and call FactoryAssistant.ask.
# TODO: Show a friendly warning if ANTHROPIC_API_KEY is not set in .env.
st.info("Coming soon: chat with the factory assistant.")
st.chat_input("Ask about your lines...", disabled=True)
