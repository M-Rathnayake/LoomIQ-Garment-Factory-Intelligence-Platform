"""Line Monitor: live status, target-hit prediction and SHAP reasons per line."""

import streamlit as st

st.set_page_config(page_title="Line Monitor | LoomIQ", page_icon="📈", layout="wide")

st.title("📈 Line Monitor")
st.write(
    "Watch each line hour by hour. LoomIQ predicts whether the line will hit "
    "its daily target and shows the main reasons behind the prediction."
)

# TODO: Auto-refresh every few seconds and read the latest simulator rows.
# TODO: st.selectbox to choose a line; show hourly output vs target (Plotly line chart).
# TODO: Show risk probability and top SHAP drivers from loomiq.models.explain.explain_line.
st.selectbox("Select a line", ["Line 1", "Line 2", "Line 3"], disabled=True)
st.info("Coming soon: live hourly output, risk score and SHAP explanation for each line.")
