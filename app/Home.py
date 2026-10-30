"""LoomIQ dashboard entry point: overview of all production lines.

Run with:
    streamlit run app/Home.py
"""

import streamlit as st

st.set_page_config(page_title="LoomIQ", page_icon="🧵", layout="wide")

st.title("🧵 LoomIQ: Garment Factory Intelligence")
st.write(
    "A live overview of every production line: who is on track, who is at risk, "
    "and what to do about it. Use the pages in the sidebar to drill down."
)

# TODO: Read the latest simulator snapshot (simulation.output_file in config.yaml).
# TODO: Run loomiq.models.predict.predict_line for every line.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Lines running", "—")
col2.metric("Lines at risk", "—")
col3.metric("Avg. line efficiency", "—")
col4.metric("Today's DHU", "—")

st.subheader("Line overview")
# TODO: Show a Plotly bar chart of output vs target per line, coloured by risk.
st.info("Coming soon: live line status table and output-vs-target chart.")

st.subheader("Modules")
st.markdown(
    "- **Line Monitor**: live status and predictions per line\n"
    "- **Recommendations**: suggested operator moves and overtime\n"
    "- **Defect Scanner**: upload a fabric image and find defects\n"
    "- **AI Assistant**: ask questions like *\"Why is Line 4 behind?\"*"
)
