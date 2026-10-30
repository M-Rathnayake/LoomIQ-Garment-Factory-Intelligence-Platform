"""Recommendations: optimization-based fixes for at-risk lines."""

import streamlit as st

st.set_page_config(page_title="Recommendations | LoomIQ", page_icon="🛠️", layout="wide")

st.title("🛠️ Recommendations")
st.write(
    "For lines at risk of missing target, LoomIQ suggests the best fix, such as "
    "moving operators from a line that is ahead or adding overtime, using OR-Tools."
)

# TODO: Sliders for max overtime and max operators moved (defaults from config.yaml).
# TODO: Call loomiq.optimization.recommender.recommend_actions and show a table.
# TODO: Show expected pieces gained per recommendation.
st.slider("Max overtime per line (minutes)", 0, 240, 120, disabled=True)
st.info("Coming soon: suggested operator moves and overtime for each at-risk line.")
