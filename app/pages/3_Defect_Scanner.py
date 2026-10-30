"""Defect Scanner: upload a fabric image and see detected defects and DHU."""

import streamlit as st

st.set_page_config(page_title="Defect Scanner | LoomIQ", page_icon="🔍", layout="wide")

st.title("🔍 Defect Scanner")
st.write(
    "Upload a fabric image. The defect model finds holes, stains and broken "
    "threads, draws boxes around them and updates the DHU for the batch."
)

uploaded = st.file_uploader("Upload a fabric image", type=["png", "jpg", "jpeg", "bmp"])

# TODO: Load the model once with st.cache_resource(loomiq.vision.detect.load_defect_model).
# TODO: Run detect_defects on the upload and draw boxes on the image.
# TODO: Track defects and units in st.session_state and show compute_dhu(...).
if uploaded is not None:
    st.image(uploaded, caption="Uploaded image")
st.info("Coming soon: defect detection with bounding boxes and a running DHU score.")
