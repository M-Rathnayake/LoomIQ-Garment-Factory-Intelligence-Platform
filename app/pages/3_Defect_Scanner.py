"""Defect Scanner: upload fabric images, see defects, and track DHU per line."""

import streamlit as st

st.set_page_config(page_title="Defect Scanner | LoomIQ", page_icon="🔍", layout="wide")

from loomiq.dashboard import session, theme  # noqa: E402
from loomiq.vision.detect import (  # noqa: E402
    compute_dhu, defect_closeups, detect_defects, draw_defects, get_model_path, to_gray_array,
)
from loomiq.vision.prepare_dataset import get_settings  # noqa: E402

theme.apply_theme()
session.require_setup()
session.sidebar_controls()
sim = session.get_simulator()

theme.page_header(
    "Defect Scanner",
    "Upload fabric images from a line. The model boxes each defect (fuzzyballs, broken yarn, "
    "creases...) and keeps a running DHU: defects per 100 units inspected.",
    "🔍",
)

if not get_model_path().exists():
    st.info(
        "**No defect model yet.** Download the AITEX fabric images into `data/images/aitex/` "
        "(see docs/DATASETS.md), then train the detector:",
        icon=":material/info:",
    )
    st.code("python -m loomiq.vision.train_defect", language="bash")
    st.stop()


@st.cache_resource(show_spinner="Loading the defect model...")
def load_model():
    from loomiq.vision.detect import load_defect_model

    return load_defect_model()


st.session_state.setdefault("inspections", [])
line_ids = [line.line_id for line in sim.lines]
left, right = st.columns([2, 1])
line_id = left.selectbox("Line these panels come from", line_ids)
confidence = right.slider(
    "Confidence threshold", 0.05, 0.9, float(get_settings()["confidence_threshold"]), 0.05,
    help="Lower finds more defects but raises more false alarms.",
)
uploads = st.file_uploader(
    "Fabric images (each image = one unit inspected)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploads and st.button("Scan images", icon=":material/search:", type="primary"):
    model = load_model()
    for upload in uploads:
        pixels = to_gray_array(upload)  # read the file once
        defects = detect_defects(model, pixels, confidence)
        st.session_state.inspections.append({"line_id": line_id, "file": upload.name, "defects": len(defects)})
        with st.container(border=True):
            if defects:
                st.badge(f"{len(defects)} defect(s)", icon=":material/error:", color="red")
            else:
                st.badge("No defects found", icon=":material/check_circle:", color="green")
            st.markdown(f"**{upload.name}**")
            st.image(draw_defects(pixels, defects), width="stretch")
            if defects:
                st.caption(":material/zoom_in: Close-ups (3× zoom)")
                with st.container(horizontal=True, gap="small"):
                    for crop, defect in zip(defect_closeups(pixels, defects), defects):
                        st.image(crop, caption=f"confidence {defect.confidence:.2f}", width=200)

# --- Running DHU ---
inspections = st.session_state.inspections
if inspections:
    total_defects = sum(i["defects"] for i in inspections)
    st.session_state.dhu = compute_dhu(total_defects, len(inspections))

    st.markdown("#### :material/fact_check: Today's quality")
    col1, col2, col3 = st.columns(3)
    col1.metric("Units inspected", len(inspections), border=True)
    col2.metric("Defects found", total_defects, border=True)
    col3.metric("DHU (defects per 100 units)", f"{st.session_state.dhu:.1f}", border=True)

    by_line = {}
    for inspection in inspections:
        stats = by_line.setdefault(inspection["line_id"], {"Line": inspection["line_id"], "Units": 0, "Defects": 0})
        stats["Units"] += 1
        stats["Defects"] += inspection["defects"]
    table = [{**s, "DHU": compute_dhu(s["Defects"], s["Units"])} for s in by_line.values()]
    st.session_state.dhu_by_line = {row["Line"]: row["DHU"] for row in table}
    st.dataframe(table, hide_index=True, column_config={"DHU": st.column_config.NumberColumn(format="%.1f")})

    if st.button("Reset today's inspections", icon=":material/delete_sweep:"):
        st.session_state.inspections = []
        st.session_state.pop("dhu", None)
        st.session_state.pop("dhu_by_line", None)
        st.rerun()
else:
    st.caption("No units inspected yet today.")
