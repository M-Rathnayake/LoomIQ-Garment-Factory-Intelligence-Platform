"""Line Monitor: one line's output through the day, its status and SHAP reasons."""

import streamlit as st

st.set_page_config(page_title="Line Monitor | LoomIQ", page_icon="📈", layout="wide")

from loomiq.dashboard import charts, session, theme  # noqa: E402
from loomiq.dashboard.live import STATUS_ON_TRACK  # noqa: E402
from loomiq.models.explain import explain_line  # noqa: E402
from loomiq.models.predict import prepare_features  # noqa: E402

colors = theme.apply_theme()
session.require_setup()
session.sidebar_controls()
sim = session.get_simulator()
view = session.get_view()

theme.page_header("Line Monitor", "Follow one line hour by hour, and see why the model rates its risk the way it does.", "📈")

# Default: the line clicked on Home, otherwise the line in the worst shape
line_ids = view["line_id"].tolist()
labels = dict(zip(view["line_id"], view["status_label"]))
default = st.session_state.get("selected_line")
if default not in line_ids:
    default = view.sort_values("projected_share")["line_id"].iloc[0]
line_id = st.selectbox(
    "Line", line_ids, index=line_ids.index(default),
    format_func=lambda line: f"{line}  ·  {labels[line]}",
)
st.session_state.selected_line = line_id
line = view.set_index("line_id").loc[line_id].copy()
line["line_id"] = line_id

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", line["status_label"], border=True, height="stretch")
col2.metric("Pieces so far", f"{line['pieces_so_far']:,.0f}", border=True, height="stretch",
            delta=f"pace {line['live_productivity']:.2f} vs target {line['targeted_productivity']:.2f}"
            if sim.current_hour else None,
            delta_color="normal" if line["live_productivity"] >= line["targeted_productivity"] else "inverse",
            delta_arrow="off")
col3.metric(
    "Projected pieces", f"{line['projected_pieces']:,.0f}", border=True, height="stretch",
    delta=f"{line['projected_share']:.0%} of target {line['target_pieces']:,.0f}",
    delta_color="normal" if line["status"] == STATUS_ON_TRACK else "inverse", delta_arrow="off",
)
col4.metric(
    "Model risk score", f"{line['risk_probability']:.2f}", border=True, height="stretch",
    help="From the line's setup (SMV, team, workers...). A ranking for 'check these first', not a literal probability.",
)

history = sim.history_frame()
line_history = history[history["line_id"] == line_id]
output_tab, why_tab, details_tab = st.tabs([
    ":material/show_chart: Output", ":material/psychology: Why this risk", ":material/list_alt: Details & events",
])

with output_tab, st.container(border=True):
    st.plotly_chart(charts.line_progress(line_history, line, sim.hours_per_shift, colors),
                    width="stretch", theme=None, config={"displayModeBar": False})

with why_tab:
    drivers = explain_line(session.load_explainer(), prepare_features(session.load_bundle(), view[view["line_id"] == line_id]))
    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        st.plotly_chart(charts.drivers_chart(drivers, colors), width="stretch", theme=None,
                        config={"displayModeBar": False})
    with right, st.container(border=True):
        st.markdown("#### In plain words")
        for driver in drivers[:3]:
            icon = ":material/trending_up:" if driver["direction"] == "raises risk" else ":material/trending_down:"
            verb = "pushes risk **up**" if driver["direction"] == "raises risk" else "pushes risk **down**"
            label = driver["label"][0].upper() + driver["label"][1:]
            st.markdown(f"{icon} **{label}** ({driver['value']}) {verb}")
        st.caption(
            "SHAP explains the model's risk score from the line's setup. Live output "
            "(the Output tab) can tell a different story as the day goes on."
        )

with details_tab:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("#### Line details")
        st.markdown(
            f"- **Department:** {str(line['department']).title()}\n"
            f"- **Workers:** {line['no_of_workers']:.1f}\n"
            f"- **SMV:** {line['smv']:.2f} min per garment\n"
            f"- **Target productivity:** {line['targeted_productivity']:.2f}\n"
            f"- **Live pace so far:** {line['live_productivity']:.2f}"
        )
    with right, st.container(border=True):
        st.markdown("#### Events today")
        if line["events"]:
            for event in line["events"]:
                st.markdown(f"- {event}")
        else:
            st.caption("No breakdowns, style changes or plan changes on this line yet.")

if line["status"] != STATUS_ON_TRACK:
    st.page_link("pages/2_Recommendations.py", label="See recommended fixes", icon=":material/build:")
