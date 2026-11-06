"""LoomIQ dashboard entry point: overview of all production lines.

Run with:
    streamlit run app/Home.py
"""

import streamlit as st

st.set_page_config(page_title="LoomIQ", page_icon="🧵", layout="wide")

from loomiq.dashboard import charts, session, theme  # noqa: E402
from loomiq.dashboard.live import STATUS_ON_TRACK, factory_summary  # noqa: E402

colors = theme.apply_theme()
session.require_setup()
session.sidebar_controls()
sim = session.get_simulator()
view = session.get_view()
summary = factory_summary(view)

clock = f"{8 + sim.current_hour}:00" if not sim.is_finished else "end of shift"
theme.page_header(
    "Factory Overview",
    f"{sim.date:%A %d %B %Y} · {clock} · hour {sim.current_hour} of {sim.hours_per_shift}",
    "🧵",
)

# --- Headline numbers ---
target_so_far = summary["target_pieces"] * sim.current_hour / sim.hours_per_shift
projected_share = summary["projected_pieces"] / summary["target_pieces"]
dhu = st.session_state.get("dhu")
hourly = session.hourly_output()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Lines on track", f"{summary['on_track']} / {summary['lines']}", border=True, height="stretch")
col2.metric(
    "At risk or behind", summary["at_risk"], border=True, height="stretch",
    delta="needs attention" if summary["at_risk"] else "all clear",
    delta_color="inverse" if summary["at_risk"] else "normal", delta_arrow="off",
)
col3.metric(
    "Pieces so far", f"{summary['pieces_so_far']:,.0f}", border=True, height="stretch",
    delta=f"{summary['pieces_so_far'] - target_so_far:+,.0f} vs pace" if sim.current_hour else None,
    chart_data=hourly if len(hourly) > 2 else None, chart_type="area",
)
col4.metric(
    "Projected vs target", f"{projected_share:.0%}", border=True, height="stretch",
    help="Starts from the model's prediction and moves towards the real pace as the hours pass.",
)
col5.metric("Today's DHU", f"{dhu:.1f}" if dhu is not None else "—", border=True, height="stretch",
            help="Defects per 100 units, from the Defect Scanner. Lower is better.")

if sim.current_hour == 0:
    st.info(
        "**8:00, start of shift.** Projections come from the productivity model only. "
        "Press **Next hour** in the sidebar to watch real output come in.",
        icon=":material/wb_twilight:",
    )

# --- Filter ---
department = st.segmented_control(
    "Department", ["All lines", "Sewing", "Finishing"], default="All lines", key="home_department",
    label_visibility="collapsed",
)
shown = view if department in (None, "All lines") else view[view["department"] == department.lower()]

# --- All lines ---
left, right = st.columns([3, 2], gap="medium")
with left, st.container(border=True):
    # A fresh key after each click clears the selection, so returning here doesn't jump again
    clicks = st.session_state.setdefault("overview_clicks", 0)
    event = st.plotly_chart(
        charts.line_overview(shown, colors), width="stretch", theme=None,
        on_select="rerun", selection_mode="points", key=f"overview_chart_{clicks}",
        config={"displayModeBar": False},
    )
    st.caption(":material/touch_app: Click a bar to open that line in the Line Monitor.")
    points = event.selection.points if event and event.selection else []
    if points:
        st.session_state.selected_line = points[0]["y"]
        st.session_state.overview_clicks += 1
        st.switch_page("pages/1_Line_Monitor.py")

with right:
    with st.container(border=True):
        st.markdown("#### :material/priority_high: Needs attention")
        attention = shown[shown["status"] != STATUS_ON_TRACK].sort_values("projected_share")
        if attention.empty:
            st.success("All lines are on track.", icon=":material/check_circle:")
        for row in attention.itertuples():
            st.markdown(
                f"{row.status_label} &nbsp;**{row.line_id}**  \n"
                f"<span class='loomiq-muted'>projected {row.projected_pieces:,.0f} of "
                f"{row.target_pieces:,.0f} pieces ({row.projected_share:.0%})</span>",
                unsafe_allow_html=True,
            )
        if not attention.empty:
            st.page_link("pages/2_Recommendations.py", label="See recommended fixes", icon=":material/build:")
            st.page_link("pages/1_Line_Monitor.py", label="Investigate a line", icon=":material/monitoring:")

    events = [(line.line_id, event_text) for line in sim.lines for event_text in line.events]
    with st.container(border=True):
        st.markdown("#### :material/notifications: Latest events")
        if not events:
            st.caption("Quiet so far: no breakdowns, style changes or plan changes.")
        for line_id, event_text in events[-6:][::-1]:
            st.markdown(f"**{line_id}** <span class='loomiq-muted'>{event_text}</span>", unsafe_allow_html=True)

# --- Table ---
with st.container(border=True):
    st.markdown("#### :material/table_rows: All lines")
    table = shown[[
        "line_id", "status_label", "no_of_workers", "pieces_so_far",
        "projected_pieces", "target_pieces", "projected_share", "risk_probability",
    ]].assign(dhu=shown["line_id"].map(st.session_state.get("dhu_by_line", {})))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "line_id": "Line",
            "status_label": "Status",
            "no_of_workers": st.column_config.NumberColumn("Workers", format="%.1f"),
            "pieces_so_far": st.column_config.NumberColumn("Pieces so far", format="%,.0f"),
            "projected_pieces": st.column_config.NumberColumn("Projected", format="%,.0f"),
            "target_pieces": st.column_config.NumberColumn("Target", format="%,.0f"),
            "projected_share": st.column_config.ProgressColumn(
                "Projected % of target", min_value=0, max_value=1.3, format="percent",
            ),
            "risk_probability": st.column_config.ProgressColumn(
                "Model risk score", min_value=0, max_value=1, format="%.2f",
                help="From the productivity model, based on the line's setup. A ranking, not a literal probability.",
            ),
            "dhu": st.column_config.NumberColumn("DHU", format="%.1f", help="Defects per 100 units scanned today."),
        },
    )
    if dhu is None:
        st.page_link("pages/3_Defect_Scanner.py", label="Scan fabric for defects", icon=":material/search:")
