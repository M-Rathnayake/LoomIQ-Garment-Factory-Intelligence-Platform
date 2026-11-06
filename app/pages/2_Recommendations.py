"""Recommendations: OR-Tools suggestions for at-risk lines, which can be applied to the day."""

import streamlit as st

st.set_page_config(page_title="Recommendations | LoomIQ", page_icon="🛠️", layout="wide")

from loomiq.dashboard import charts, session, theme  # noqa: E402
from loomiq.dashboard.live import plan_for_view  # noqa: E402
from loomiq.optimization.recommender import get_settings  # noqa: E402

colors = theme.apply_theme()
session.require_setup()
session.sidebar_controls()
sim = session.get_simulator()
view = session.get_view()
defaults = get_settings()

theme.page_header(
    "Recommendations",
    "Moves operators between lines of the same department and adds overtime, to leave as few pieces "
    "short as possible. Lines that lend operators must still meet their own target.",
    "🛠️",
)

if sim.is_finished:
    st.info("The shift is over, so there is nothing left to change. Start a new day in the sidebar.",
            icon=":material/event_busy:")
    st.stop()

with st.expander("Limits", icon=":material/tune:"):
    col1, col2 = st.columns(2)
    max_moved = col1.slider(
        "Max operators moved into or out of a line", 0, 10, defaults["max_operators_moved_per_line"]
    )
    max_overtime = col2.slider(
        "Max extra overtime per worker (minutes)", 0, 240, defaults["max_overtime_minutes_per_line"],
        step=defaults["overtime_step_minutes"],
    )

plan = plan_for_view(view, max_operators_moved_per_line=max_moved, max_overtime_minutes_per_line=max_overtime)
moves = [a for a in plan.actions if a.action == "move_operators"]
overtime = [a for a in plan.actions if a.action == "add_overtime"]
unresolved = [a for a in plan.actions if a.action == "unresolved"]

if plan.status == "NO_ACTION_NEEDED":
    st.success("Every line is projected to meet its target. No changes needed.", icon=":material/check_circle:")
    st.stop()

worker_hours = sum(a.amount * view.set_index("line_id").at[a.line_id, "no_of_workers"] for a in overtime) / 60
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pieces short now", f"{plan.pieces_short_before:,.0f}", border=True, height="stretch")
col2.metric(
    "With this plan", f"{plan.pieces_short_after:,.0f}", border=True, height="stretch",
    delta=f"{-plan.pieces_recovered:,.0f} pieces", delta_color="inverse",
)
col3.metric("Operators moved", sum(int(a.amount) for a in moves), border=True, height="stretch")
col4.metric("Extra overtime", f"{worker_hours:,.0f} h", border=True, height="stretch", help="Total worker-hours of extra overtime.")

left, right = st.columns([2, 3], gap="medium")
with left:
    st.markdown("#### Suggested actions")
    for action in moves + overtime:
        with st.container(border=True):
            if action.action == "move_operators":
                st.markdown(
                    f":material/swap_horiz: **Move {int(action.amount)} operator(s)**  \n"
                    f"{action.source_line_id} → **{action.line_id}**"
                )
            else:
                st.markdown(f":material/more_time: **{action.amount:.0f} min overtime** per worker  \non **{action.line_id}**")
            st.badge(f"+{action.expected_gain_pieces:,.0f} pieces", icon=":material/trending_up:", color="green")
            st.caption(action.reason)
    for action in unresolved:
        st.warning(f"**{action.line_id}**: {action.reason}", icon=":material/warning:")
with right, st.container(border=True):
    st.plotly_chart(charts.plan_chart(plan.lines, colors), width="stretch", theme=None,
                    config={"displayModeBar": False})

with st.container(border=True):
    st.markdown(
        ":material/info: These are **suggestions for a supervisor to confirm**. They rely on projections, "
        "and the model's early-day warnings are right less than half the time."
    )
    if st.button("Apply this plan to the simulated day", icon=":material/check:", type="primary"):
        applied = sim.apply_plan(plan)
        st.session_state.applied_actions.extend(f"Hour {sim.current_hour}: {note}" for note in applied)
        st.toast(f"Applied {len(applied)} change(s). Move to later hours to see the effect.", icon=":material/check_circle:")
        st.rerun()

if st.session_state.get("applied_actions"):
    with st.container(border=True):
        st.markdown("#### :material/history: Applied today")
        for note in st.session_state.applied_actions:
            st.markdown(f"- {note}")
        st.caption("Move to later hours in the sidebar to see the effect on the lines.")
