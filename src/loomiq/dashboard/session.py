"""Streamlit helpers shared by every dashboard page.

* Loads the model and SHAP explainer once (cached).
* Keeps one simulated factory day per browser session, shared across pages.
* Draws the sidebar controls for moving through the day.
"""

from typing import Any

import pandas as pd
import streamlit as st

from loomiq.dashboard.live import build_live_view
from loomiq.simulation.factory_simulator import FactorySimulator


def require_setup() -> None:
    """Stop the page with clear instructions if the data or model is missing."""
    from loomiq.data.clean import get_clean_data_path
    from loomiq.models.train_productivity import get_model_path

    missing = [p.name for p in (get_clean_data_path(), get_model_path()) if not p.exists()]
    if missing:
        st.error(f"Missing: {', '.join(missing)}. Prepare the data and train the model first:")
        st.code("python scripts/train_all.py --skip-vision", language="bash")
        st.caption("Download the UCI dataset into data/raw/ first (see docs/DATASETS.md).")
        st.stop()


@st.cache_resource(show_spinner="Loading the productivity model...")
def load_bundle() -> dict[str, Any]:
    from loomiq.models.predict import load_model

    return load_model()


@st.cache_resource(show_spinner="Preparing SHAP explanations...")
def load_explainer() -> Any:
    """SHAP explainer with the model's own training weeks as the reference."""
    from loomiq.config import load_config
    from loomiq.data.clean import load_clean_productivity
    from loomiq.models.explain import build_explainer
    from loomiq.models.predict import prepare_features
    from loomiq.models.train_productivity import time_based_split

    bundle = load_bundle()
    train_df, _ = time_based_split(load_clean_productivity(), load_config()["productivity_model"]["test_size"])
    return build_explainer(bundle["model"], prepare_features(bundle, train_df))


def get_simulator() -> FactorySimulator:
    """The simulated day for this browser session (created on first use)."""
    if "simulator" not in st.session_state:
        st.session_state.day_number = 1
        st.session_state.simulator = FactorySimulator(seed=st.session_state.day_number)
        st.session_state.applied_actions = []
    return st.session_state.simulator


def get_view() -> pd.DataFrame:
    """Live view of every line (model risk, live pace, projection, status)."""
    return build_live_view(get_simulator(), load_bundle())


def hourly_output() -> list[float]:
    """Total pieces made in each hour so far (for sparklines)."""
    history = get_simulator().history_frame()
    return history.groupby("hour")["pieces_this_hour"].sum().round(0).tolist()


def sidebar_controls() -> None:
    """Day info, buttons to move through the shift, and the light/dark toggle."""
    from loomiq.dashboard.theme import mode_toggle

    sim = get_simulator()
    with st.sidebar:
        st.markdown("##### :material/schedule: Factory clock")
        st.badge(f"{sim.date:%a %d %b %Y}", icon=":material/calendar_today:", color="blue")
        st.caption(f"{len(sim.lines)} lines · a real day from the test weeks")
        clock = f"{8 + sim.current_hour}:00" if not sim.is_finished else "Shift over"
        st.progress(sim.current_hour / sim.hours_per_shift, text=f"**{clock}** · hour {sim.current_hour} of {sim.hours_per_shift}")

        if st.button("Next hour", icon=":material/skip_next:", type="primary", disabled=sim.is_finished, width="stretch"):
            sim.step()
            st.rerun()
        end_col, new_col = st.columns(2)
        if end_col.button("End shift", icon=":material/fast_forward:", disabled=sim.is_finished, width="stretch"):
            for _ in sim.stream():
                pass
            st.rerun()
        if new_col.button("New day", icon=":material/restart_alt:", width="stretch"):
            st.session_state.day_number += 1
            sim.reset(seed=st.session_state.day_number)
            st.session_state.applied_actions = []
            # Quality is tracked per day too
            st.session_state.inspections = []
            st.session_state.pop("dhu", None)
            st.session_state.pop("dhu_by_line", None)
            st.rerun()

        if sim.is_finished:
            st.success("Shift finished. Start a new day to continue.", icon=":material/check_circle:")
        st.divider()
        mode_toggle()
        st.caption(
            "Each simulated day replays a real day the model never saw, revealed hour by hour "
            "with random noise and machine breakdowns."
        )
