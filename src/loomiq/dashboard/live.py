"""Combine the simulator, the productivity model and the recommender for the dashboard.

How each line's end-of-day output is projected:
    * At the start of the day nothing has been produced yet, so the projection
      uses the **model's prediction**: target plus the typical gap for lines
      that miss or meet (same assumption as the recommender).
    * As hours pass, the projection moves towards the line's **real pace so far**.
      Weight on the live pace = share of the shift already worked.

This module has no Streamlit code, so it can be tested on its own.
"""

from typing import Any

import pandas as pd

from loomiq.models.predict import predict_line
from loomiq.optimization.recommender import RecommendationPlan, get_settings, recommend_actions

STATUS_ON_TRACK = "On track"
STATUS_AT_RISK = "At risk"
STATUS_BEHIND = "Behind"
STATUS_ICONS = {STATUS_ON_TRACK: "✅", STATUS_AT_RISK: "⚠️", STATUS_BEHIND: "🔴"}

# Projected share of target below which a line counts as "Behind" instead of "At risk"
BEHIND_THRESHOLD = 0.90


def line_status(projected_share: float, tolerance: float = 0.02) -> str:
    """Turn projected output / target into a status label.

    Within ``tolerance`` of target (normal wobble) still counts as on track.
    """
    if projected_share >= 1 - tolerance:
        return STATUS_ON_TRACK
    if projected_share >= BEHIND_THRESHOLD:
        return STATUS_AT_RISK
    return STATUS_BEHIND


def build_live_view(simulator: Any, bundle: dict[str, Any], settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return one row per line with model risk, live pace, projection and status.

    Args:
        simulator: A :class:`~loomiq.simulation.factory_simulator.FactorySimulator`.
        bundle: Loaded productivity model bundle.
        settings: Optimization settings (for the typical gaps); defaults to config.

    Returns:
        The simulator snapshot plus ``risk_probability``, ``met_target_pred``,
        ``live_productivity``, ``expected_productivity``, ``projected_pieces``,
        ``projected_share``, ``status`` and ``status_label``.
    """
    settings = settings or get_settings()
    view = predict_line(bundle, simulator.snapshot())

    gaps = settings["typical_gap"]
    outcome = view["met_target_pred"].map({0: "missed", 1: "met"})
    prior = view["targeted_productivity"] + [gaps[d][o] for d, o in zip(view["department"], outcome)]

    worked = view["available_minutes_so_far"]
    live = (view["pieces_so_far"] * view["smv"] / worked.where(worked > 0)).fillna(prior)
    weight = simulator.current_hour / simulator.hours_per_shift

    view["prior_productivity"] = prior
    view["live_productivity"] = live
    view["expected_productivity"] = weight * live + (1 - weight) * prior
    view["projected_pieces"] = (
        view["pieces_so_far"] + view["expected_productivity"] * view["remaining_minutes"] / view["smv"]
    )
    view["projected_share"] = view["projected_pieces"] / view["target_pieces"]
    tolerance = settings.get("on_track_tolerance", 0)
    view["status"] = view["projected_share"].map(lambda share: line_status(share, tolerance))
    view["status_label"] = view["status"].map(lambda s: f"{STATUS_ICONS[s]} {s}")
    return view


def plan_for_view(view: pd.DataFrame, **overrides: Any) -> RecommendationPlan:
    """Run the recommender on the live view.

    Uses the live projection as expected output, and only the **remaining**
    working time when valuing a moved operator.

    Args:
        view: Output of :func:`build_live_view`.
        **overrides: Optimization settings to change, e.g. ``max_operators_moved_per_line=2``.
    """
    lines = view.assign(expected_pieces=view["projected_pieces"])
    return recommend_actions(lines, get_settings(overrides))


def factory_summary(view: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the home page."""
    return {
        "lines": len(view),
        "on_track": int((view["status"] == STATUS_ON_TRACK).sum()),
        "at_risk": int((view["status"] != STATUS_ON_TRACK).sum()),
        "pieces_so_far": float(view["pieces_so_far"].sum()),
        "target_pieces": float(view["target_pieces"].sum()),
        "projected_pieces": float(view["projected_pieces"].sum()),
    }
