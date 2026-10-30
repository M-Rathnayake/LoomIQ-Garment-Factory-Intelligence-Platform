"""Tests for loomiq.optimization.recommender.

Hand-made lines with a known ``expected_productivity`` are used, so the tests
need no trained model and check the factory rules directly.
"""

import pandas as pd
import pytest

from loomiq.optimization import recommender as rec


def make_lines(**overrides) -> pd.DataFrame:
    """Two finishing lines (one behind, one ahead) and one sewing line (behind)."""
    lines = pd.DataFrame(
        {
            "department": ["finishing", "finishing", "sewing"],
            "team": [1, 2, 3],
            "targeted_productivity": [0.75, 0.75, 0.80],
            "expected_productivity": [0.60, 0.95, 0.70],
            "smv": [4.0, 4.0, 20.0],
            "no_of_workers": [10.0, 10.0, 50.0],
            "over_time": [0, 0, 6000],
        }
    )
    for column, values in overrides.items():
        lines[column] = values
    return lines


@pytest.fixture
def settings() -> dict:
    return rec.get_settings()


def test_piece_math(settings: dict) -> None:
    lines = rec.estimate_shortfall(make_lines(), settings)
    # 10 workers x 480 min = 4,800 min; 0.75 x 4,800 / 4.0 SMV = 900 target pieces
    assert lines.loc[0, "target_pieces"] == pytest.approx(900)
    assert lines.loc[0, "expected_pieces"] == pytest.approx(720)
    assert lines.loc[0, "shortfall_pieces"] == pytest.approx(180)
    assert lines.loc[1, "surplus_pieces"] == pytest.approx(240)
    assert lines.loc[0, "line_id"] == "Finishing 1"


def test_estimate_from_model_prediction(settings: dict) -> None:
    lines = make_lines().drop(columns="expected_productivity")
    lines["met_target_pred"] = [0, 1, 0]
    lines = rec.estimate_shortfall(lines, settings)
    gaps = settings["typical_gap"]
    assert lines.loc[0, "expected_productivity"] == pytest.approx(0.75 + gaps["finishing"]["missed"])
    assert lines.loc[1, "expected_productivity"] == pytest.approx(0.75 + gaps["finishing"]["met"])


def test_no_action_when_all_lines_on_track(settings: dict) -> None:
    plan = rec.recommend_actions(make_lines(expected_productivity=[0.80, 0.80, 0.85]), settings)
    assert plan.status == "NO_ACTION_NEEDED"
    assert plan.actions == []


def test_plan_reduces_shortfall_and_follows_rules(settings: dict) -> None:
    plan = rec.recommend_actions(make_lines(), settings)
    table = plan.lines

    assert plan.pieces_short_after < plan.pieces_short_before
    # Operators are only moved, never created or lost
    assert table["operators_in"].sum() == table["operators_out"].sum()
    # Limits are respected
    assert table["extra_overtime_minutes"].max() <= settings["max_overtime_minutes_per_line"]
    assert table[["operators_in", "operators_out"]].max().max() <= settings["max_operators_moved_per_line"]
    # A line that lends operators still meets its own target
    lenders = table[table["operators_out"] > 0]
    assert (lenders["expected_pieces_after"] >= lenders["target_pieces"] - 1e-6).all()


def test_no_moves_between_departments(settings: dict) -> None:
    plan = rec.recommend_actions(make_lines(), settings)
    departments = plan.lines.set_index("line_id")["department"]
    for action in plan.actions:
        if action.action == "move_operators":
            assert departments[action.source_line_id] == departments[action.line_id]
    # The sewing line has no sewing lender, so it can only get overtime
    sewing_actions = {a.action for a in plan.actions if a.line_id == "Sewing 3"}
    assert "move_operators" not in sewing_actions


def test_unresolved_when_no_options(settings: dict) -> None:
    no_help = dict(settings, max_overtime_minutes_per_line=0, max_operators_moved_per_line=0)
    plan = rec.recommend_actions(make_lines(), no_help)
    unresolved = {a.line_id for a in plan.actions if a.action == "unresolved"}
    assert unresolved == {"Finishing 1", "Sewing 3"}
    assert plan.pieces_short_after == pytest.approx(plan.pieces_short_before)


def test_missing_inputs_raise_error(settings: dict) -> None:
    with pytest.raises(ValueError):
        rec.estimate_shortfall(make_lines().drop(columns="expected_productivity"), settings)


def test_plan_to_frame(settings: dict) -> None:
    frame = rec.plan_to_frame(rec.recommend_actions(make_lines(), settings))
    assert {"line_id", "action", "amount", "expected_gain_pieces", "reason"} <= set(frame.columns)
