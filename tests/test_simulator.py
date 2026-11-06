"""Tests for the factory simulator and the dashboard's live view.

A small hand-made "real day" is used, so the tests need no downloads or trained model.
"""

import pandas as pd
import pytest

from loomiq.dashboard import live
from loomiq.optimization.recommender import Recommendation, RecommendationPlan
from loomiq.simulation.factory_simulator import FactorySimulator


@pytest.fixture
def one_day() -> pd.DataFrame:
    """Four lines on one day: two sewing, two finishing, one of each running slow."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-03-04"] * 4),
            "quarter": ["Quarter1"] * 4,
            "department": pd.Categorical(["sewing", "sewing", "finishing", "finishing"], categories=["sewing", "finishing"]),
            "day": ["Wednesday"] * 4,
            "team": [1, 2, 3, 4],
            "targeted_productivity": [0.80, 0.80, 0.75, 0.75],
            "smv": [20.0, 20.0, 4.0, 4.0],
            "wip": [1000.0, 1000.0, 0.0, 0.0],
            "over_time": [6000, 6000, 0, 0],
            "incentive": [0, 0, 0, 0],
            "idle_time": [0.0] * 4,
            "idle_men": [0] * 4,
            "no_of_style_change": [0] * 4,
            "no_of_workers": [50.0, 50.0, 10.0, 10.0],
            "actual_productivity": [0.80, 0.60, 0.95, 0.55],
            "wip_missing": [False, False, True, True],
        }
    )


@pytest.fixture
def sim(one_day: pd.DataFrame) -> FactorySimulator:
    return FactorySimulator(n_lines=4, seed=0, data=one_day, date="2015-03-04")


def test_replays_the_real_day(sim: FactorySimulator) -> None:
    assert sim.date == pd.Timestamp("2015-03-04")
    assert [line.line_id for line in sim.lines] == ["Sewing 1", "Sewing 2", "Finishing 3", "Finishing 4"]
    # Target pieces = target x (workers x 480 + overtime) / SMV
    assert sim.lines[0].target_pieces == pytest.approx(0.80 * (50 * 480 + 6000) / 20)


def test_same_seed_same_day(one_day: pd.DataFrame) -> None:
    first = FactorySimulator(n_lines=4, seed=3, data=one_day, date="2015-03-04")
    second = FactorySimulator(n_lines=4, seed=3, data=one_day, date="2015-03-04")
    for _ in range(3):
        first.step()
        second.step()
    pd.testing.assert_frame_equal(first.history_frame(), second.history_frame())


def test_full_day_matches_hidden_pace(sim: FactorySimulator) -> None:
    sim.settings = dict(sim.settings, hourly_noise=0.0, breakdown_probability=0.0, style_change_probability=0.0)
    hours = list(sim.stream())
    assert len(hours) == sim.hours_per_shift and sim.is_finished
    snap = sim.snapshot()
    pace = snap["pieces_so_far"] * snap["smv"] / snap["available_minutes_so_far"]
    assert pace.round(3).tolist() == [0.80, 0.60, 0.95, 0.55]
    with pytest.raises(RuntimeError):
        sim.step()


def test_history_has_columns_before_first_hour(sim: FactorySimulator) -> None:
    history = sim.history_frame()
    assert history.empty
    assert {"hour", "line_id", "pieces_so_far"} <= set(history.columns)


def test_apply_plan_moves_people_and_adds_overtime(sim: FactorySimulator) -> None:
    sim.step()
    plan = RecommendationPlan(
        actions=[
            Recommendation("Finishing 4", "move_operators", 2, source_line_id="Finishing 3"),
            Recommendation("Sewing 2", "add_overtime", 60),
            Recommendation("Sewing 2", "unresolved", 10),
        ],
        lines=pd.DataFrame(),
        status="OPTIMAL",
    )
    total_before = sum(line.no_of_workers for line in sim.lines)
    applied = sim.apply_plan(plan)

    by_id = {line.line_id: line for line in sim.lines}
    assert len(applied) == 2  # "unresolved" is advice, not a change
    assert by_id["Finishing 3"].no_of_workers == 8 and by_id["Finishing 4"].no_of_workers == 12
    assert sum(line.no_of_workers for line in sim.lines) == total_before
    assert by_id["Sewing 2"].extra_overtime_left == 60 * 50
    assert any("Moved 2" in event for event in by_id["Finishing 4"].events)


def test_line_status_with_tolerance() -> None:
    assert live.line_status(1.05) == live.STATUS_ON_TRACK
    assert live.line_status(0.99, tolerance=0.02) == live.STATUS_ON_TRACK
    assert live.line_status(0.95, tolerance=0.02) == live.STATUS_AT_RISK
    assert live.line_status(0.80) == live.STATUS_BEHIND


def test_live_view_moves_from_model_to_real_pace(sim: FactorySimulator, monkeypatch) -> None:
    sim.settings = dict(sim.settings, hourly_noise=0.0, breakdown_probability=0.0, style_change_probability=0.0)

    def fake_predict(bundle, rows):  # the model says every line will meet its target
        return rows.assign(met_target_pred=1, risk_probability=0.1)

    monkeypatch.setattr(live, "predict_line", fake_predict)

    start = live.build_live_view(sim, bundle={})
    # Hour 0: projection comes only from the model (sewing "met" gap is 0 -> exactly on target)
    assert start.loc[1, "projected_share"] == pytest.approx(1.0)

    for _ in sim.stream():
        pass
    end = live.build_live_view(sim, bundle={})
    # End of day: projection is the real output, so the slow sewing line shows as behind
    assert end.loc[1, "projected_share"] == pytest.approx(0.60 / 0.80)
    assert end.loc[1, "status"] == live.STATUS_BEHIND

    summary = live.factory_summary(end)
    assert summary["lines"] == 4 and summary["on_track"] + summary["at_risk"] == 4


def test_plan_for_view_only_counts_remaining_time(sim: FactorySimulator, monkeypatch) -> None:
    monkeypatch.setattr(live, "predict_line", lambda bundle, rows: rows.assign(met_target_pred=0, risk_probability=0.9))
    for _ in range(4):
        sim.step()
    view = live.build_live_view(sim, bundle={})
    plan = live.plan_for_view(view)
    # Halfway through the day, a moved operator works only the remaining half
    assert view["remaining_minutes_per_worker"].iloc[0] == pytest.approx((50 * 480 + 6000) / 2 / 50)
    assert plan.status in {"OPTIMAL", "FEASIBLE", "NO_ACTION_NEEDED"}
