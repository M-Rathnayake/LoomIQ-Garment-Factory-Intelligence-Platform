"""Tests for loomiq.features.build_features."""

import pandas as pd
import pytest

from loomiq.features import build_features as bf


@pytest.fixture
def clean_rows() -> pd.DataFrame:
    """Three cleaned rows: one meets target exactly, one beats it, one misses."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-01", "2015-01-01", "2015-01-03"]),
            "quarter": ["Quarter1", "Quarter1", "Quarter1"],
            "department": ["sewing", "finishing", "sewing"],
            "day": ["Thursday", "Thursday", "Saturday"],
            "team": [8, 1, 7],
            "targeted_productivity": [0.80, 0.75, 0.70],
            "smv": [26.16, 3.94, 24.26],
            "wip": [1108.0, 0.0, 900.0],
            "over_time": [7080, 960, 0],
            "incentive": [98, 0, 0],
            "idle_time": [0.0, 0.0, 90.0],
            "idle_men": [0, 0, 10],
            "no_of_style_change": [0, 0, 1],
            "no_of_workers": [59.0, 8.0, 0.0],
            "actual_productivity": [0.80, 0.89, 0.50],
            "wip_missing": [False, True, False],
        }
    )


def test_target_definition(clean_rows: pd.DataFrame) -> None:
    table = bf.add_target(clean_rows)
    # Exactly on target counts as met
    assert table[bf.TARGET_COLUMN].tolist() == [1, 1, 0]
    assert table["productivity_gap"].round(2).tolist() == [0.0, 0.14, -0.2]


def test_per_worker_features(clean_rows: pd.DataFrame) -> None:
    table = bf.add_per_worker_features(clean_rows)
    assert table.loc[0, "overtime_per_worker"] == pytest.approx(7080 / 59)
    assert table.loc[1, "wip_per_worker"] == 0
    # A team size of 0 must not create infinite values
    assert table.loc[2, ["overtime_per_worker", "idle_ratio"]].tolist() == [0, 0]


def test_no_leakage_in_features(clean_rows: pd.DataFrame) -> None:
    X, y = bf.build_features(clean_rows, use_incentive=True)
    for column in bf.LEAKAGE_COLUMNS + [bf.TARGET_COLUMN, "date", "wip_missing"]:
        assert column not in X.columns
    assert y.tolist() == [1, 1, 0]


def test_incentive_switch(clean_rows: pd.DataFrame) -> None:
    X_without, _ = bf.build_features(clean_rows, use_incentive=False)
    X_with, _ = bf.build_features(clean_rows, use_incentive=True)
    assert "incentive" not in X_without.columns
    assert {"incentive", "incentive_per_worker"} <= set(X_with.columns)


def test_single_row_has_same_columns(clean_rows: pd.DataFrame) -> None:
    # A single live line must produce exactly the training columns
    X_all, _ = bf.build_features(clean_rows)
    X_one, y_one = bf.build_features(clean_rows.iloc[[1]].drop(columns="actual_productivity"))
    assert list(X_one.columns) == list(X_all.columns)
    assert y_one is None
    # 11 numeric + 1 department + 5 days + 3 quarters + 11 teams (first category dropped)
    assert X_all.shape[1] == 31


def test_live_rows_without_incentive_column(clean_rows: pd.DataFrame) -> None:
    live = clean_rows.drop(columns=["actual_productivity", "incentive"])
    X, _ = bf.build_features(live, use_incentive=False)
    assert X.notna().all().all()
