"""Tests for loomiq.models (training helpers, prediction and SHAP explanations).

A small synthetic dataset is used so the tests are fast and need no downloads.
"""

import numpy as np
import pandas as pd
import pytest

from loomiq.features.build_features import build_features
from loomiq.models import explain, predict, train_productivity
from loomiq.models.train_productivity import ModelResult


@pytest.fixture
def synthetic_clean() -> pd.DataFrame:
    """60 days x 4 teams of fake cleaned data where idle time causes misses."""
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2015-01-01", periods=60, freq="D")
    for date in dates:
        for team in [1, 2, 3, 4]:
            idle = float(rng.random() < 0.3) * 60
            target = 0.75
            rows.append(
                {
                    "date": date,
                    "quarter": "Quarter1",
                    "department": "sewing" if team <= 2 else "finishing",
                    "day": "Monday",
                    "team": team,
                    "targeted_productivity": target,
                    "smv": 20.0 if team <= 2 else 4.0,
                    "wip": 1000.0 if team <= 2 else 0.0,
                    "over_time": 3000,
                    "incentive": 0,
                    "idle_time": idle,
                    "idle_men": 5 if idle else 0,
                    "no_of_style_change": 0,
                    "no_of_workers": 50.0,
                    "actual_productivity": target - (0.2 if idle else -0.05) + rng.normal(0, 0.02),
                    "wip_missing": team > 2,
                }
            )
    return pd.DataFrame(rows)


def test_time_based_split_has_no_overlap(synthetic_clean: pd.DataFrame) -> None:
    train, test = train_productivity.time_based_split(synthetic_clean, test_size=0.2)
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 12
    assert len(train) + len(test) == len(synthetic_clean)


def test_select_best_model() -> None:
    results = [
        ModelResult("a", None, {"missed_f1": 0.4}),
        ModelResult("b", None, {"missed_f1": 0.7}),
    ]
    assert train_productivity.select_best_model(results).name == "b"


def test_train_predict_explain_end_to_end(synthetic_clean: pd.DataFrame, tmp_path, monkeypatch) -> None:
    # Keep the test fast and write nothing to the real models/ folder
    monkeypatch.setattr(train_productivity, "get_model_path", lambda: tmp_path / "model.joblib")
    result = train_productivity.train_productivity_model(synthetic_clean, save=False)

    # Idle time fully explains misses here, so a real model must beat the baseline
    assert result.best.name != "baseline"
    assert result.test_metrics["overall"]["missed_f1"] > 0.8

    bundle = {
        "model": result.best.model,
        "model_name": result.best.name,
        "feature_names": result.feature_names,
        "use_incentive": False,
    }
    live = synthetic_clean.tail(4).drop(columns="actual_productivity")
    predictions = predict.predict_line(bundle, live)
    assert predictions["risk_probability"].between(0, 1).all()
    # Lines with idle time should get the higher risk
    idle = predictions["idle_time"] > 0
    if idle.any() and (~idle).any():
        assert predictions.loc[idle, "risk_probability"].min() > predictions.loc[~idle, "risk_probability"].max()

    X_train, _ = build_features(synthetic_clean)
    explainer = explain.build_explainer(bundle["model"], X_train)
    drivers = explain.explain_line(explainer, predict.prepare_features(bundle, live.iloc[[0]]), top_n=3)
    assert len(drivers) <= 3
    assert {"feature", "label", "value", "shap_value", "direction"} <= set(drivers[0])
    # One-hot columns are grouped back, so no encoded names like "team_2" appear
    assert all("_" not in d["feature"] or d["feature"] in explain.FEATURE_LABELS for d in drivers)


def test_original_column_mapping() -> None:
    assert explain.original_column("team_7") == "team"
    assert explain.original_column("department_finishing") == "department"
    assert explain.original_column("over_time") == "over_time"


def test_load_model_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "nope.joblib")
