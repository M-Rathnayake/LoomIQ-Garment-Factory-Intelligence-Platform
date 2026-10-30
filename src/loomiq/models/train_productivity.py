"""Train models that predict whether a team meets its daily target.

How models are compared:
    * **Time-based split:** the last 20% of dates are the test set, so the model
      is always tested on days it has never seen, after the days it learned from.
    * **Cross-validation** (stratified 5-fold) on the training days picks the best model.
    * **Main metric: F1 for the "missed target" class.** Most days are "met",
      so a model that always says "met" gets 73% accuracy but catches nothing.

Candidates:
    1. Baseline: always predict the most common class.
    2. Logistic regression: a simple, explainable linear model.
    3. Random forest.
    4. XGBoost.

Run from the command line:
    python -m loomiq.models.train_productivity
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.config import get_path, load_config
from loomiq.features.build_features import build_features
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

MISSED = 0  # label value for "missed target"; the class we want to catch


@dataclass
class ModelResult:
    """Cross-validation scores for one model."""

    name: str
    model: Any
    cv_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Everything produced by one training run."""

    best: ModelResult
    all_results: list[ModelResult]
    test_metrics: dict[str, Any]
    feature_names: list[str]
    use_incentive: bool


def time_based_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date: the last ``test_size`` share of dates becomes the test set.

    Args:
        df: Data with a ``date`` column.
        test_size: Share of (unique) dates to hold out.

    Returns:
        ``(train_df, test_df)``. Every test date is later than every train date.
    """
    dates = sorted(df["date"].unique())
    n_test = max(1, round(len(dates) * test_size))
    first_test_date = dates[-n_test]
    is_test = df["date"] >= first_test_date
    return df[~is_test], df[is_test]


def get_scorers() -> dict[str, Any]:
    """Scoring functions used everywhere, focused on the "missed target" class."""
    from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score

    return {
        "missed_f1": make_scorer(f1_score, pos_label=MISSED),
        "missed_precision": make_scorer(precision_score, pos_label=MISSED, zero_division=0),
        "missed_recall": make_scorer(recall_score, pos_label=MISSED),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }


def get_candidate_models(config: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Build the candidate models from config params.

    Args:
        config: The full project config (uses the ``productivity_model`` section).
        y_train: Training labels, used to balance the classes for XGBoost.

    Returns:
        Mapping of model name to an unfitted estimator.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    params = config["productivity_model"]
    seed = config["project"]["random_seed"]
    # Give "missed" days the same total weight as "met" days
    balance = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(**params["logistic_regression"], class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            **params["random_forest"], class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            **params["xgboost"], scale_pos_weight=balance, random_state=seed, n_jobs=-1
        ),
    }


def evaluate_with_cv(
    models: dict[str, Any], X: pd.DataFrame, y: pd.Series, folds: int, seed: int
) -> list[ModelResult]:
    """Score every model with stratified k-fold cross-validation.

    Returns:
        One :class:`ModelResult` per model with the mean score for each metric.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=get_scorers())
        mean_scores = {
            metric.removeprefix("test_"): float(values.mean())
            for metric, values in scores.items()
            if metric.startswith("test_")
        }
        results.append(ModelResult(name=name, model=model, cv_scores=mean_scores))
        logger.info("CV %-20s missed F1 = %.3f", name, mean_scores["missed_f1"])
    return results


def select_best_model(results: list[ModelResult], metric: str = "missed_f1") -> ModelResult:
    """Return the model with the highest mean CV score for ``metric``."""
    return max(results, key=lambda result: result.cv_scores[metric])


def evaluate_on_test(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, departments: pd.Series
) -> dict[str, Any]:
    """Score a fitted model on the held-out test days, overall and per department.

    Returns:
        A JSON-friendly dict of metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    def score(y_true: pd.Series, y_pred, y_prob) -> dict[str, float]:
        return {
            "rows": int(len(y_true)),
            "missed_f1": float(f1_score(y_true, y_pred, pos_label=MISSED)),
            "missed_precision": float(precision_score(y_true, y_pred, pos_label=MISSED, zero_division=0)),
            "missed_recall": float(recall_score(y_true, y_pred, pos_label=MISSED)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float("nan"),
        }

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {"overall": score(y_test, y_pred, y_prob)}
    for dept in departments.unique():
        mask = (departments == dept).to_numpy()
        metrics[str(dept)] = score(y_test[mask], y_pred[mask], y_prob[mask])
    # Rows: actual missed / met; columns: predicted missed / met
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[MISSED, 1]).tolist()
    return metrics


def get_model_path() -> Path:
    """Return the location of the saved model (``models/productivity_model.joblib``)."""
    return get_path("models") / load_config()["productivity_model"]["model_file"]


def save_model(bundle: dict[str, Any], path: Path | None = None) -> Path:
    """Save the model bundle (model + feature names + settings) with joblib."""
    import joblib

    path = path or get_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved model to %s", path)
    return path


def save_metrics(result: TrainingResult, path: Path | None = None) -> Path:
    """Write CV and test metrics to ``reports/productivity_metrics.json``."""
    path = path or get_path("reports") / load_config()["productivity_model"]["metrics_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "best_model": result.best.name,
        "use_incentive": result.use_incentive,
        "cv": {r.name: r.cv_scores for r in result.all_results},
        "test": result.test_metrics,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", path)
    return path


def train_productivity_model(
    df: pd.DataFrame | None = None, use_incentive: bool | None = None, save: bool = True
) -> TrainingResult:
    """Full training pipeline: split -> features -> CV -> refit best -> test -> save.

    Args:
        df: Cleaned data. Loaded from ``data/interim/`` if not given.
        use_incentive: Override the config setting for incentive features.
        save: Save the model and metrics to disk.

    Returns:
        A :class:`TrainingResult` with the refitted best model and its test metrics.
    """
    from loomiq.data.clean import load_clean_productivity

    config = load_config()
    params = config["productivity_model"]
    seed = config["project"]["random_seed"]
    if use_incentive is None:
        use_incentive = params["use_incentive"]

    df = load_clean_productivity() if df is None else df
    train_df, test_df = time_based_split(df, params["test_size"])
    X_train, y_train = build_features(train_df, use_incentive)
    X_test, y_test = build_features(test_df, use_incentive)
    logger.info(
        "Train: %s rows up to %s | Test: %s rows from %s",
        len(X_train), train_df["date"].max().date(), len(X_test), test_df["date"].min().date(),
    )

    models = get_candidate_models(config, y_train)
    results = evaluate_with_cv(models, X_train, y_train, params["cv_folds"], seed)
    best = select_best_model(results, params["scoring"])
    best.model.fit(X_train, y_train)
    test_metrics = evaluate_on_test(best.model, X_test, y_test, test_df["department"])
    logger.info("Best model: %s | test missed F1 = %.3f", best.name, test_metrics["overall"]["missed_f1"])

    result = TrainingResult(best, results, test_metrics, list(X_train.columns), use_incentive)
    if save:
        save_model(
            {
                "model": best.model,
                "model_name": best.name,
                "feature_names": result.feature_names,
                "use_incentive": use_incentive,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        save_metrics(result)
    return result


if __name__ == "__main__":
    train_productivity_model()
