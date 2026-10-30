"""Train models that predict whether a line meets its daily target.

Models compared (all with the same stratified cross-validation):
    1. Baseline: always predict the majority class (sklearn DummyClassifier).
    2. RandomForestClassifier.
    3. XGBClassifier.

The best model by the configured metric (default F1) is saved to ``models/``.
Heavy libraries (scikit-learn, xgboost, joblib) are imported inside functions
so ``import loomiq`` stays fast.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Cross-validation scores for one model."""

    name: str
    model: Any
    cv_scores: dict[str, float] = field(default_factory=dict)


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified train and test sets.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    # TODO: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    raise NotImplementedError("split_data is not implemented yet")


def get_candidate_models(config: dict[str, Any]) -> dict[str, Any]:
    """Build the baseline, RandomForest and XGBoost models from config params.

    Args:
        config: The full project config (uses ``productivity_model`` section).

    Returns:
        Mapping of model name to an unfitted estimator.
    """
    # TODO: Return {"baseline": DummyClassifier(strategy="most_frequent"),
    #               "random_forest": RandomForestClassifier(**params, random_state=seed),
    #               "xgboost": XGBClassifier(**params, random_state=seed)}
    raise NotImplementedError("get_candidate_models is not implemented yet")


def evaluate_with_cv(
    models: dict[str, Any], X: pd.DataFrame, y: pd.Series, folds: int, seed: int
) -> list[ModelResult]:
    """Score every model with stratified k-fold cross-validation.

    Returns:
        One :class:`ModelResult` per model with mean accuracy, F1, precision, recall.
    """
    # TODO: Use StratifiedKFold + cross_validate with scoring
    #       ["accuracy", "f1", "precision", "recall"].
    raise NotImplementedError("evaluate_with_cv is not implemented yet")


def select_best_model(results: list[ModelResult], metric: str = "f1") -> ModelResult:
    """Return the model with the highest mean CV score for ``metric``."""
    # TODO: max(results, key=lambda r: r.cv_scores[metric])
    raise NotImplementedError("select_best_model is not implemented yet")


def save_model(model: Any, path: Path) -> None:
    """Save a fitted model to disk with joblib."""
    # TODO: path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
    raise NotImplementedError("save_model is not implemented yet")


def train_productivity_model() -> ModelResult:
    """Full training pipeline: load -> clean -> features -> CV -> fit best -> save.

    Returns:
        The winning model result, refit on the full training set.
    """
    # TODO: 1. Load and clean data (loomiq.data).
    # TODO: 2. Build features (loomiq.features.build_features).
    # TODO: 3. Split data, evaluate candidates with CV, select best.
    # TODO: 4. Refit best on train, report test-set metrics, save to models/.
    # TODO: 5. Save a metrics JSON to reports/ for the README results table.
    raise NotImplementedError("train_productivity_model is not implemented yet")
