"""Load the trained productivity model and predict for production lines.

Example:
    >>> from loomiq.models.predict import load_model, predict_line
    >>> bundle = load_model()
    >>> predict_line(bundle, todays_rows)   # rows in the cleaned-data format
"""

from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.features.build_features import build_features
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(path: Path | None = None) -> dict[str, Any]:
    """Load the saved model bundle.

    Args:
        path: Optional custom path. Defaults to ``models/productivity_model.joblib``.

    Returns:
        A dict with ``model``, ``model_name``, ``feature_names``, ``use_incentive``
        and ``trained_at``.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    import joblib

    from loomiq.models.train_productivity import get_model_path

    path = path or get_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first: python scripts/train_all.py"
        )
    return joblib.load(path)


def prepare_features(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs for ``rows`` with exactly the columns the model was trained on."""
    X, _ = build_features(rows, use_incentive=bundle["use_incentive"])
    return X.reindex(columns=bundle["feature_names"], fill_value=0)


def predict_line(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Predict whether each line will meet its target.

    Args:
        bundle: Output of :func:`load_model`.
        rows: One row per line, in the cleaned-data format (``department``, ``team``,
            ``targeted_productivity``, ``smv``, ``wip``, ``over_time``, ...).
            ``actual_productivity`` is not needed.

    Returns:
        ``rows`` plus ``met_target_pred`` (1 = will meet, 0 = will miss) and
        ``risk_probability`` (probability of missing the target, 0 to 1).
    """
    X = prepare_features(bundle, rows)
    model = bundle["model"]
    result = rows.copy()
    result["met_target_pred"] = model.predict(X).astype(int)
    result["risk_probability"] = 1 - model.predict_proba(X)[:, 1]
    return result
