"""Load the trained productivity model and predict for production lines."""

from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(path: Path | None = None) -> Any:
    """Load the saved productivity model.

    Args:
        path: Optional custom path. Defaults to ``models/<model_file>`` from config.

    Returns:
        The fitted model.
    """
    # TODO: Resolve default path with loomiq.config.get_path("models").
    # TODO: joblib.load(path); raise a clear error if the model is not trained yet.
    raise NotImplementedError("load_model is not implemented yet")


def predict_line(model: Any, line_features: pd.DataFrame) -> pd.DataFrame:
    """Predict whether each line will meet its target.

    Args:
        model: Fitted model from :func:`load_model`.
        line_features: One row per line, with the same columns used in training.

    Returns:
        DataFrame with ``met_target_pred`` (0/1) and ``risk_probability``
        (probability of missing the target).
    """
    # TODO: Align columns to model.feature_names_in_ (fill missing with 0).
    # TODO: risk_probability = 1 - model.predict_proba(X)[:, 1]
    raise NotImplementedError("predict_line is not implemented yet")
