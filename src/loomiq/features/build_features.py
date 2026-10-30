"""Build model features from the cleaned productivity data.

Main outputs:
    * ``met_target`` (target): 1 if actual_productivity >= targeted_productivity.
    * ``productivity_gap``: actual minus targeted productivity (for analysis only,
      it must NOT be used as a model input because it leaks the target).
    * Per-worker features, e.g. overtime and incentive per worker.
    * One-hot encoding for categorical columns.
"""

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "met_target"
CATEGORICAL_COLUMNS = ["quarter", "department", "day", "team"]

# Columns that reveal the answer and must never be model inputs
LEAKAGE_COLUMNS = ["actual_productivity", "productivity_gap"]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary ``met_target`` column and the ``productivity_gap`` column.

    Args:
        df: Cleaned dataset.

    Returns:
        A copy of ``df`` with the two new columns.
    """
    # TODO: met_target = (actual_productivity >= targeted_productivity).astype(int)
    # TODO: productivity_gap = actual_productivity - targeted_productivity
    raise NotImplementedError("add_target is not implemented yet")


def add_per_worker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features normalized by team size.

    Args:
        df: Dataset with ``no_of_workers``.

    Returns:
        A copy of ``df`` with per-worker columns.
    """
    # TODO: overtime_per_worker = over_time / no_of_workers
    # TODO: incentive_per_worker = incentive / no_of_workers
    # TODO: wip_per_worker = wip / no_of_workers
    # TODO: idle_ratio = idle_men / no_of_workers
    # TODO: Guard against division by zero.
    raise NotImplementedError("add_per_worker_features is not implemented yet")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns.

    Args:
        df: Dataset with categorical columns.

    Returns:
        A copy of ``df`` with encoded columns.
    """
    # TODO: pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    # TODO: Consider keeping "team" as a category for XGBoost's native support instead.
    raise NotImplementedError("encode_categoricals is not implemented yet")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Run all feature steps and split into inputs (X) and target (y).

    Args:
        df: Cleaned dataset.

    Returns:
        A tuple ``(X, y)`` where ``X`` excludes the target, leakage columns and ``date``.
    """
    # TODO: add_target -> add_per_worker_features -> encode_categoricals.
    # TODO: Drop TARGET_COLUMN, LEAKAGE_COLUMNS and "date" from X.
    # TODO: Save the full feature table to data/processed/<features_file>.
    raise NotImplementedError("build_features is not implemented yet")
