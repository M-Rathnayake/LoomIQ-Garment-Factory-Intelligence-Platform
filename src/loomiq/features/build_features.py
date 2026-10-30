"""Build model features from the cleaned productivity data.

Main outputs:
    * ``met_target`` (target): 1 if actual_productivity >= targeted_productivity.
    * ``productivity_gap``: actual minus targeted productivity. For analysis only:
      it reveals the answer, so it must NEVER be a model input.
    * Per-worker features, so big and small teams can be compared fairly.
    * One-hot encoding for categorical columns.

Two steps, on purpose:
    1. :func:`build_feature_table` adds new columns but keeps the data readable.
       This is what gets saved to ``data/processed/``.
    2. :func:`build_features` encodes the table into model inputs ``X`` and target ``y``.
       The same function is used at training time and for live predictions,
       so the columns always match.

Run from the command line:
    python -m loomiq.features.build_features
"""

from pathlib import Path

import pandas as pd

from loomiq.config import get_path, load_config
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "met_target"
CATEGORICAL_COLUMNS = ["department", "day", "quarter", "team"]
TEAMS = list(range(1, 13))

# Columns that reveal the answer and must never be model inputs
LEAKAGE_COLUMNS = ["actual_productivity", "productivity_gap"]

# Incentive may be paid *because* a team hit its target (see notebook 02),
# so it is only used when explicitly asked for.
INCENTIVE_COLUMNS = ["incentive", "incentive_per_worker"]

NUMERIC_FEATURES = [
    "targeted_productivity",
    "smv",
    "wip",
    "over_time",
    "idle_time",
    "idle_men",
    "no_of_style_change",
    "no_of_workers",
    "overtime_per_worker",
    "wip_per_worker",
    "idle_ratio",
]
PER_WORKER_SOURCES = {
    "overtime_per_worker": "over_time",
    "incentive_per_worker": "incentive",
    "wip_per_worker": "wip",
    "idle_ratio": "idle_men",
}
# Not used as inputs: "date" (only 10 weeks of data, would not generalize) and
# "wip_missing" (identical to department == finishing in this dataset).


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary ``met_target`` column and the ``productivity_gap`` column.

    Args:
        df: Cleaned dataset.

    Returns:
        A copy of ``df`` with the two new columns.
    """
    df = df.copy()
    df["productivity_gap"] = df["actual_productivity"] - df["targeted_productivity"]
    df[TARGET_COLUMN] = (df["productivity_gap"] >= 0).astype(int)
    return df


def add_per_worker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features normalized by team size.

    Args:
        df: Dataset with ``no_of_workers``.

    Returns:
        A copy of ``df`` with ``overtime_per_worker``, ``incentive_per_worker``,
        ``wip_per_worker`` and ``idle_ratio``.
    """
    df = df.copy()
    # Treat a team size of 0 as missing to avoid division by zero, then use 0
    workers = df["no_of_workers"].where(df["no_of_workers"] > 0)
    for new_column, source in PER_WORKER_SOURCES.items():
        if source in df.columns:  # live rows may not have every column (e.g. incentive)
            df[new_column] = (df[source] / workers).fillna(0)
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add the target and engineered features, keeping the table readable.

    Args:
        df: Cleaned dataset (see :mod:`loomiq.data.clean`).

    Returns:
        The cleaned data plus ``met_target``, ``productivity_gap`` and per-worker columns.
    """
    return add_per_worker_features(add_target(df))


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns.

    Fixed category lists are used, so a single row (e.g. one live line)
    produces exactly the same columns as the full training data.

    Args:
        df: Dataset with categorical columns.

    Returns:
        A copy of ``df`` with encoded columns such as ``department_finishing``.
    """
    from loomiq.data.clean import set_category_dtypes

    df = set_category_dtypes(df)
    df["team"] = pd.Categorical(df["team"], categories=TEAMS)
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)


def get_feature_columns(use_incentive: bool = False) -> list[str]:
    """Return the numeric input columns (before one-hot encoding)."""
    return NUMERIC_FEATURES + (INCENTIVE_COLUMNS if use_incentive else [])


def build_features(
    df: pd.DataFrame, use_incentive: bool = False
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Turn cleaned (or feature-table) rows into model inputs ``X`` and target ``y``.

    Args:
        df: Cleaned dataset or output of :func:`build_feature_table`.
            For live predictions the target columns may be missing.
        use_incentive: Include incentive columns as inputs.

    Returns:
        A tuple ``(X, y)``. ``y`` is ``None`` when ``actual_productivity`` is not
        available (e.g. predicting a line during the shift).
    """
    has_answer = "actual_productivity" in df.columns
    table = build_feature_table(df) if has_answer else add_per_worker_features(df)

    inputs = table[get_feature_columns(use_incentive) + CATEGORICAL_COLUMNS]
    X = encode_categoricals(inputs).astype(float)

    leaked = [c for c in X.columns if c in LEAKAGE_COLUMNS or c == TARGET_COLUMN]
    if leaked:
        raise ValueError(f"Leakage columns found in features: {leaked}")

    y = table[TARGET_COLUMN] if has_answer else None
    return X, y


def get_features_path() -> Path:
    """Return the location of the saved feature table."""
    return get_path("processed_data") / load_config()["data"]["features_file"]


def save_feature_table(table: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the readable feature table to CSV and return the path."""
    path = path or get_features_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Saved feature table (%s rows, %s columns) to %s", *table.shape, path)
    return path


def load_feature_table(path: Path | None = None) -> pd.DataFrame:
    """Read the saved feature table back with the correct dtypes restored."""
    from loomiq.data.clean import set_category_dtypes

    df = pd.read_csv(path or get_features_path(), parse_dates=["date"])
    return set_category_dtypes(df)


def run_feature_building() -> pd.DataFrame:
    """Load the cleaned data, build the feature table, save it, and return it."""
    from loomiq.data.clean import load_clean_productivity

    table = build_feature_table(load_clean_productivity())
    save_feature_table(table)
    return table


if __name__ == "__main__":
    run_feature_building()
