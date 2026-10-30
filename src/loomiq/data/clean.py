"""Clean the raw UCI garment productivity dataset.

Issues found in the raw data (see notebooks/01_data_understanding.ipynb):
    * ``department`` has a typo ("sweing" instead of "sewing") and trailing
      spaces ("finishing ").
    * ``wip`` (work in progress) is missing in 506 rows, all in finishing,
      because finishing does not track WIP. Filled with 0 plus a flag column.
    * ``targeted_productivity`` has one value of 0.07 (team 7, 2015-02-05)
      where the same team is at 0.70 on the surrounding days. Fixed to 0.70.
    * ``quarter`` has a "Quarter5" that only covers 29 and 31 January.
      It is a two-day stub of the last week, so it is merged into "Quarter4".
    * ``date`` is a text column in M/D/YYYY format.

Values that look odd but are kept on purpose (they are plausible, not errors):
    * ``actual_productivity`` above 1.0 (teams beating the standard).
    * Very high ``incentive`` (all on one day in finishing, a bonus event).
    * Very high ``over_time`` and ``idle_time`` (real events such as breakdowns).
    * Fractional ``no_of_workers`` like 56.5 (part-shift workers), kept as float.

Run the full pipeline from the command line:
    python -m loomiq.data.clean
"""

from pathlib import Path

import numpy as np
import pandas as pd

from loomiq.config import get_path, load_config
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

DATE_FORMAT = "%m/%d/%Y"
DEPARTMENTS = ["sewing", "finishing"]
# The factory week runs Saturday to Thursday; Friday is the weekly day off.
DAY_ORDER = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
QUARTER_ORDER = ["Quarter1", "Quarter2", "Quarter3", "Quarter4"]

DEPARTMENT_FIXES = {"sweing": "sewing"}
QUARTER_FIXES = {"Quarter5": "Quarter4"}
TARGET_FIXES = {0.07: 0.70}


def fix_department_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and fix the "sweing" typo in ``department``.

    Args:
        df: Raw dataset.

    Returns:
        A copy of ``df`` with ``department`` values in {"sewing", "finishing"}.
    """
    df = df.copy()
    df["department"] = df["department"].str.strip().replace(DEPARTMENT_FIXES)
    return df


def fill_missing_wip(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``wip`` with 0 for finishing rows and add a ``wip_missing`` flag.

    Finishing does not track work in progress, so a missing value means
    "not tracked" rather than "unknown". Filling with 0 keeps that meaning;
    the average of sewing WIP would invent numbers that never existed.

    Args:
        df: Dataset with cleaned department names.

    Returns:
        A copy of ``df`` with no missing ``wip``.

    Raises:
        ValueError: If any sewing row is missing ``wip`` (unexpected for this data).
    """
    df = df.copy()
    missing = df["wip"].isna()
    unexpected = missing & (df["department"] != "finishing")
    if unexpected.any():
        raise ValueError(f"{unexpected.sum()} non-finishing rows have missing wip")

    df["wip_missing"] = missing
    df["wip"] = df["wip"].fillna(0)
    return df


def fix_target_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Correct known data-entry errors in ``targeted_productivity``.

    Args:
        df: Dataset to fix.

    Returns:
        A copy of ``df`` with typos such as 0.07 replaced by 0.70.
    """
    df = df.copy()
    for wrong, right in TARGET_FIXES.items():
        is_typo = np.isclose(df["targeted_productivity"], wrong)
        df.loc[is_typo, "targeted_productivity"] = right
    return df


def merge_quarter5(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the tiny "Quarter5" (days 29 and 31 of January) into "Quarter4".

    Args:
        df: Dataset to fix.

    Returns:
        A copy of ``df`` with only Quarter1 to Quarter4.
    """
    df = df.copy()
    df["quarter"] = df["quarter"].replace(QUARTER_FIXES)
    return df


def set_category_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set ``department``, ``day`` and ``quarter`` as categorical columns.

    ``day`` and ``quarter`` are ordered, so plots and tables sort naturally.
    Also used when reading the saved CSV back in, since CSV loses dtypes.

    Args:
        df: Dataset with cleaned text values.

    Returns:
        A copy of ``df`` with categorical dtypes.
    """
    df = df.copy()
    df["department"] = pd.Categorical(df["department"], categories=DEPARTMENTS)
    df["day"] = pd.Categorical(df["day"], categories=DAY_ORDER, ordered=True)
    df["quarter"] = pd.Categorical(df["quarter"], categories=QUARTER_ORDER, ordered=True)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``date`` to datetime and text columns to categories.

    ``team`` stays an integer ID and ``no_of_workers`` stays a float,
    because some sewing teams have half-shift workers (e.g. 56.5).

    Args:
        df: Dataset to convert.

    Returns:
        A copy of ``df`` with proper dtypes.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    return set_category_dtypes(df)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and log how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %s duplicate rows", before - len(df))
    return df


def find_odd_values(df: pd.DataFrame) -> dict[str, int]:
    """Count unusual values worth knowing about. Nothing is changed.

    Outliers use the 1.5 x IQR rule.

    Args:
        df: Cleaned dataset.

    Returns:
        Mapping of check name to the number of rows it flags.
    """

    def iqr_outliers(column: str) -> int:
        q1, q3 = df[column].quantile([0.25, 0.75])
        spread = 1.5 * (q3 - q1)
        return int(((df[column] < q1 - spread) | (df[column] > q3 + spread)).sum())

    return {
        "actual_productivity_above_1": int((df["actual_productivity"] > 1).sum()),
        "incentive_outliers": iqr_outliers("incentive"),
        "over_time_outliers": iqr_outliers("over_time"),
        "rows_with_idle_time": int((df["idle_time"] > 0).sum()),
        "fractional_workers": int((df["no_of_workers"] % 1 != 0).sum()),
    }


def validate_clean(df: pd.DataFrame) -> None:
    """Check that the cleaned data has no missing values and no leftover typos.

    Raises:
        ValueError: Describing the first problem found.
    """
    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(f"Missing values remain: {missing[missing > 0].to_dict()}")
    if set(df["department"].unique()) - set(DEPARTMENTS):
        raise ValueError(f"Unexpected departments: {df['department'].unique()}")
    if (df["date"].dt.day_name() != df["day"].astype(str)).any():
        raise ValueError("Some 'day' values do not match their 'date'")
    if df.duplicated().any():
        raise ValueError("Duplicate rows remain")


def clean_productivity(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order and validate the result.

    Args:
        df: Raw dataset from :func:`loomiq.data.load.load_raw_productivity`.

    Returns:
        The cleaned dataset, ready for feature engineering.
    """
    df = fix_department_names(df)
    df = fill_missing_wip(df)
    df = fix_target_typos(df)
    df = merge_quarter5(df)
    df = fix_dtypes(df)
    df = remove_duplicates(df)

    validate_clean(df)
    logger.info("Odd values kept on purpose: %s", find_odd_values(df))
    logger.info("Cleaned data: %s rows, %s columns", *df.shape)
    return df


def get_clean_data_path() -> Path:
    """Return the location of the cleaned CSV (``data/interim/garments_clean.csv``)."""
    return get_path("interim_data") / load_config()["data"]["cleaned_file"]


def save_clean(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the cleaned data to CSV and return the path."""
    path = path or get_clean_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)
    return path


def load_clean_productivity(path: Path | None = None) -> pd.DataFrame:
    """Read the cleaned CSV back with the correct dtypes restored."""
    path = path or get_clean_data_path()
    df = pd.read_csv(path, parse_dates=["date"])
    return set_category_dtypes(df)


def run_cleaning() -> pd.DataFrame:
    """Load the raw data, clean it, save it, and return the cleaned DataFrame."""
    # Imported here to keep load.py free to import from this module later
    from loomiq.data.load import load_raw_productivity

    df = clean_productivity(load_raw_productivity())
    save_clean(df)
    return df


if __name__ == "__main__":
    run_cleaning()
