"""Clean the raw UCI garment productivity dataset.

Known issues in the raw data:
    * ``department`` has a typo ("sweing" instead of "sewing") and trailing
      spaces ("finishing ").
    * ``wip`` (work in progress) is missing for almost all finishing rows,
      because finishing lines do not hold WIP the same way sewing lines do.
    * ``no_of_workers`` is stored as a float (e.g. 59.0) but should be an int.
    * ``actual_productivity`` can be slightly above 1.0 and may need capping.
    * ``quarter`` has a rare "Quarter5" value (days 29-31 of a month).
"""

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def fix_department_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and fix the "sweing" typo in ``department``.

    Args:
        df: Raw dataset.

    Returns:
        A copy of ``df`` with ``department`` values in {"sewing", "finishing"}.
    """
    # TODO: df["department"].str.strip().replace({"sweing": "sewing"})
    raise NotImplementedError("fix_department_names is not implemented yet")


def fill_missing_wip(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing ``wip`` values.

    Args:
        df: Dataset with cleaned department names.

    Returns:
        A copy of ``df`` with no missing ``wip``.
    """
    # TODO: Fill wip with 0 for finishing rows (no WIP held there).
    # TODO: Add a "wip_missing" flag column before filling, so the model can use it.
    raise NotImplementedError("fill_missing_wip is not implemented yet")


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to the correct data types.

    Args:
        df: Dataset to convert.

    Returns:
        A copy of ``df`` with proper dtypes.
    """
    # TODO: date -> datetime64, no_of_workers -> int,
    #       quarter / department / day / team -> category.
    raise NotImplementedError("fix_dtypes is not implemented yet")


def clean_productivity(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order.

    Args:
        df: Raw dataset from :func:`loomiq.data.load.load_raw_productivity`.

    Returns:
        The cleaned dataset, ready for feature engineering.
    """
    # TODO: Chain fix_department_names -> fill_missing_wip -> fix_dtypes.
    # TODO: Decide whether to cap actual_productivity at 1.0 (document the choice).
    # TODO: Drop exact duplicate rows and log how many were removed.
    # TODO: Save the result to data/interim/<cleaned_file>.
    raise NotImplementedError("clean_productivity is not implemented yet")
