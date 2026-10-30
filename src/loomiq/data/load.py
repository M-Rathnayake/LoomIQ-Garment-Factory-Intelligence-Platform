"""Load the raw UCI "Productivity Prediction of Garment Employees" dataset.

The CSV must be downloaded manually into ``data/raw/`` first.
See ``docs/DATASETS.md`` for instructions.

Raw columns: date, quarter, department, day, team, targeted_productivity,
smv, wip, over_time, incentive, idle_time, idle_men, no_of_style_change,
no_of_workers, actual_productivity.
"""

from pathlib import Path

import pandas as pd

from loomiq.config import get_path, load_config
from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def get_raw_data_path() -> Path:
    """Return the expected location of the raw productivity CSV."""
    file_name = load_config()["data"]["productivity_file"]
    return get_path("raw_data") / file_name


def load_raw_productivity(path: Path | None = None) -> pd.DataFrame:
    """Load the raw productivity CSV into a DataFrame.

    Args:
        path: Optional custom path. Defaults to ``data/raw/<productivity_file>``.

    Returns:
        The raw, unmodified dataset.

    Raises:
        FileNotFoundError: If the CSV has not been downloaded yet.
    """
    path = path or get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download it first (see docs/DATASETS.md)."
        )

    # TODO: Parse the "date" column as datetime (pd.read_csv(..., parse_dates=["date"])).
    # TODO: Log the shape and a quick summary of missing values.
    df = pd.read_csv(path)
    logger.info("Loaded raw productivity data: %s rows, %s columns", *df.shape)
    return df
