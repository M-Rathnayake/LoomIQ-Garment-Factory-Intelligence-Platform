"""Tests for loomiq.data (loading and cleaning)."""

import pandas as pd

from loomiq.config import PROJECT_ROOT, load_config
from loomiq.data import clean, load


def test_config_loads() -> None:
    config = load_config()
    assert config["project"]["random_seed"] == 42
    assert (PROJECT_ROOT / "configs" / "config.yaml").exists()


def test_raw_data_path_points_to_data_raw() -> None:
    path = load.get_raw_data_path()
    assert path.parent.name == "raw"
    assert path.suffix == ".csv"


def test_cleaning_functions_exist() -> None:
    for name in ["fix_department_names", "fill_missing_wip", "fix_dtypes", "clean_productivity"]:
        assert callable(getattr(clean, name))


def sample_raw_rows() -> pd.DataFrame:
    """Tiny example of raw data with the known issues, for future tests."""
    return pd.DataFrame(
        {
            "department": ["sweing", "finishing ", "sweing"],
            "wip": [1108.0, None, 968.0],
            "no_of_workers": [59.0, 8.0, 30.5],
        }
    )


# TODO: Once fix_department_names is implemented, assert that
#       set(result["department"]) == {"sewing", "finishing"}.
# TODO: Once fill_missing_wip is implemented, assert result["wip"].isna().sum() == 0.
