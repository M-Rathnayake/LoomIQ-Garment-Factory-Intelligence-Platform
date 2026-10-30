"""Tests for loomiq.data (loading and cleaning).

Unit tests use a tiny hand-made sample so they run without the real dataset.
The last test checks the real file and is skipped if it is not downloaded yet.
"""

import pandas as pd
import pytest

from loomiq.config import load_config
from loomiq.data import clean, load


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Small example of raw data containing every known issue."""
    return pd.DataFrame(
        {
            "date": ["1/1/2015", "1/1/2015", "1/29/2015", "2/5/2015", "2/5/2015"],
            "quarter": ["Quarter1", "Quarter1", "Quarter5", "Quarter1", "Quarter1"],
            "department": ["sweing", "finishing ", "finishing", "sweing", "sweing"],
            "day": ["Thursday", "Thursday", "Thursday", "Thursday", "Thursday"],
            "team": [8, 1, 2, 7, 7],
            "targeted_productivity": [0.80, 0.75, 0.70, 0.07, 0.07],
            "smv": [26.16, 3.94, 3.94, 24.26, 24.26],
            "wip": [1108.0, None, None, 900.0, 900.0],
            "over_time": [7080, 960, 960, 6960, 6960],
            "incentive": [98, 0, 0, 0, 0],
            "idle_time": [0.0, 0.0, 0.0, 0.0, 0.0],
            "idle_men": [0, 0, 0, 0, 0],
            "no_of_style_change": [0, 0, 0, 0, 0],
            "no_of_workers": [59.0, 8.0, 10.0, 56.5, 56.5],
            "actual_productivity": [0.94, 0.89, 1.05, 0.52, 0.52],
        }
    )


@pytest.fixture
def cleaned(raw_sample: pd.DataFrame) -> pd.DataFrame:
    return clean.clean_productivity(raw_sample)


def test_config_points_to_garments_clean() -> None:
    assert load_config()["data"]["cleaned_file"] == "garments_clean.csv"
    assert clean.get_clean_data_path().parent.name == "interim"


def test_raw_data_path_points_to_data_raw() -> None:
    path = load.get_raw_data_path()
    assert path.parent.name == "raw"
    assert path.suffix == ".csv"


def test_no_missing_values(cleaned: pd.DataFrame) -> None:
    assert cleaned.isna().sum().sum() == 0


def test_no_department_typos(cleaned: pd.DataFrame) -> None:
    assert set(cleaned["department"]) == {"sewing", "finishing"}


def test_missing_wip_filled_with_zero_and_flagged(cleaned: pd.DataFrame) -> None:
    finishing = cleaned[cleaned["department"] == "finishing"]
    assert (finishing["wip"] == 0).all()
    assert finishing["wip_missing"].all()
    assert not cleaned.loc[cleaned["department"] == "sewing", "wip_missing"].any()


def test_missing_sewing_wip_is_an_error(raw_sample: pd.DataFrame) -> None:
    raw_sample.loc[0, "wip"] = None
    with pytest.raises(ValueError):
        clean.fill_missing_wip(clean.fix_department_names(raw_sample))


def test_target_typo_fixed(cleaned: pd.DataFrame) -> None:
    assert cleaned["targeted_productivity"].min() >= 0.35
    assert 0.07 not in cleaned["targeted_productivity"].round(2).tolist()


def test_quarter5_merged(cleaned: pd.DataFrame) -> None:
    assert "Quarter5" not in cleaned["quarter"].astype(str).tolist()


def test_dtypes(cleaned: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    for column in ["department", "day", "quarter"]:
        assert isinstance(cleaned[column].dtype, pd.CategoricalDtype)
    # Half-shift workers (56.5) must survive cleaning
    assert 56.5 in cleaned["no_of_workers"].tolist()


def test_duplicates_removed(cleaned: pd.DataFrame) -> None:
    assert len(cleaned) == 4
    assert not cleaned.duplicated().any()


def test_odd_values_are_kept(cleaned: pd.DataFrame) -> None:
    assert cleaned["actual_productivity"].max() > 1.0
    assert clean.find_odd_values(cleaned)["actual_productivity_above_1"] == 1


def test_save_and_reload_keeps_dtypes(cleaned: pd.DataFrame, tmp_path) -> None:
    path = clean.save_clean(cleaned, tmp_path / "garments_clean.csv")
    reloaded = clean.load_clean_productivity(path)
    assert reloaded.shape == cleaned.shape
    assert isinstance(reloaded["day"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["date"])


@pytest.mark.skipif(not load.get_raw_data_path().exists(), reason="raw dataset not downloaded")
def test_real_dataset_cleans() -> None:
    df = clean.clean_productivity(load.load_raw_productivity())
    assert df.shape[0] == 1197
    assert df.isna().sum().sum() == 0
    assert set(df["department"]) == {"sewing", "finishing"}
    assert df["wip_missing"].sum() == 506
