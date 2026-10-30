"""Tests for loomiq.features.build_features."""

from loomiq.features import build_features as bf


def test_target_column_name() -> None:
    assert bf.TARGET_COLUMN == "met_target"


def test_leakage_columns_are_listed() -> None:
    # actual_productivity reveals the answer, so it must never be a model input
    assert "actual_productivity" in bf.LEAKAGE_COLUMNS


def test_feature_functions_exist() -> None:
    for name in ["add_target", "add_per_worker_features", "encode_categoricals", "build_features"]:
        assert callable(getattr(bf, name))


# TODO: Once add_target is implemented, check that a row with
#       actual=0.82 and targeted=0.80 gives met_target == 1.
# TODO: Check that build_features never returns LEAKAGE_COLUMNS in X.
