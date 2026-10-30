"""Tests for loomiq.vision.detect (DHU helper)."""

import pytest

from loomiq.vision.detect import compute_dhu


def test_compute_dhu() -> None:
    assert compute_dhu(total_defects=5, units_inspected=200) == pytest.approx(2.5)


def test_compute_dhu_rejects_zero_units() -> None:
    with pytest.raises(ValueError):
        compute_dhu(total_defects=1, units_inspected=0)
