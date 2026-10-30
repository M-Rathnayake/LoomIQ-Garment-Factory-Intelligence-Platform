"""Tests for loomiq.optimization.recommender (plus the DHU helper)."""

import pytest

from loomiq.optimization.recommender import Recommendation, recommend_actions
from loomiq.vision.detect import compute_dhu


def test_recommendation_defaults() -> None:
    rec = Recommendation(line_id="Line 4", action="add_overtime", amount=60)
    assert rec.source_line_id is None
    assert rec.expected_gain_pieces == 0.0


def test_recommend_actions_exists() -> None:
    assert callable(recommend_actions)


def test_compute_dhu() -> None:
    assert compute_dhu(total_defects=5, units_inspected=200) == pytest.approx(2.5)


def test_compute_dhu_rejects_zero_units() -> None:
    with pytest.raises(ValueError):
        compute_dhu(total_defects=1, units_inspected=0)


# TODO: Once recommend_actions is implemented, test that:
#       - a line already on track gets "no_change",
#       - overtime never exceeds max_overtime_minutes,
#       - total operators stay the same after reallocation.
