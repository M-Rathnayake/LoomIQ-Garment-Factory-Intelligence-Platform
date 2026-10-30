"""Recommend fixes for at-risk lines using Google OR-Tools.

Idea:
    Given each line's predicted shortfall (pieces behind target), choose how
    many operators to move between lines and how much overtime to add, so that
    total expected shortfall is minimized under limits:
        * a line cannot give away operators it needs for its own target,
        * overtime per line is capped (config: max_overtime_minutes_per_line),
        * operators moved per line is capped (config: max_operators_moved_per_line).

``ortools`` is imported inside functions so ``import loomiq`` stays fast.
"""

from dataclasses import dataclass

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """One suggested action for a production line."""

    line_id: str
    action: str  # e.g. "move_operators", "add_overtime", "no_change"
    amount: float  # operators moved or overtime minutes
    source_line_id: str | None = None  # where operators come from, if moved
    expected_gain_pieces: float = 0.0
    reason: str = ""


def estimate_shortfall(lines: pd.DataFrame) -> pd.DataFrame:
    """Estimate pieces behind target for each line.

    Args:
        lines: Current line status with target, output so far, SMV and workers.

    Returns:
        ``lines`` with an added ``shortfall_pieces`` column (0 if on track).
    """
    # TODO: Project end-of-shift output from current hourly rate, compare with target.
    raise NotImplementedError("estimate_shortfall is not implemented yet")


def recommend_actions(
    lines: pd.DataFrame,
    max_overtime_minutes: int = 120,
    max_operators_moved: int = 5,
) -> list[Recommendation]:
    """Solve the operator / overtime reallocation problem.

    Args:
        lines: One row per line with shortfall, SMV, workers and risk probability.
        max_overtime_minutes: Overtime cap per line.
        max_operators_moved: Cap on operators moved into or out of a line.

    Returns:
        A list of :class:`Recommendation`, one or more per at-risk line.
    """
    # TODO: Build a CP-SAT model (ortools.sat.python.cp_model):
    #       - integer var move[i][j] = operators moved from line i to line j
    #       - integer var overtime[j] in [0, max_overtime_minutes]
    #       - gain per operator-minute = 1 / SMV (pieces per minute)
    #       - objective: minimize total remaining shortfall (+ small cost for overtime)
    # TODO: Respect solver_time_limit_seconds from config.
    # TODO: Convert the solution into Recommendation objects with readable reasons.
    raise NotImplementedError("recommend_actions is not implemented yet")
