"""Simulate hourly production data for N sewing lines.

The real UCI data is daily, so the live dashboard is fed by this simulator.
It should generate realistic values by sampling from distributions learned
from the cleaned UCI data (SMV, team size, incentive, idle time, etc.) and
adding random events such as machine breakdowns or style changes.

Example:
    >>> sim = FactorySimulator(n_lines=6, seed=42)
    >>> for hourly_df in sim.stream(hours=8):
    ...     print(hourly_df.head())
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LineState:
    """Current state of one production line."""

    line_id: str
    no_of_workers: int
    smv: float
    targeted_productivity: float
    daily_target_pieces: int
    output_so_far: int = 0
    idle_minutes: float = 0.0
    over_time_minutes: float = 0.0
    style_changes: int = 0


class FactorySimulator:
    """Generates one row per line per simulated hour."""

    def __init__(self, n_lines: int = 6, hours_per_shift: int = 8, seed: int = 42) -> None:
        """Create the simulator.

        Args:
            n_lines: Number of production lines to simulate.
            hours_per_shift: Hours in one shift.
            seed: Random seed so runs are reproducible.
        """
        self.n_lines = n_lines
        self.hours_per_shift = hours_per_shift
        self.seed = seed
        self.current_hour = 0
        self.lines: list[LineState] = []
        # TODO: self.rng = numpy.random.default_rng(seed)
        # TODO: Call self._init_lines() to create starting LineState objects.

    def _init_lines(self) -> list[LineState]:
        """Create starting states for each line, using realistic UCI ranges."""
        # TODO: Sample workers, SMV and targeted_productivity from cleaned UCI data.
        raise NotImplementedError("_init_lines is not implemented yet")

    def step(self) -> pd.DataFrame:
        """Advance one hour and return a DataFrame with one row per line."""
        # TODO: For each line, produce pieces = workers * 60 / smv * efficiency + noise.
        # TODO: Randomly inject events (breakdown -> idle time, style change).
        # TODO: Include columns matching the model's input features.
        raise NotImplementedError("step is not implemented yet")

    def stream(self, hours: int | None = None) -> Iterator[pd.DataFrame]:
        """Yield one hourly snapshot at a time until the shift ends."""
        # TODO: for _ in range(hours or self.hours_per_shift): yield self.step()
        raise NotImplementedError("stream is not implemented yet")

    def reset(self) -> None:
        """Start a new shift with fresh line states."""
        # TODO: Reset current_hour and re-create lines.
        raise NotImplementedError("reset is not implemented yet")
