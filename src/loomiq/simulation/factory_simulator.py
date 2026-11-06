"""Simulate a factory day hour by hour, for the live dashboard.

The UCI data is daily, so the dashboard needs a simulator. To stay realistic,
each simulated day **replays a real day** from the test weeks (which the model
never saw in training):

    * Every line gets that day's real setup: department, team, SMV, workers,
      WIP, planned overtime and target.
    * Its real end-of-day productivity becomes the line's **hidden true pace**.
    * Output is revealed hour by hour, with random noise, and occasional
      **machine breakdowns** (idle workers) and **style changes** (slower pace).

Supervisors can change the day with :meth:`FactorySimulator.apply_plan`
(move operators, add overtime) and watch the effect in the following hours.

Productivity follows the dataset's definition:
    productivity = pieces x SMV / available minutes

Example:
    >>> sim = FactorySimulator(n_lines=12, seed=42)
    >>> for hourly_df in sim.stream():
    ...     print(hourly_df[["line_id", "pieces_so_far"]].head())
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

# Columns in the cleaned-data format, so live rows can go straight into the model
SETUP_COLUMNS = [
    "department", "team", "targeted_productivity", "smv", "wip",
    "over_time", "incentive", "no_of_workers",
]


@dataclass
class LineState:
    """Current state of one production line during the simulated day."""

    line_id: str
    department: str
    team: int
    targeted_productivity: float
    smv: float
    wip: float
    over_time: float  # planned overtime for the whole team (minutes)
    incentive: float
    no_of_workers: float
    true_productivity: float  # hidden: the pace the line will really run at
    target_pieces: float = 0.0
    pieces_so_far: float = 0.0
    available_minutes_so_far: float = 0.0
    extra_overtime_left: float = 0.0  # team minutes added by a plan, not yet worked
    idle_time: float = 0.0  # idle minutes so far
    idle_men: int = 0  # most workers idle at once so far
    no_of_style_change: int = 0
    style_penalty: float = 0.0
    events: list[str] = field(default_factory=list)


class FactorySimulator:
    """Generates one row per line per simulated hour."""

    def __init__(
        self,
        n_lines: int | None = None,
        hours_per_shift: int | None = None,
        seed: int = 42,
        data: pd.DataFrame | None = None,
        date: str | None = None,
    ) -> None:
        """Create the simulator and set up the first day.

        Args:
            n_lines: Number of production lines (defaults to config).
            hours_per_shift: Hours in one shift (defaults to config).
            seed: Random seed so runs are reproducible.
            data: Cleaned UCI data to sample real days from (loaded if not given).
            date: Replay this exact date (YYYY-MM-DD) instead of a random test day.
        """
        from loomiq.config import load_config

        config = load_config()
        self.settings = config["simulation"]
        self.shift_minutes = config["optimization"]["shift_minutes"]
        self.n_lines = n_lines or self.settings["n_lines"]
        self.hours_per_shift = hours_per_shift or self.settings["hours_per_shift"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._data = data
        self._fixed_date = date
        self.lines: list[LineState] = []
        self.history: list[dict[str, Any]] = []
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def _load_days(self) -> pd.DataFrame:
        """Real days to replay: the test weeks, which the model never trained on."""
        from loomiq.data.clean import load_clean_productivity
        from loomiq.models.train_productivity import time_based_split

        if self._data is None:
            self._data = load_clean_productivity()
        if self._fixed_date:
            return self._data
        _, test_days = time_based_split(self._data, test_size=0.2)
        return test_days

    def _init_lines(self) -> list[LineState]:
        """Pick a real day and turn (up to ``n_lines`` of) its lines into line states."""
        days = self._load_days()
        if self._fixed_date:
            self.date = pd.Timestamp(self._fixed_date)
        else:
            self.date = pd.Timestamp(self.rng.choice(sorted(days["date"].unique())))
        day = days[days["date"] == self.date]
        if day.empty:
            raise ValueError(f"No data for {self.date.date()}")

        # Keep the day's sewing / finishing mix (at least 2 of each, so lines can lend)
        if len(day) > self.n_lines:
            parts = []
            for dept, group in day.groupby("department", observed=True):
                share = max(2, round(self.n_lines * len(group) / len(day)))
                parts.append(group.sample(min(share, len(group)), random_state=int(self.rng.integers(1e9))))
            day = pd.concat(parts)
        day = day.sort_values(["department", "team"])  # sewing first, then team number

        self.day_name = str(day["day"].iloc[0])
        self.quarter = str(day["quarter"].iloc[0])
        lines = []
        for row in day.itertuples():
            line = LineState(
                line_id=f"{str(row.department).title()} {row.team}",
                department=str(row.department),
                team=int(row.team),
                targeted_productivity=float(row.targeted_productivity),
                smv=float(row.smv),
                wip=float(row.wip),
                over_time=float(row.over_time),
                incentive=float(row.incentive),
                no_of_workers=float(row.no_of_workers),
                true_productivity=float(row.actual_productivity),
            )
            line.target_pieces = line.targeted_productivity * self._day_minutes(line) / line.smv
            lines.append(line)
        return lines

    def _day_minutes(self, line: LineState) -> float:
        """Planned team minutes for the whole day: shift plus planned overtime."""
        return line.no_of_workers * self.shift_minutes + line.over_time

    # ------------------------------------------------------------- simulation
    @property
    def hours_left(self) -> int:
        return self.hours_per_shift - self.current_hour

    @property
    def is_finished(self) -> bool:
        return self.current_hour >= self.hours_per_shift

    def minutes_left(self, line: LineState) -> float:
        """Team minutes still to be worked today, including any added overtime."""
        return self._day_minutes(line) * self.hours_left / self.hours_per_shift + line.extra_overtime_left

    def step(self) -> pd.DataFrame:
        """Advance one hour and return a DataFrame with one row per line."""
        if self.is_finished:
            raise RuntimeError("The shift is over. Call reset() to start a new day.")

        rows = []
        for line in self.lines:
            minutes = self._day_minutes(line) / self.hours_per_shift
            if line.extra_overtime_left:
                share = line.extra_overtime_left / self.hours_left
                minutes += share
                line.extra_overtime_left -= share

            lost_minutes, event = self._random_event(line)
            pace = line.true_productivity * (1 - line.style_penalty)
            noise = self.rng.normal(1, self.settings["hourly_noise"])
            pieces = max(0.0, pace * (minutes - lost_minutes) / line.smv * noise)

            line.pieces_so_far += pieces
            line.available_minutes_so_far += minutes
            rows.append(self._row(line, pieces, event))

        self.current_hour += 1
        hourly = pd.DataFrame(rows)
        self.history.extend(rows)
        return hourly

    def _random_event(self, line: LineState) -> tuple[float, str]:
        """Maybe trigger a breakdown or style change. Returns (lost team minutes, event text)."""
        if self.rng.random() < self.settings["breakdown_probability"]:
            idle_men = int(min(self.rng.integers(5, 16), line.no_of_workers // 2))
            idle_minutes = float(self.rng.integers(3, 10) * 10)
            line.idle_time += idle_minutes
            line.idle_men = max(line.idle_men, idle_men)
            event = f"Machine breakdown: {idle_men} workers idle for {idle_minutes:.0f} min"
            line.events.append(f"Hour {self.current_hour + 1}: {event}")
            return idle_men * idle_minutes, event
        if self.rng.random() < self.settings["style_change_probability"]:
            line.no_of_style_change += 1
            line.style_penalty = self.settings["style_change_penalty"]
            event = "Style change: new garment, slower pace while workers learn it"
            line.events.append(f"Hour {self.current_hour + 1}: {event}")
            return 0.0, event
        return 0.0, ""

    def _row(self, line: LineState, pieces: float, event: str) -> dict[str, Any]:
        """One hourly record for the history table."""
        return {
            "date": self.date,
            "hour": self.current_hour + 1,
            "line_id": line.line_id,
            "department": line.department,
            "team": line.team,
            "no_of_workers": line.no_of_workers,
            "pieces_this_hour": round(pieces, 1),
            "pieces_so_far": round(line.pieces_so_far, 1),
            "target_so_far": round(line.target_pieces * (self.current_hour + 1) / self.hours_per_shift, 1),
            "target_pieces": round(line.target_pieces, 1),
            "event": event,
        }

    def stream(self, hours: int | None = None) -> Iterator[pd.DataFrame]:
        """Yield one hourly snapshot at a time until the shift ends (or ``hours`` pass)."""
        for _ in range(min(hours or self.hours_left, self.hours_left)):
            yield self.step()

    def reset(self, seed: int | None = None) -> None:
        """Start a new day (a different real day unless ``date`` was fixed)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_hour = 0
        self.history = []
        self.lines = self._init_lines()
        logger.info("New simulated day: replaying %s with %s lines", self.date.date(), len(self.lines))

    # ------------------------------------------------------------ interaction
    def apply_plan(self, plan: Any) -> list[str]:
        """Apply a :class:`~loomiq.optimization.recommender.RecommendationPlan`.

        Moves change team sizes for the rest of the day; overtime adds team
        minutes, spread over the remaining hours. Targets stay the same.

        Returns:
            A short description of each change applied.
        """
        if self.is_finished:
            raise RuntimeError("The shift is over; there is nothing left to change.")
        by_id = {line.line_id: line for line in self.lines}
        applied = []
        for action in plan.actions:
            if action.action == "move_operators":
                count = int(action.amount)
                by_id[action.source_line_id].no_of_workers -= count
                by_id[action.line_id].no_of_workers += count
                note = f"Moved {count} operator(s) from {action.source_line_id} to {action.line_id}"
            elif action.action == "add_overtime":
                line = by_id[action.line_id]
                line.extra_overtime_left += action.amount * line.no_of_workers
                note = f"Added {action.amount:.0f} min overtime per worker on {action.line_id}"
            else:
                continue
            applied.append(note)
            for line_id in {action.line_id, action.source_line_id} - {None}:
                by_id[line_id].events.append(f"Hour {self.current_hour}: {note}")
        return applied

    # --------------------------------------------------------------- outputs
    def snapshot(self) -> pd.DataFrame:
        """Current state of every line, in the cleaned-data format plus live columns."""
        rows = []
        for line in self.lines:
            row = {column: getattr(line, column) for column in SETUP_COLUMNS}
            row.update(
                {
                    "line_id": line.line_id,
                    "date": self.date,
                    "day": self.day_name,
                    "quarter": self.quarter,
                    "idle_time": line.idle_time,
                    "idle_men": line.idle_men,
                    "no_of_style_change": line.no_of_style_change,
                    "hour": self.current_hour,
                    "target_pieces": line.target_pieces,
                    "pieces_so_far": line.pieces_so_far,
                    "available_minutes_so_far": line.available_minutes_so_far,
                    "remaining_minutes": self.minutes_left(line),
                    "remaining_minutes_per_worker": self.minutes_left(line) / max(line.no_of_workers, 1),
                    "events": list(line.events),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def history_frame(self) -> pd.DataFrame:
        """All hourly rows generated so far today (empty, with columns, before hour 1)."""
        columns = [
            "date", "hour", "line_id", "department", "team", "no_of_workers", "pieces_this_hour",
            "pieces_so_far", "target_so_far", "target_pieces", "event",
        ]
        return pd.DataFrame(self.history, columns=columns)
