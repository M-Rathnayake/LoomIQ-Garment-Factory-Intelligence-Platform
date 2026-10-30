"""Recommend fixes for at-risk lines using Google OR-Tools.

The idea, in factory terms:
    1. For every line, compare **target pieces** with **expected pieces** for the day.
       Expected pieces come from a live figure if available (the simulator), or
       otherwise from the productivity model's prediction (see :func:`estimate_shortfall`).
    2. Lines expected to fall short can be helped in two ways:
         * **move operators** from a line in the same department that has spare
           capacity (sewing operators are not moved to finishing, or back), or
         * **add overtime** for the whole line, in steps (e.g. 30 minutes).
    3. The CP-SAT solver picks the combination that leaves the fewest pieces short,
       with a small cost for every change so it does not suggest pointless moves.

Rules the solver must follow:
    * A line that lends operators must still meet its own target.
    * At most ``max_operators_moved_per_line`` operators move into or out of a line.
    * At most ``max_overtime_minutes_per_line`` extra overtime per worker.

Productivity in this dataset means (pieces x SMV) / available minutes, so:
    pieces = productivity x available minutes / SMV

Run a demo on a real day from the dataset:
    python -m loomiq.optimization.recommender --date 2015-03-09
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

SCALE = 10  # CP-SAT works with whole numbers, so pieces are counted in tenths


@dataclass
class Recommendation:
    """One suggested action for a production line."""

    line_id: str
    action: str  # "move_operators", "add_overtime" or "unresolved"
    amount: float  # operators moved, overtime minutes per worker, or pieces still short
    source_line_id: str | None = None  # where operators come from, if moved
    expected_gain_pieces: float = 0.0
    reason: str = ""


@dataclass
class RecommendationPlan:
    """The solver's full answer for one day."""

    actions: list[Recommendation]
    lines: pd.DataFrame  # one row per line, before and after the plan
    status: str  # "OPTIMAL", "FEASIBLE" or "NO_ACTION_NEEDED"
    pieces_short_before: float = 0.0
    pieces_short_after: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def pieces_recovered(self) -> float:
        return self.pieces_short_before - self.pieces_short_after


def get_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``optimization`` config section, with optional overrides."""
    from loomiq.config import load_config

    settings = dict(load_config()["optimization"])
    settings.update(overrides or {})
    return settings


def add_line_ids(lines: pd.DataFrame) -> pd.DataFrame:
    """Add a readable ``line_id`` such as "Sewing 7" if the data does not have one."""
    lines = lines.copy()
    if "line_id" not in lines.columns:
        lines["line_id"] = lines["department"].astype(str).str.title() + " " + lines["team"].astype(str)
    return lines


def estimate_shortfall(lines: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Estimate target and expected pieces for each line.

    Expected productivity is taken from an ``expected_productivity`` column if present
    (e.g. a live projection from the simulator). Otherwise it is estimated from the
    model's prediction: target plus the typical gap for that department when a line
    misses (``met_target_pred == 0``) or meets its target.

    Args:
        lines: One row per line with ``department``, ``team``, ``targeted_productivity``,
            ``smv``, ``no_of_workers``, ``over_time`` and either ``expected_productivity``
            or ``met_target_pred``.
        settings: Optimization settings (defaults to config).

    Returns:
        ``lines`` plus ``available_minutes``, ``target_pieces``, ``expected_pieces``,
        ``shortfall_pieces`` (0 if on track) and ``surplus_pieces`` (0 if behind).
    """
    settings = settings or get_settings()
    lines = add_line_ids(lines)

    if "expected_productivity" not in lines.columns:
        if "met_target_pred" not in lines.columns:
            raise ValueError("lines need 'expected_productivity' or 'met_target_pred'")
        gaps = settings["typical_gap"]
        outcome = lines["met_target_pred"].map({0: "missed", 1: "met"})
        typical = [gaps[str(dept)][result] for dept, result in zip(lines["department"], outcome)]
        lines["expected_productivity"] = lines["targeted_productivity"] + typical

    lines["available_minutes"] = lines["no_of_workers"] * settings["shift_minutes"] + lines["over_time"]
    lines["target_pieces"] = lines["targeted_productivity"] * lines["available_minutes"] / lines["smv"]
    lines["expected_pieces"] = lines["expected_productivity"] * lines["available_minutes"] / lines["smv"]
    gap = lines["expected_pieces"] - lines["target_pieces"]
    lines["shortfall_pieces"] = (-gap).clip(lower=0)
    lines["surplus_pieces"] = gap.clip(lower=0)
    return lines


def recommend_actions(lines: pd.DataFrame, settings: dict[str, Any] | None = None) -> RecommendationPlan:
    """Solve the operator / overtime reallocation problem for one day.

    Args:
        lines: Output of :func:`estimate_shortfall`, or raw lines it can be run on.
        settings: Optimization settings (defaults to config). Useful keys:
            ``max_overtime_minutes_per_line``, ``max_operators_moved_per_line``.

    Returns:
        A :class:`RecommendationPlan` with the actions and a before/after table.
    """
    from ortools.sat.python import cp_model

    settings = settings or get_settings()
    if "shortfall_pieces" not in lines.columns:
        lines = estimate_shortfall(lines, settings)
    lines = lines.reset_index(drop=True)

    shift = settings["shift_minutes"]
    step = settings["overtime_step_minutes"]
    max_steps = settings["max_overtime_minutes_per_line"] // step
    max_moved = settings["max_operators_moved_per_line"]

    # Pieces one operator makes in a full shift on each line, and pieces one
    # overtime step adds for the whole line
    rate = lines["expected_productivity"] / lines["smv"]  # pieces per worker-minute
    per_operator = rate * shift
    per_overtime_step = rate * step * lines["no_of_workers"]

    receivers = lines.index[lines["shortfall_pieces"] > 0].tolist()
    donors = lines.index[lines["surplus_pieces"] > 0].tolist()
    if not receivers:
        return _build_plan(lines, {}, {}, "NO_ACTION_NEEDED", settings, per_operator, per_overtime_step)

    model = cp_model.CpModel()

    # Decision variables
    moves = {
        (i, j): model.NewIntVar(0, min(max_moved, int(lines.at[i, "no_of_workers"]) - 1), f"move_{i}_{j}")
        for i in donors
        for j in receivers
        if lines.at[i, "department"] == lines.at[j, "department"]
    }
    overtime = {j: model.NewIntVar(0, max_steps, f"overtime_{j}") for j in receivers}

    def scaled(value: float) -> int:
        return int(round(value * SCALE))

    shortfalls = []
    for idx in lines.index:
        gained = sum(moves[i, j] * scaled(per_operator[j]) for (i, j) in moves if j == idx)
        lost = sum(moves[i, j] * scaled(per_operator[i]) for (i, j) in moves if i == idx)
        extra = overtime[idx] * scaled(per_overtime_step[idx]) if idx in overtime else 0
        output = scaled(lines.at[idx, "expected_pieces"]) + gained - lost + extra
        target = scaled(lines.at[idx, "target_pieces"])

        outgoing = [var for (i, _), var in moves.items() if i == idx]
        incoming = [var for (_, j), var in moves.items() if j == idx]
        if outgoing:
            # A lending line must still meet its own target
            model.Add(output >= target)
            model.Add(sum(outgoing) <= max_moved)
        if incoming:
            model.Add(sum(incoming) <= max_moved)
        if idx in receivers:
            short = model.NewIntVar(0, target, f"short_{idx}")
            model.Add(short >= target - output)
            shortfalls.append(short)

    move_cost = scaled(settings["move_cost_pieces"])
    overtime_cost = {
        j: scaled(settings["overtime_cost_pieces_per_worker_hour"] * lines.at[j, "no_of_workers"] * step / 60)
        for j in overtime
    }
    model.Minimize(
        sum(shortfalls)
        + sum(move_cost * var for var in moves.values())
        + sum(overtime_cost[j] * overtime[j] for j in overtime)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings["solver_time_limit_seconds"]
    solver.parameters.random_seed = 42
    solver.parameters.num_workers = 1  # deterministic results
    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Solver found no plan (status {solver.StatusName(result)})")

    chosen_moves = {key: solver.Value(var) for key, var in moves.items() if solver.Value(var) > 0}
    chosen_overtime = {j: solver.Value(var) * step for j, var in overtime.items() if solver.Value(var) > 0}
    return _build_plan(
        lines, chosen_moves, chosen_overtime, solver.StatusName(result), settings, per_operator, per_overtime_step
    )


def _build_plan(
    lines: pd.DataFrame,
    moves: dict[tuple[int, int], int],
    overtime_minutes: dict[int, int],
    status: str,
    settings: dict[str, Any],
    per_operator: pd.Series,
    per_overtime_step: pd.Series,
) -> RecommendationPlan:
    """Turn the solver's numbers into readable actions and a before/after table."""
    table = lines.copy()
    table["operators_in"] = 0
    table["operators_out"] = 0
    table["extra_overtime_minutes"] = 0
    table["expected_pieces_after"] = table["expected_pieces"]
    actions: list[Recommendation] = []

    for (i, j), count in moves.items():
        gain = count * per_operator[j]
        table.at[i, "operators_out"] += count
        table.at[j, "operators_in"] += count
        table.at[i, "expected_pieces_after"] -= count * per_operator[i]
        table.at[j, "expected_pieces_after"] += gain
        actions.append(
            Recommendation(
                line_id=table.at[j, "line_id"],
                action="move_operators",
                amount=count,
                source_line_id=table.at[i, "line_id"],
                expected_gain_pieces=round(gain, 1),
                reason=(
                    f"{table.at[j, 'line_id']} is expected to finish about "
                    f"{table.at[j, 'shortfall_pieces']:.0f} pieces short. "
                    f"{table.at[i, 'line_id']} is about {table.at[i, 'surplus_pieces']:.0f} pieces ahead "
                    f"and still meets its target after lending {count} operator(s)."
                ),
            )
        )

    step = settings["overtime_step_minutes"]
    for j, minutes in overtime_minutes.items():
        gain = minutes / step * per_overtime_step[j]
        table.at[j, "extra_overtime_minutes"] = minutes
        table.at[j, "expected_pieces_after"] += gain
        actions.append(
            Recommendation(
                line_id=table.at[j, "line_id"],
                action="add_overtime",
                amount=minutes,
                expected_gain_pieces=round(gain, 1),
                reason=(
                    f"{minutes} extra minutes for all {table.at[j, 'no_of_workers']:.0f} workers on "
                    f"{table.at[j, 'line_id']} adds about {gain:.0f} pieces."
                ),
            )
        )

    table["shortfall_pieces_after"] = (table["target_pieces"] - table["expected_pieces_after"]).clip(lower=0)
    for j in table.index[table["shortfall_pieces_after"] > 0.5]:
        remaining = table.at[j, "shortfall_pieces_after"]
        actions.append(
            Recommendation(
                line_id=table.at[j, "line_id"],
                action="unresolved",
                amount=round(remaining, 1),
                reason=(
                    f"About {remaining:.0f} pieces still short after the limits on overtime and moves. "
                    "Escalate to the production manager (e.g. adjust target or split the order)."
                ),
            )
        )

    return RecommendationPlan(
        actions=actions,
        lines=table,
        status=status,
        pieces_short_before=round(float(table["shortfall_pieces"].sum()), 1),
        pieces_short_after=round(float(table["shortfall_pieces_after"].sum()), 1),
        settings=settings,
    )


def recommend_for_rows(rows: pd.DataFrame, bundle: dict[str, Any] | None = None, **overrides: Any) -> RecommendationPlan:
    """Predict risk for each line with the saved model, then recommend actions.

    Args:
        rows: One row per line in the cleaned-data format (``actual_productivity`` not needed).
        bundle: Loaded model bundle; loaded from ``models/`` if not given.
        **overrides: Optimization settings to change, e.g. ``max_operators_moved_per_line=3``.
    """
    from loomiq.models.predict import load_model, predict_line

    bundle = bundle or load_model()
    predictions = predict_line(bundle, rows.drop(columns=["actual_productivity"], errors="ignore"))
    settings = get_settings(overrides)
    return recommend_actions(estimate_shortfall(predictions, settings), settings)


def plan_to_frame(plan: RecommendationPlan) -> pd.DataFrame:
    """Actions as a table, for the dashboard or a notebook."""
    return pd.DataFrame([vars(action) for action in plan.actions])


def main() -> None:
    """Command-line demo: recommend actions for one real day from the dataset."""
    import argparse

    from loomiq.data.clean import load_clean_productivity

    parser = argparse.ArgumentParser(description="Recommend fixes for one day of the UCI data.")
    parser.add_argument("--date", default="2015-03-09", help="date in YYYY-MM-DD format")
    args = parser.parse_args()

    df = load_clean_productivity()
    day = df[df["date"] == args.date]
    if day.empty:
        raise SystemExit(f"No data for {args.date}")

    plan = recommend_for_rows(day)
    print(f"\n{args.date}: {len(day)} lines | solver status: {plan.status}")
    print(f"Pieces short before: {plan.pieces_short_before:.0f} | after: {plan.pieces_short_after:.0f}\n")
    for action in plan.actions:
        print(f"- [{action.action}] {action.line_id}: {action.reason}")


if __name__ == "__main__":
    main()
