"""Factory assistant that answers questions like "Why is Sewing 7 behind?".

The assistant does not guess. Every question is sent together with a
**factory briefing** built from the dashboard's own numbers:
    * each line's status, output so far, projection and model risk score,
    * the top SHAP reasons behind each line's risk (Component 1),
    * today's events (breakdowns, style changes, applied plans),
    * the recommended plan (Component 2),
    * today's DHU, if defects have been scanned (Component 3).

Two modes:
    * **Claude** (``ANTHROPIC_API_KEY`` set in ``.env``): natural answers, grounded
      in the briefing by the system prompt.
    * **Offline** (no key): a rule-based answer built from the same briefing, so
      the page still works in a demo without an API key.
"""

import re
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are LoomIQ, an assistant for a garment factory production manager.

Answer using only the factory briefing in the user's message. It is live data from the
factory floor: line status, output, model risk scores, SHAP reasons, events and the
optimizer's recommended plan. If the briefing does not contain the answer, say so plainly.

How to answer:
- Lead with the direct answer in one or two sentences, then the supporting numbers.
- When explaining why a line is behind, combine its live output and events with the
  model's SHAP reasons, and mention the recommended fix if there is one.
- The model risk score is a ranking for "check these lines first", not a literal
  probability, so do not present it as a percentage chance.
- Use the factory's terms (SMV, target, overtime, operators) and keep answers short,
  in plain language a line supervisor would use."""


def _line_drivers_text(drivers: list[dict[str, Any]]) -> str:
    return "; ".join(f"{d['label']} = {d['value']} ({d['direction']})" for d in drivers)


def build_context(
    view: pd.DataFrame,
    drivers_by_line: dict[str, list[dict[str, Any]]],
    plan: Any | None = None,
    hour: int = 0,
    hours_per_shift: int = 8,
    date: Any = None,
    dhu: float | None = None,
    dhu_by_line: dict[str, float] | None = None,
) -> str:
    """Turn the dashboard's numbers into a compact text briefing.

    Args:
        view: Live view from :func:`loomiq.dashboard.live.build_live_view`.
        drivers_by_line: Top SHAP drivers per ``line_id`` (from ``explain_line``).
        plan: Recommended plan (:class:`~loomiq.optimization.recommender.RecommendationPlan`).
        hour: Hours of the shift already worked.
        hours_per_shift: Length of the shift in hours.
        date: Date being simulated.
        dhu: Today's defects per hundred units, if known.
        dhu_by_line: DHU per ``line_id`` for lines that have been scanned.

    Returns:
        A plain-text briefing the model (or the offline mode) can read.
    """
    when = f"{pd.Timestamp(date):%A %d %B %Y}, " if date is not None else ""
    lines = [
        f"FACTORY BRIEFING: {when}hour {hour} of {hours_per_shift}",
        f"Total: {view['pieces_so_far'].sum():,.0f} pieces so far; projected "
        f"{view['projected_pieces'].sum():,.0f} of {view['target_pieces'].sum():,.0f} target pieces.",
        f"Today's DHU (defects per 100 units): {dhu:.1f}" if dhu is not None else "Today's DHU: not measured yet.",
        "",
        "LINES (status | pieces so far | projected / target | model risk score | team size | SMV):",
    ]
    for row in view.sort_values("projected_share").itertuples():
        lines.append(
            f"- {row.line_id}: {row.status} | {row.pieces_so_far:,.0f} | {row.projected_pieces:,.0f} / "
            f"{row.target_pieces:,.0f} ({row.projected_share:.0%}) | risk {row.risk_probability:.2f} | "
            f"{row.no_of_workers:.1f} workers | SMV {row.smv:.2f}"
        )
        if row.line_id in drivers_by_line:
            lines.append(f"    SHAP reasons: {_line_drivers_text(drivers_by_line[row.line_id])}")
        if dhu_by_line and row.line_id in dhu_by_line:
            lines.append(f"    Quality: DHU {dhu_by_line[row.line_id]:.1f} from today's fabric scans")
        for event in row.events:
            lines.append(f"    Event: {event}")

    lines += ["", "RECOMMENDED PLAN:"]
    if plan is None or not plan.actions:
        lines.append("- No changes needed: all lines are projected to meet target.")
    else:
        lines.append(
            f"- Expected pieces short: {plan.pieces_short_before:,.0f} now, {plan.pieces_short_after:,.0f} with the plan."
        )
        lines += [f"- [{a.action}] {a.reason}" for a in plan.actions]
    return "\n".join(lines)


def find_line(question: str, line_ids: list[str]) -> str | None:
    """Find a line mentioned in a question, e.g. "line 7", "Sewing 7" or "finishing 10"."""
    text = question.lower()
    for line_id in sorted(line_ids, key=len, reverse=True):  # "Sewing 10" before "Sewing 1"
        if re.search(rf"\b{re.escape(line_id.lower())}\b", text):
            return line_id
    number = re.search(r"\b(?:line|team)\s*(\d+)\b", text)
    if number:
        matches = [line_id for line_id in line_ids if line_id.endswith(f" {number.group(1)}")]
        if len(matches) == 1:
            return matches[0]
    return None


def offline_answer(
    question: str,
    view: pd.DataFrame,
    drivers_by_line: dict[str, list[dict[str, Any]]],
    plan: Any | None = None,
) -> str:
    """Answer from the briefing with simple rules, for use without an API key."""
    line_id = find_line(question, view["line_id"].tolist())
    if line_id:
        row = view.set_index("line_id").loc[line_id]
        parts = [
            f"**{line_id}** is **{row['status'].lower()}**: projected {row['projected_pieces']:,.0f} "
            f"of {row['target_pieces']:,.0f} pieces ({row['projected_share']:.0%}), "
            f"with {row['pieces_so_far']:,.0f} made so far."
        ]
        raising = [d for d in drivers_by_line.get(line_id, []) if d["direction"] == "raises risk"]
        if raising:
            reasons = ", ".join(f"{d['label']} ({d['value']})" for d in raising[:3])
            parts.append(f"The model's main risk factors: {reasons}.")
        if row["events"]:
            parts.append("Events today: " + "; ".join(row["events"]) + ".")
        if plan is not None:
            fixes = [a.reason for a in plan.actions if a.line_id == line_id]
            if fixes:
                parts.append("Recommended: " + " ".join(fixes))
        return " ".join(parts)

    behind = view[view["status"] != "On track"].sort_values("projected_share")
    if behind.empty:
        summary = "All lines are projected to meet their targets."
    else:
        names = ", ".join(f"{r.line_id} ({r.projected_share:.0%})" for r in behind.itertuples())
        summary = f"{len(behind)} of {len(view)} lines need attention: {names}."
    if plan is not None and plan.actions:
        summary += (
            f" The recommended plan cuts the expected shortfall from {plan.pieces_short_before:,.0f} "
            f"to {plan.pieces_short_after:,.0f} pieces."
        )
    return summary + " Ask about a specific line (e.g. \"Why is Sewing 7 behind?\") for details."


class FactoryAssistant:
    """Answers questions with Claude, grounded in the factory briefing."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None, client: Any | None = None) -> None:
        """Set up the assistant.

        Args:
            model: Claude model ID; defaults to ``assistant.model`` in config.yaml.
            max_tokens: Response length cap; defaults to config.
            client: An ``anthropic.Anthropic`` client (created from ``.env`` if not given).
        """
        from loomiq.config import get_env, load_config

        settings = load_config()["assistant"]
        self.model = model or settings["model"]
        self.max_tokens = max_tokens or settings["max_tokens"]
        self.effort = settings["effort"]
        self.history_turns = settings["history_turns"]
        self.client = client
        if self.client is None and (get_env("ANTHROPIC_API_KEY") or get_env("ANTHROPIC_AUTH_TOKEN")):
            import anthropic

            self.client = anthropic.Anthropic()

    @property
    def is_online(self) -> bool:
        """True when Claude can be called (an API key or client is available)."""
        return self.client is not None

    def ask(self, question: str, context: str, history: list[dict[str, str]] | None = None) -> str:
        """Answer a question using the factory briefing.

        Args:
            question: The manager's question.
            context: Output of :func:`build_context` (the current factory state).
            history: Earlier chat turns as ``{"role", "content"}`` dicts (text only).
                Only the latest question carries the briefing, so old states don't pile up.

        Returns:
            The answer as plain text (Markdown).

        Raises:
            RuntimeError: If no client is configured.
        """
        if not self.is_online:
            raise RuntimeError("No API key: set ANTHROPIC_API_KEY in .env, or use offline_answer().")

        recent = (history or [])[-2 * self.history_turns:]
        messages = recent + [{"role": "user", "content": f"{context}\n\nQUESTION: {question}"}]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": self.effort},
            # If a safety filter declines, retry on a fallback model instead of failing
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "Sorry, I can't answer that one. Try asking about line status, risks or the plan."

        answer = "".join(block.text for block in response.content if block.type == "text").strip()
        if response.stop_reason == "max_tokens":
            answer += "\n\n_(Answer cut short.)_"
        return answer
