"""Tests for loomiq.assistant.llm_assistant.

Claude is replaced by a fake client, so no API key or network is needed.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from loomiq.assistant import llm_assistant as la
from loomiq.optimization.recommender import Recommendation, RecommendationPlan


@pytest.fixture
def view() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line_id": ["Sewing 7", "Sewing 1", "Finishing 10"],
            "status": ["Behind", "On track", "At risk"],
            "pieces_so_far": [300.0, 500.0, 400.0],
            "projected_pieces": [600.0, 1000.0, 1150.0],
            "target_pieces": [800.0, 990.0, 1240.0],
            "projected_share": [0.75, 1.01, 0.93],
            "risk_probability": [0.95, 0.10, 0.60],
            "no_of_workers": [55.0, 59.0, 8.0],
            "smv": [30.48, 26.82, 2.90],
            "events": [["Hour 2: Machine breakdown: 10 workers idle for 60 min"], [], []],
        }
    )


@pytest.fixture
def drivers() -> dict:
    return {
        "Sewing 7": [
            {"label": "SMV (garment complexity)", "value": 30.48, "direction": "raises risk"},
            {"label": "team size", "value": 55.0, "direction": "lowers risk"},
        ]
    }


@pytest.fixture
def plan() -> RecommendationPlan:
    return RecommendationPlan(
        actions=[Recommendation("Sewing 7", "add_overtime", 60, reason="60 extra minutes on Sewing 7 adds about 90 pieces.")],
        lines=pd.DataFrame(),
        status="OPTIMAL",
        pieces_short_before=290,
        pieces_short_after=110,
    )


def test_context_contains_the_facts(view, drivers, plan) -> None:
    context = la.build_context(view, drivers, plan, hour=3, date="2015-03-09", dhu=2.5)
    assert "hour 3 of 8" in context and "Monday 09 March 2015" in context
    assert "Sewing 7: Behind" in context and "risk 0.95" in context
    assert "SMV (garment complexity) = 30.48 (raises risk)" in context
    assert "Machine breakdown" in context
    assert "60 extra minutes on Sewing 7" in context
    assert "DHU (defects per 100 units): 2.5" in context
    # Worst line first
    assert context.index("Sewing 7") < context.index("Finishing 10") < context.index("Sewing 1")


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Why is Sewing 7 behind?", "Sewing 7"),
        ("what about finishing 10", "Finishing 10"),
        ("How is line 1 doing?", "Sewing 1"),
        ("Which lines are late?", None),
    ],
)
def test_find_line(question, expected) -> None:
    assert la.find_line(question, ["Sewing 7", "Sewing 1", "Finishing 10"]) == expected


def test_offline_answer_for_a_line(view, drivers, plan) -> None:
    answer = la.offline_answer("Why is Sewing 7 behind?", view, drivers, plan)
    assert "Sewing 7" in answer and "behind" in answer
    assert "SMV" in answer and "Machine breakdown" in answer and "60 extra minutes" in answer


def test_offline_answer_for_the_factory(view, drivers, plan) -> None:
    answer = la.offline_answer("Which lines need attention?", view, drivers, plan)
    assert "2 of 3 lines" in answer and "Sewing 7 (75%)" in answer


class FakeMessages:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Sewing 7 is behind.")],
        )


def make_assistant(stop_reason: str = "end_turn") -> tuple[la.FactoryAssistant, FakeMessages]:
    messages = FakeMessages(stop_reason)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return la.FactoryAssistant(client=client), messages


def test_ask_sends_briefing_and_settings() -> None:
    assistant, messages = make_assistant()
    history = [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"} for i in range(30)]

    assert assistant.is_online
    assert assistant.ask("Why is Sewing 7 behind?", "BRIEFING", history) == "Sewing 7 is behind."

    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == la.SYSTEM_PROMPT
    assert call["fallbacks"] == "default"
    assert "temperature" not in call
    # Only recent turns are kept, and only the latest question carries the briefing
    assert len(call["messages"]) == 2 * assistant.history_turns + 1
    assert call["messages"][-1]["content"].startswith("BRIEFING") and call["messages"][-1]["content"].endswith("Why is Sewing 7 behind?")


def test_ask_handles_refusal() -> None:
    assistant, _ = make_assistant(stop_reason="refusal")
    assert "can't answer" in assistant.ask("anything", "BRIEFING")


def test_offline_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assistant = la.FactoryAssistant()
    assert not assistant.is_online
    with pytest.raises(RuntimeError):
        assistant.ask("hello", "BRIEFING")
