from __future__ import annotations

import pytest

from atlasnav.evaluation.metrics import endpoint_summary


ROWS = [{
    "query_id": "q1",
    "is_correct": True,
    "turn_count": 3,
    "recorded_agent_cost": 2.0,
    "recorded_judge_cost": 0.5,
}]


def test_same_currency_agent_and_judge_can_be_combined() -> None:
    result = endpoint_summary(
        ROWS, agent_currency="CNY", judge_currency="CNY",
        comparable_components=("agent", "judge"),
    )
    assert result["paper_comparable_online_cost"] == 2.5
    assert result["currency"] == "CNY"
    assert result["recorded_total_by_currency"] == {"CNY": 2.5}


def test_mixed_currency_judge_is_kept_separate() -> None:
    result = endpoint_summary(
        ROWS, agent_currency="USD", judge_currency="CNY",
        comparable_components=("agent",),
    )
    assert result["paper_comparable_online_cost"] == 2.0
    assert result["currency"] == "USD"
    assert result["recorded_total_by_currency"] == {"USD": 2.0, "CNY": 0.5}


def test_mixed_currency_total_is_rejected() -> None:
    with pytest.raises(ValueError, match="different currencies"):
        endpoint_summary(
            ROWS, agent_currency="USD", judge_currency="CNY",
            comparable_components=("agent", "judge"),
        )
