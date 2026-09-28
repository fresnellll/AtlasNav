from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlasnav.evaluation.evaluate import evaluate_suite
from atlasnav.io import atomic_json


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_live_evidence_funnel_and_report_inputs(tmp_path: Path) -> None:
    run = tmp_path / "run/7"
    judge = tmp_path / "judge/7"
    run.mkdir(parents=True)
    judge.mkdir(parents=True)
    _jsonl(run / "events.jsonl", [
        {"type": "input", "role": "user", "content": "route anchor D42 title"},
        {"type": "tool", "turn": 2, "name": "open", "arguments": {"handle": "D42"},
         "is_error": False, "output": "# D42 opened\nThe decisive answer is Cedar."},
    ])
    atomic_json(run / "result.json", {
        "query_id": "7", "terminal_valid": True, "turn_count": 3,
        "safe_release_sent": False,
        "agent_usage": {"cost_total": 0.4, "currency": "CNY"},
    })
    atomic_json(judge / "judge_result.json", {
        "query_id": "7", "is_correct": True,
        "judge_usage": {"cost_total": 0.01, "currency": "CNY"},
    })
    qrels = tmp_path / "qrels.jsonl"
    _jsonl(qrels, [{
        "query_id": "7", "answer_slots": [{"id": "A1", "required": True}],
        "spans": [{"span_id": "S1", "docid": "42", "slot_ids": ["A1"],
                   "quote": "The decisive answer is Cedar."}],
    }])
    output = tmp_path / "evaluation"
    report = evaluate_suite(
        run_directory=tmp_path / "run", judge_directory=tmp_path / "judge",
        output=output, qrels=qrels,
        turn_checkpoints=[2, 3], cost_checkpoints=[0.3, 0.4],
    )
    assert report["endpoint"]["correct"] == 1
    assert report["endpoint"]["recorded_online_cost"] == pytest.approx(0.41)
    evidence = report["evidence_blindness"]
    assert evidence["surface"]["realization_all"] == 1.0
    assert evidence["open"]["realization_all"] == 1.0
    assert evidence["locate"]["realization_all"] == 1.0
    assert report["passive_checkpoints"]["turn"][0]["strict_accuracy"] == 0.0
    assert report["passive_checkpoints"]["turn"][1]["strict_accuracy"] == 1.0


def test_mixed_currencies_are_not_summed(tmp_path: Path) -> None:
    run = tmp_path / "run/1"
    judge = tmp_path / "judge/1"
    run.mkdir(parents=True)
    judge.mkdir(parents=True)
    atomic_json(run / "result.json", {
        "query_id": "1", "terminal_valid": True, "turn_count": 1,
        "agent_usage": {"cost_total": 1.0, "currency": "CNY"},
    })
    atomic_json(judge / "judge_result.json", {
        "query_id": "1", "is_correct": True,
        "judge_usage": {"cost_total": 0.1, "currency": "USD"},
    })
    report = evaluate_suite(
        run_directory=tmp_path / "run", judge_directory=tmp_path / "judge",
        output=tmp_path / "evaluation",
    )
    assert report["endpoint"]["recorded_online_cost"] is None
    assert report["endpoint"]["mixed_currency_costs_not_summed"] is True
