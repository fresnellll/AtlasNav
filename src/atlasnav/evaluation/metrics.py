"""Deterministic endpoint and Evidence Blindness aggregation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def endpoint_summary(
    rows: Iterable[dict[str, Any]],
    agent_currency: str | None = None,
    judge_currency: str | None = None,
    comparable_components: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Aggregate endpoint quality and recorded cost without mixing currencies.

    ``recorded_agent_cost`` and ``recorded_judge_cost`` are independent ledgers.
    A paper-comparable total is emitted only for components explicitly declared
    by the frozen package manifest and sharing one currency.  This matters when,
    for example, a USD agent run is evaluated by a CNY judge.
    """
    rows = list(rows)
    query_ids = [str(row["query_id"]) for row in rows]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query IDs")
    correct = sum(row.get("is_correct") is True for row in rows)
    total = len(rows)
    agent_cost = sum(float(row.get("recorded_agent_cost") or 0.0) for row in rows)
    judge_cost = sum(float(row.get("recorded_judge_cost") or 0.0) for row in rows)
    turns = sum(int(row.get("turn_count") or 0) for row in rows)
    terminal_invalid = sum(row.get("terminal_valid") is False for row in rows)
    lower, upper = wilson(correct, total)
    included = tuple(comparable_components or ("agent", "judge"))
    unknown_components = sorted(set(included) - {"agent", "judge"})
    if unknown_components:
        raise ValueError(f"unknown comparable cost components: {unknown_components}")
    if "agent" in included and agent_cost and not agent_currency:
        raise ValueError("agent cost currency is required for a comparable total")
    if "judge" in included and judge_cost and not judge_currency:
        raise ValueError("judge cost currency is required for a comparable total")
    currencies = {
        currency for component, currency in (
            ("agent", agent_currency), ("judge", judge_currency)
        ) if component in included and currency
    }
    if len(currencies) > 1:
        raise ValueError(
            "paper-comparable cost components use different currencies; "
            "declare a single-currency subset in the package manifest"
        )
    comparable_cost = (
        (agent_cost if "agent" in included else 0.0)
        + (judge_cost if "judge" in included else 0.0)
    )
    comparable_currency = next(iter(currencies), None)
    totals_by_currency: dict[str, float] = {}
    for value, currency in ((agent_cost, agent_currency), (judge_cost, judge_currency)):
        if currency:
            totals_by_currency[currency] = totals_by_currency.get(currency, 0.0) + value
    return {
        "schema": "atlasnav_endpoint_summary_v1",
        "queries": total,
        "correct": correct,
        "strict_accuracy": correct / total if total else None,
        "accuracy_percent": 100.0 * correct / total if total else None,
        "wilson_95": [lower, upper],
        "recorded_online_cost": comparable_cost,
        "paper_comparable_online_cost": comparable_cost,
        "paper_comparable_cost_components": list(included),
        "recorded_agent_cost": agent_cost,
        "recorded_agent_cost_currency": agent_currency,
        "recorded_judge_cost": judge_cost,
        "recorded_judge_cost_currency": judge_currency,
        "recorded_total_by_currency": totals_by_currency,
        "currency": comparable_currency,
        "turns": turns,
        "mean_turns": turns / total if total else None,
        "invalid_terminals": terminal_invalid,
    }


def _stage_values(row: dict[str, Any], stage: str) -> tuple[bool, bool, float]:
    prefix = {"surface": "support_surface", "open": "support_open", "locate": "answer_evidence"}[stage]
    any_value = bool(row[f"{prefix}_any"])
    all_value = bool(row[f"{prefix}_all"])
    if stage == "locate":
        recall = float(row["answer_slot_recall"])
    else:
        denominator = int(row.get("required_answer_slot_count") or 0)
        realized = int(row[f"{prefix}_slot_count"])
        recall = realized / denominator if denominator else 0.0
    return any_value, all_value, recall


def evidence_blindness_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("Evidence Blindness rows are empty")
    required_slots = sum(int(row.get("required_answer_slot_count") or 0) for row in rows)
    construction_realized = sum(int(row.get("construction_slot_count") or 0) for row in rows)
    construction_any = sum(bool(row.get("construction_any")) for row in rows)
    construction_all = sum(bool(row.get("construction_all")) for row in rows)
    output: dict[str, Any] = {
        "schema": "atlasnav_evidence_blindness_summary_v1",
        "queries": len(rows),
        "construction": {
            "any": 1.0 - construction_any / len(rows),
            "mean": 1.0 - construction_realized / required_slots if required_slots else 0.0,
            "all": 1.0 - construction_all / len(rows),
            "realization_any": construction_any / len(rows),
            "macro_recall": float(np.mean([
                int(row.get("construction_slot_count") or 0)
                / max(1, int(row.get("required_answer_slot_count") or 0)) for row in rows
            ])),
            "micro_recall": construction_realized / required_slots if required_slots else 0.0,
            "realization_all": construction_all / len(rows),
        },
    }
    for stage in ("surface", "open", "locate"):
        values = [_stage_values(row, stage) for row in rows]
        if stage == "surface":
            realized_slots = sum(int(row["support_surface_slot_count"]) for row in rows)
        elif stage == "open":
            realized_slots = sum(int(row["support_open_slot_count"]) for row in rows)
        else:
            realized_slots = sum(int(row["hit_answer_slot_count"]) for row in rows)
        stage_required_slots = sum(int(row.get("required_answer_slot_count") or 0) for row in rows)
        micro_recall = realized_slots / stage_required_slots if stage_required_slots else 0.0
        output[stage] = {
            "any": float(np.mean([not value[0] for value in values])),
            "mean": float(1.0 - np.mean([value[2] for value in values])),
            "all": float(np.mean([not value[1] for value in values])),
            "realization_any": float(np.mean([value[0] for value in values])),
            "macro_recall": float(np.mean([value[2] for value in values])),
            "micro_recall": float(micro_recall),
            "realization_all": float(np.mean([value[1] for value in values])),
        }
    output["surface_to_open_all_conversion"] = (
        sum(bool(row["support_surface_all"]) and bool(row["support_open_all"]) for row in rows)
        / max(1, sum(bool(row["support_surface_all"]) for row in rows))
    )
    output["open_to_locate_all_conversion"] = (
        sum(bool(row["support_open_all"]) and bool(row["answer_evidence_all"]) for row in rows)
        / max(1, sum(bool(row["support_open_all"]) for row in rows))
    )
    if all("is_correct" in row for row in rows):
        correct = sum(row.get("is_correct") is True for row in rows)
        located_all = sum(bool(row["answer_evidence_all"]) for row in rows)
        correct_after_locate_all = sum(
            bool(row["answer_evidence_all"]) and row.get("is_correct") is True for row in rows
        )
        output["closure"] = {
            "realization": correct / len(rows),
            "blindness": 1.0 - correct / len(rows),
            "located_all_to_correct_conversion": correct_after_locate_all / max(1, located_all),
        }
    return output


def paired_accuracy(left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]) -> dict[str, int]:
    left_by_id = {str(row["query_id"]): row.get("is_correct") is True for row in left}
    right_by_id = {str(row["query_id"]): row.get("is_correct") is True for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired result query sets differ")
    return {
        "both_correct": sum(left_by_id[qid] and right_by_id[qid] for qid in left_by_id),
        "left_only": sum(left_by_id[qid] and not right_by_id[qid] for qid in left_by_id),
        "right_only": sum(not left_by_id[qid] and right_by_id[qid] for qid in left_by_id),
        "both_wrong": sum(not left_by_id[qid] and not right_by_id[qid] for qid in left_by_id),
    }
