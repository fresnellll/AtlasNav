"""Outcome-blind passive turn/cost checkpoint reconstruction."""

from __future__ import annotations

from typing import Any, Iterable


def passive_checkpoints(
    rows: Iterable[dict[str, Any]],
    *,
    turn_checkpoints: list[int],
    cost_checkpoints: list[float],
) -> dict[str, Any]:
    """Score only answers that had already terminated by a checkpoint.

    This is a zero-API replay. It never invents an answer for an unfinished
    trajectory and is deliberately named *passive*. Active Safe Release at a
    historical boundary requires an additional model call and is a separate
    experiment rather than a retrospective arithmetic operation.
    """
    values = list(rows)
    total = len(values)

    def summarize(selected: list[dict[str, Any]], boundary: int | float, kind: str) -> dict[str, Any]:
        return {
            "checkpoint": boundary, "kind": kind, "queries": total,
            "completed_by_checkpoint": len(selected),
            "correct_by_checkpoint": sum(row.get("is_correct") is True for row in selected),
            "strict_accuracy": (
                sum(row.get("is_correct") is True for row in selected) / total if total else None
            ),
            "unfinished_counted_wrong": total - len(selected),
        }

    turn = [
        summarize([row for row in values if int(row.get("turn_count") or 0) <= point], point, "turn")
        for point in sorted(set(turn_checkpoints))
    ]
    cost = [
        summarize([
            row for row in values
            if float(row.get("recorded_agent_cost") or 0.0) <= point
        ], point, "recorded_agent_cost")
        for point in sorted(set(cost_checkpoints))
    ]
    return {
        "schema": "atlasnav_passive_checkpoint_replay_v1",
        "queries": total, "turn": turn, "cost": cost,
        "active_safe_release_calls_performed": 0,
        "unfinished_policy": "strictly_counted_wrong",
    }
