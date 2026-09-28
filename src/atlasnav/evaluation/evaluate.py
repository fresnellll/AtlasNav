"""Join live runs, judgments, Qrels, and trajectory-derived evidence metrics."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from atlasnav.evaluation.blindness import evaluate_trajectory, read_jsonl, write_jsonl
from atlasnav.evaluation.checkpoints import passive_checkpoints
from atlasnav.evaluation.metrics import endpoint_summary, evidence_blindness_summary
from atlasnav.io import atomic_json


def _corpus_docids(corpus: Path | None) -> set[str] | None:
    if corpus is None:
        return None
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    with gzip.open(corpus / str(manifest["catalog_file"]), "rt", encoding="utf-8") as stream:
        return {str(json.loads(line)["docid"]) for line in stream if line.strip()}


def evaluate_suite(
    *,
    run_directory: Path,
    judge_directory: Path,
    output: Path,
    qrels: Path | None = None,
    corpus: Path | None = None,
    turn_checkpoints: list[int] | None = None,
    cost_checkpoints: list[float] | None = None,
) -> dict[str, Any]:
    run_directory, judge_directory, output = (
        run_directory.resolve(), judge_directory.resolve(), output.resolve()
    )
    query_ids = sorted(
        path.name for path in run_directory.iterdir()
        if path.is_dir() and (path / "result.json").is_file()
    )
    rows: list[dict[str, Any]] = []
    for qid in query_ids:
        run = json.loads((run_directory / qid / "result.json").read_text(encoding="utf-8"))
        judge_path = judge_directory / qid / "judge_result.json"
        if not judge_path.is_file():
            continue
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        rows.append({
            "query_id": qid, "is_correct": judge.get("is_correct") is True,
            "terminal_valid": run.get("terminal_valid") is True,
            "turn_count": int(run.get("turn_count") or 0),
            "recorded_agent_cost": float((run.get("agent_usage") or {}).get("cost_total") or 0.0),
            "recorded_judge_cost": float((judge.get("judge_usage") or {}).get("cost_total") or 0.0),
            "agent_currency": (run.get("agent_usage") or {}).get("currency"),
            "judge_currency": (judge.get("judge_usage") or {}).get("currency"),
            "safe_release_sent": run.get("safe_release_sent") is True,
        })
    agent_currencies = {row["agent_currency"] for row in rows if row["agent_currency"]}
    judge_currencies = {row["judge_currency"] for row in rows if row["judge_currency"]}
    agent_currency = next(iter(agent_currencies)) if len(agent_currencies) == 1 else None
    judge_currency = next(iter(judge_currencies)) if len(judge_currencies) == 1 else None
    mixed_currency = any(
        row["judge_currency"] not in {None, row["agent_currency"]} for row in rows
    )
    if mixed_currency:
        # Never silently add different currencies. The endpoint table retains
        # each component; recorded_online_cost is intentionally omitted below.
        comparable_components = ("agent",)
    else:
        comparable_components = ("agent", "judge")
    endpoint = endpoint_summary(
        rows,
        agent_currency=agent_currency,
        judge_currency=judge_currency or agent_currency,
        comparable_components=comparable_components,
    )
    if mixed_currency:
        endpoint["recorded_online_cost"] = None
        endpoint["paper_comparable_online_cost"] = None
        endpoint["currency"] = None
        endpoint["mixed_currency_costs_not_summed"] = True
    evidence_summary = None
    if qrels is not None:
        qrel_rows = {str(row["query_id"]): row for row in read_jsonl(qrels.resolve())}
        docids = _corpus_docids(corpus.resolve() if corpus else None)
        evidence_rows = []
        correctness = {row["query_id"]: row["is_correct"] for row in rows}
        for qid in query_ids:
            if qid not in qrel_rows or qid not in correctness:
                continue
            value = evaluate_trajectory(qid, run_directory, qrel_rows[qid], docids)
            value["is_correct"] = correctness[qid]
            evidence_rows.append(value)
        write_jsonl(output / "evidence_blindness/per_query.jsonl", evidence_rows)
        evidence_summary = evidence_blindness_summary(evidence_rows)
        atomic_json(output / "evidence_blindness/summary.json", evidence_summary)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", rows)
    checkpoints = passive_checkpoints(
        rows,
        turn_checkpoints=turn_checkpoints or [15, 30, 60, 120, 300],
        cost_checkpoints=cost_checkpoints or [],
    )
    atomic_json(output / "checkpoints.json", checkpoints)
    report = {
        "schema": "atlasnav_live_evaluation_v1", "endpoint": endpoint,
        "evidence_blindness": evidence_summary,
        "passive_checkpoints": checkpoints,
        "run_directory": str(run_directory), "judge_directory": str(judge_directory),
        "queries_with_run_and_judgment": len(rows),
    }
    atomic_json(output / "summary.json", report)
    return report
