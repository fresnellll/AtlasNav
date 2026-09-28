"""Stable auxiliary-benchmark exports and deterministic official metrics."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

from atlasnav.io import atomic_json, atomic_jsonl

from .common import load_jsonl


FINAL_SECTION = re.compile(r"Relevant Documents.*?(1\..*?)(?:\n\n|\Z)", re.I | re.S)


def _runs(run_directory: Path) -> Iterator[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    for directory in sorted(run_directory.iterdir(), key=lambda path: path.name):
        result = directory / "result.json"
        if not directory.is_dir() or not result.is_file():
            continue
        events = []
        event_path = directory / "events.jsonl"
        if event_path.is_file():
            events = list(load_jsonl(event_path))
        yield directory.name, json.loads(result.read_text(encoding="utf-8")), events


def _opened(events: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "tool" or event.get("name") != "open" or event.get("is_error") is True:
            continue
        output = str(event.get("output") or "")
        handle = str((event.get("arguments") or {}).get("handle") or "")
        if not re.search(rf"#\s+{re.escape(handle)}\s+opened\b", output, re.I):
            continue
        docid = handle[1:] if handle.startswith("D") else handle
        if docid and docid not in seen:
            seen.add(docid)
            result.append(docid)
    return result


def export_predictions(bundle: Path, run_directory: Path, output: Path) -> dict[str, Any]:
    """Export answer plus truly opened canonical IDs, without reading Gold."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    query_ids = {str(row["query_id"]) for row in load_jsonl(bundle / "questions.jsonl")}
    rows: list[dict[str, Any]] = []
    for qid, result, events in _runs(run_directory.resolve()):
        if qid not in query_ids:
            continue
        opened = _opened(events)
        official_ids = [docid.split("__dup", 1)[0] for docid in opened]
        official_ids = list(dict.fromkeys(official_ids))
        rows.append({
            "question_id": qid, "answer": str(result.get("final_text") or "").strip(),
            "document_ids": official_ids, "canonical_open_document_ids": opened,
            "terminal_valid": result.get("terminal_valid") is True,
        })
    rows.sort(key=lambda row: row["question_id"])
    output = output.resolve()
    atomic_jsonl(output, rows)
    report = {
        "schema": "atlasnav_benchmark_predictions_v1", "benchmark": manifest["benchmark"],
        "questions_in_bundle": manifest["questions"], "predictions": len(rows),
        "selection_reads_scoring_file": False, "document_policy": "successful canonical open only",
    }
    atomic_json(output.with_suffix(".manifest.json"), report)
    return report


def _retrieved(text: str) -> list[str]:
    match = FINAL_SECTION.search(text.replace("\\n", "\n"))
    if match is None:
        return []
    result: list[str] = []
    for line in match.group(1).splitlines():
        value = re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", line).strip().strip("`'\"")
        value = re.sub(r"^docs/", "", value)
        if value.startswith("D") and value.endswith(".txt"):
            value = value[1:-4]
        if value and value not in result:
            result.append(value)
    return result


def _ndcg(retrieved: list[str], relevance: dict[str, float], k: int = 10) -> float:
    positive = {docid: float(value) for docid, value in relevance.items() if float(value) > 0}
    dcg = sum(positive.get(docid, 0.0) / math.log2(rank + 2)
              for rank, docid in enumerate(retrieved[:k]))
    ideal = sorted(positive.values(), reverse=True)[:k]
    idcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _fanout_atoms(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [atom for key, item in value.items() for atom in _fanout_atoms(key) + _fanout_atoms(item)]
    if isinstance(value, list):
        return [atom for item in value for atom in _fanout_atoms(item)]
    if isinstance(value, bool):
        return ["yes" if value else "no"]
    return [str(value)]


def _fanout_normalize(value: object) -> str:
    try:
        import ftfy
        import spacy
    except ImportError as error:
        raise RuntimeError("install atlasnav[benchmarks] for FanOutQA official string scoring") from error
    text = ftfy.fix_text(str(value).lower())
    text = re.sub(r"(\d+,)+\d+(\.\d+)?", lambda match: match[0].replace(",", ""), text)
    if not hasattr(_fanout_normalize, "nlp"):
        try:
            setattr(_fanout_normalize, "nlp", spacy.load("en_core_web_sm"))
        except OSError as error:
            raise RuntimeError("install spaCy model en_core_web_sm for exact FanOutQA scoring") from error
    text = " ".join(token.lemma_ for token in getattr(_fanout_normalize, "nlp")(text))
    return re.sub(r"\s+", " ", re.sub(r"[,.?!:;]", "", text)).strip()


def _fanout_match(reference: Any, candidate: str) -> tuple[bool, float]:
    """Match the public FanOutQA loose/strict recursion exactly."""
    if isinstance(reference, list):
        missing = sum(not _fanout_match(value, candidate)[0] for value in reference)
        return missing == 0, (len(reference) - missing) / len(reference) if reference else 1.0
    if isinstance(reference, dict):
        values = list(reference.keys()) + list(reference.values())
        missing = sum(not _fanout_match(value, candidate)[0] for value in values)
        return missing == 0, (len(values) - missing) / len(values) if values else 1.0
    primitive = "yes" if reference is True else "no" if reference is False else reference
    normalized = _fanout_normalize(primitive)
    found = bool(re.search(rf"\b{re.escape(normalized)}\b", candidate))
    return found, 1.0 if found else 0.0


def score_official(bundle: Path, run_directory: Path, output: Path) -> dict[str, Any]:
    bundle, run_directory = bundle.resolve(), run_directory.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    scoring = {str(row["query_id"]): row for row in load_jsonl(bundle / "scoring.jsonl")}
    runs = {qid: result for qid, result, _ in _runs(run_directory)}
    per_query: list[dict[str, Any]] = []
    task_type = str(manifest.get("task_type") or "")
    if task_type == "graded_retrieval":
        for qid, row in scoring.items():
            documents = _retrieved(str(runs.get(qid, {}).get("final_text") or ""))
            relevance = {str(key): float(value) for key, value in row["graded_relevance"].items()}
            documents = [
                value[:-4] if value.endswith(".txt") and value[:-4] in relevance else value
                for value in documents
            ]
            relevant = set(relevance)
            top10 = documents[:10]
            per_query.append({
                "query_id": qid, "ndcg_at_10": _ndcg(documents, relevance),
                "recall_at_10": len(set(top10) & relevant) / len(relevant) if relevant else 0.0,
                "recall_at_100": len(set(documents[:100]) & relevant) / len(relevant) if relevant else 0.0,
                "retrieved_docids": documents,
            })
        summary = {
            "ndcg_at_10": sum(row["ndcg_at_10"] for row in per_query) / len(per_query),
            "recall_at_10": sum(row["recall_at_10"] for row in per_query) / len(per_query),
            "recall_at_100": sum(row["recall_at_100"] for row in per_query) / len(per_query),
        }
    elif task_type == "fan_out_qa":
        for qid, row in scoring.items():
            answer = str(runs.get(qid, {}).get("final_text") or "")
            normalized = _fanout_normalize(answer)
            strict, loose = _fanout_match(row["answer"], normalized)
            per_query.append({"query_id": qid, "loose_accuracy": loose,
                              "strict_correct": strict})
        summary = {
            "loose_accuracy": sum(row["loose_accuracy"] for row in per_query) / len(per_query),
            "strict_accuracy": sum(row["strict_correct"] for row in per_query) / len(per_query),
        }
    else:
        raise ValueError(f"no deterministic official scorer for task type {task_type!r}")
    report = {
        "schema": "atlasnav_official_metric_report_v1", "benchmark": manifest["benchmark"],
        "questions": len(per_query), "metrics": summary,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output / "per_query.jsonl", per_query)
    atomic_json(output / "summary.json", report)
    return report
