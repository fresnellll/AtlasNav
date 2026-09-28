"""EnterpriseRAG-Bench evaluation and document selection.

The evaluator has two deliberately separate modes.  ``evaluate_frozen``
recomputes the official metrics from stored answers without an API call.
``evaluate_with_selector`` additionally applies the official document
selection action with an OpenAI-compatible LLM API.  The latter never changes
the frozen answers or answer judgements.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import re
import statistics
from typing import Any

import httpx


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(values: Any) -> set[str]:
    return {str(value).casefold() for value in (values or [])}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall = [row["document_recall_pct"] for row in rows if row["document_recall_pct"] is not None]
    invalid = [row["invalid_extra_docs"] for row in rows if row["invalid_extra_docs"] is not None]
    return {
        "questions": len(rows),
        "correctness_percent": round(100 * statistics.mean(bool(row["answer_correct"]) for row in rows), 2) if rows else 0.0,
        "completeness_percent": round(statistics.mean(float(row["completeness_pct"]) for row in rows), 2) if rows else 0.0,
        "overall": round(statistics.mean(float(row["completeness_pct"]) if row["answer_correct"] else 0.0 for row in rows), 2) if rows else 0.0,
        "document_recall_percent": round(statistics.mean(recall), 2) if recall else None,
        "invalid_extra_documents": round(statistics.mean(invalid), 2) if invalid else None,
    }


def _breakdowns(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("question_type") or "unknown"), []).append(row)
        for source in row.get("source_types") or []:
            by_source.setdefault(str(source), []).append(row)

    def pack(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {name: {"count": len(values), **_summary(values)} for name, values in sorted(groups.items())}

    return pack(by_type), pack(by_source)


def evaluate_frozen(*, questions: Path, answers: Path, results: Path, output: Path, selection_results: Path | None = None) -> dict[str, Any]:
    """Recompute official metrics from frozen EnterpriseRAG records."""
    question_rows = {str(row["question_id"]): row for row in _jsonl(questions)}
    answer_rows = {str(row["question_id"]): row for row in _jsonl(answers)}
    frozen = json.loads(results.read_text(encoding="utf-8"))
    selection = json.loads(selection_results.read_text(encoding="utf-8")) if selection_results else None
    judged = {str(row["question_id"]): row for row in frozen["questions"]}
    selection_rows = {str(row["question_id"]): row for row in (selection or {}).get("questions", [])}
    if set(question_rows) != set(answer_rows) or set(question_rows) != set(judged):
        raise ValueError("question, answer, and result IDs do not match")
    rows: list[dict[str, Any]] = []
    for qid in sorted(question_rows):
        question = question_rows[qid]
        answer = answer_rows[qid]
        expected = _ids(question.get("expected_doc_ids"))
        selected = _ids(answer.get("document_ids"))
        valid = _ids(question.get("valid_doc_ids"))
        rows.append({
            "question_id": qid,
            "question_type": question.get("question_type"),
            "source_types": question.get("source_types") or [],
            "answer_correct": bool(judged[qid]["answer_correct"]),
            "completeness_pct": float(judged[qid]["completeness_pct"]),
            "document_recall_pct": (selection_rows[qid].get("document_recall_pct") if qid in selection_rows else round(100 * len(selected & expected) / len(expected), 2) if expected else None),
            "invalid_extra_docs": (selection_rows[qid].get("invalid_extra_docs") if qid in selection_rows else len((selected - expected) - valid) if expected else None),
        })
    output.mkdir(parents=True, exist_ok=True)
    by_type, by_source = _breakdowns(rows)
    report = {"schema": "atlasnav_enterpriserag_evaluation_v1", "mode": "frozen_with_official_selection" if selection_results else "frozen", "aggregate": _summary(rows), "by_question_type": by_type, "by_source_type": by_source, "questions": rows}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.casefold()))


def _select_prompt(question: str, answer: str, candidates: list[dict[str, str]]) -> str:
    items = "\n".join(f"- {row['document_id']} | {row.get('title') or '[title unavailable]'}" for row in candidates)
    return (
        "Question:\n" + question + "\n\nFinal answer:\n" + answer +
        "\n\nCandidate documents surfaced or opened by the frozen trajectory:\n" + items
    )


SELECTOR_SYSTEM = (
    "You reconstruct the official EnterpriseRAG document-selection action after a frozen agent trajectory. "
    "Select only candidate documents. Return the smallest set that materially supports the final answer; "
    "prefer recall when multiple documents are genuinely necessary. Do not invent IDs. "
    "Return only JSON: {\"document_ids\":[\"dsid_...\"]}."
)


def _call_selector(client: httpx.Client, *, base_url: str, model: str, question: str, answer: str, candidates: list[dict[str, str]]) -> list[str]:
    response = client.post(base_url.rstrip("/") + "/chat/completions", json={
        "model": model,
        "messages": [{"role": "system", "content": SELECTOR_SYSTEM}, {"role": "user", "content": _select_prompt(question, answer, candidates)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    })
    response.raise_for_status()
    body = response.json()
    content = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("selector returned no JSON object")
    selected = json.loads(match.group(0)).get("document_ids")
    allowed = {row["document_id"] for row in candidates}
    if not isinstance(selected, list):
        raise ValueError("selector JSON has no document_ids list")
    return list(dict.fromkeys(str(value).casefold() for value in selected if str(value).casefold() in allowed))


def evaluate_with_selector(*, questions: Path, answers: Path, results: Path, audit: Path, catalog: Path, output: Path, base_url: str, api_key: str, model: str, workers: int = 8, candidate_limit: int = 40) -> dict[str, Any]:
    """Reconstruct document selection through an OpenAI-compatible LLM API."""
    question_rows = {str(row["question_id"]): row for row in _jsonl(questions)}
    answer_rows = {str(row["question_id"]): row for row in _jsonl(answers)}
    audit_rows = {str(row["question_id"]): row for row in _jsonl(audit)}
    frozen = json.loads(results.read_text(encoding="utf-8"))
    judged = {str(row["question_id"]): row for row in frozen["questions"]}
    catalog_rows = {str(row["docid"]).casefold(): str(row.get("title") or "") for row in _jsonl(catalog)}
    ids = set(question_rows) & set(answer_rows) & set(audit_rows) & set(judged)
    if ids != set(question_rows) or ids != set(answer_rows):
        raise ValueError("question, answer, audit, and result IDs do not match")

    def prepare(qid: str) -> tuple[str, list[dict[str, str]]]:
        audit_row = audit_rows[qid]
        found = list(dict.fromkeys(_ids([*(audit_row.get("surface_document_ids") or []), *(audit_row.get("open_document_ids") or []), *(audit_row.get("answer_cited_document_ids") or [])])))
        question_words, answer_words = _tokens(question_rows[qid].get("question", "")), _tokens(answer_rows[qid].get("answer", ""))
        scored = []
        for docid in found:
            title = catalog_rows.get(docid, "")
            title_words = _tokens(title)
            score = len(title_words & answer_words) + 0.4 * len(title_words & question_words)
            if docid in _ids(audit_row.get("answer_cited_document_ids")): score += 3
            scored.append((score, docid, title))
        scored.sort(reverse=True)
        return qid, [{"document_id": docid, "title": title} for _, docid, title in scored[:candidate_limit]]

    prepared = dict(prepare(qid) for qid in sorted(ids))
    selected: dict[str, list[str]] = {}
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=180.0) as client:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_call_selector, client, base_url=base_url, model=model, question=question_rows[qid]["question"], answer=answer_rows[qid].get("answer", ""), candidates=candidates): qid for qid, candidates in prepared.items()}
            for future in as_completed(futures):
                selected[futures[future]] = future.result()
    rows = []
    for qid in sorted(ids):
        question = question_rows[qid]; expected = _ids(question.get("expected_doc_ids")); valid = _ids(question.get("valid_doc_ids")); chosen = set(selected[qid])
        rows.append({"question_id": qid, "question_type": question.get("question_type"), "source_types": question.get("source_types") or [], "answer_correct": bool(judged[qid]["answer_correct"]), "completeness_pct": float(judged[qid]["completeness_pct"]), "selected_document_ids": selected[qid], "document_recall_pct": round(100 * len(chosen & expected) / len(expected), 2) if expected else None, "invalid_extra_docs": len((chosen - expected) - valid) if expected else None})
    output.mkdir(parents=True, exist_ok=True)
    by_type, by_source = _breakdowns(rows)
    report = {"schema": "atlasnav_enterpriserag_evaluation_v1", "mode": "official_selector", "model": model, "base_url": base_url, "candidate_limit": candidate_limit, "aggregate": _summary(rows), "by_question_type": by_type, "by_source_type": by_source, "questions": rows}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
