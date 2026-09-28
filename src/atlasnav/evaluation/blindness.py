"""Compute the five-stage Evidence Blindness funnel from live trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from atlasnav.io import stable_json


HANDLE_RE = re.compile(r"(?<![\w])D([^\s\t|:;,\]\[()]+)")
LINE_PREFIXES = (
    re.compile(r"^\s*\d{1,8}\s*[\t|:]\s*"),
    re.compile(r"^\s*=+\s*D[^:]+\s*:\s*"),
)


def normalize_visible(value: str) -> str:
    lines: list[str] = []
    for raw in unicodedata.normalize("NFKC", value).splitlines():
        line = raw
        for pattern in LINE_PREFIXES:
            line = pattern.sub("", line)
        lines.append(line)
    lowered = " ".join(lines).casefold()
    lowered = lowered.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", lowered).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _handles(value: str) -> set[str]:
    return {match.group(1).rstrip(".") for match in HANDLE_RE.finditer(value)}


def _slot_model(qrel: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    slots = {
        str(row["id"]): row for row in qrel.get("answer_slots") or []
        if isinstance(row, dict) and row.get("required") is not False
    }
    spans: list[dict[str, Any]] = []
    support: dict[str, set[str]] = {slot: set() for slot in slots}
    for source in qrel.get("spans") or []:
        if not isinstance(source, dict):
            continue
        slot_ids = sorted(set(map(str, source.get("slot_ids") or [])) & set(slots))
        docid = str(source.get("docid") or "")
        if not slot_ids or not docid or not str(source.get("quote") or "").strip():
            continue
        row = {**source, "slot_ids": slot_ids, "docid": docid}
        spans.append(row)
        for slot in slot_ids:
            support[slot].add(docid)
    return slots, spans, support


def evaluate_trajectory(
    query_id: str,
    run_directory: Path,
    qrel: dict[str, Any],
    corpus_docids: set[str] | None = None,
) -> dict[str, Any]:
    events = read_jsonl(run_directory / query_id / "events.jsonl")
    slots, spans, support = _slot_model(qrel)
    surface_docids: set[str] = set()
    opened_docids: set[str] = set()
    visible_open_fragments: list[tuple[str, str]] = []
    for event in events:
        if event.get("type") == "input" and event.get("role") == "user":
            surface_docids.update(_handles(str(event.get("content") or "")))
        if event.get("type") != "tool" or event.get("is_error") is True:
            continue
        output = str(event.get("output") or "")
        surface_docids.update(_handles(output))
        if event.get("name") == "open":
            handle = str((event.get("arguments") or {}).get("handle") or "")
            # Live event v1 stores call arguments on the assistant event. Recover
            # the canonical handle from the opening marker when needed.
            values = _handles(output)
            if handle.startswith("D"):
                values.add(handle[1:])
            for docid in values:
                opened_docids.add(docid)
                visible_open_fragments.append((docid, output))
    constructed_slots = {
        slot for slot, docids in support.items()
        if corpus_docids is None or bool(docids & corpus_docids)
    }
    surfaced_slots = {slot for slot, docids in support.items() if docids & surface_docids}
    opened_slots = {slot for slot, docids in support.items() if docids & opened_docids}
    span_hits: set[str] = set()
    located_slots: set[str] = set()
    relevant_fragments: set[tuple[int, str]] = set()
    support_fragments: set[tuple[int, str]] = set()
    for fragment_index, (docid, output) in enumerate(visible_open_fragments):
        normalized = normalize_visible(output)
        if any(docid in docids for docids in support.values()):
            support_fragments.add((fragment_index, docid))
        for span in spans:
            if span["docid"] != docid:
                continue
            if normalize_visible(str(span["quote"])) not in normalized:
                continue
            span_hits.add(str(span.get("span_id") or ""))
            located_slots.update(span["slot_ids"])
            relevant_fragments.add((fragment_index, docid))
    required = set(slots)
    reference_complete = bool(required) and all(support.get(slot) for slot in required)
    result = {
        "schema": "atlasnav_evidence_blindness_row_v1", "query_id": query_id,
        "reference_complete": reference_complete,
        "required_answer_slot_count": len(required),
        "construction_slot_count": len(constructed_slots),
        "construction_any": bool(constructed_slots),
        "construction_all": bool(required) and required <= constructed_slots,
        "support_surface_slot_count": len(surfaced_slots),
        "support_surface_any": bool(surfaced_slots),
        "support_surface_all": bool(required) and required <= surfaced_slots,
        "support_open_slot_count": len(opened_slots),
        "support_open_any": bool(opened_slots),
        "support_open_all": bool(required) and required <= opened_slots,
        "hit_answer_slot_count": len(located_slots),
        "answer_slot_recall": len(located_slots) / len(required) if required else 0.0,
        "answer_evidence_any": bool(located_slots),
        "answer_evidence_all": reference_complete and required <= located_slots,
        "support_document_fragment_exposure_count": len(support_fragments),
        "evidence_bearing_fragment_count": len(relevant_fragments),
        "evidence_fragment_precision": (
            len(relevant_fragments) / len(support_fragments) if support_fragments else None
        ),
        "surface_docids": sorted(surface_docids), "opened_docids": sorted(opened_docids),
        "hit_span_ids": sorted(value for value in span_hits if value),
        "missing_answer_slot_ids": sorted(required - located_slots),
        "visibility_scoring_reads_method_answer_or_correctness": False,
    }
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(stable_json(row) + "\n")
    temporary.replace(path)
