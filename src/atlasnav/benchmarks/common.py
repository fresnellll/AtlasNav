"""Shared deterministic output contract for benchmark adapters."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from atlasnav.io import atomic_json, atomic_jsonl, sha256_file


DOCUMENT_SCHEMA = pa.schema([
    pa.field("docid", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("url", pa.string(), nullable=False),
])


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            yield value


@contextmanager
def new_bundle(output: Path) -> Iterator[Path]:
    """Create an output atomically and never replace an existing bundle."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace benchmark bundle: {output}")
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale build directory requires review: {temporary}")
    temporary.mkdir(parents=True)
    try:
        yield temporary
        temporary.replace(output)
    except BaseException:
        # An incomplete tree contains no accepted research result. Cleaning
        # this process-owned temporary is safe and prevents accidental reuse.
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_documents(path: Path, rows: Iterable[dict[str, Any]], batch_size: int = 4096) -> int:
    writer = pq.ParquetWriter(path, DOCUMENT_SCHEMA, compression="zstd")
    seen: set[str] = set()
    pending: list[dict[str, str]] = []
    count = 0
    try:
        for raw in rows:
            docid = str(raw.get("docid") or "").strip()
            if not docid or docid in seen:
                raise ValueError(f"empty or duplicate document ID: {docid!r}")
            seen.add(docid)
            pending.append({
                "docid": docid,
                "text": str(raw.get("text") or "").replace("\x00", " ").strip(),
                "url": str(raw.get("url") or "").strip(),
            })
            count += 1
            if len(pending) >= batch_size:
                writer.write_table(pa.Table.from_pylist(pending, schema=DOCUMENT_SCHEMA))
                pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=DOCUMENT_SCHEMA))
    finally:
        writer.close()
    return count


def finalize_bundle(
    root: Path,
    *,
    benchmark: str,
    documents: int,
    questions: list[dict[str, Any]],
    scoring: list[dict[str, Any]],
    sources: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(questions) != len(scoring):
        raise ValueError("question/scoring row cardinality mismatch")
    qids = [str(row.get("query_id") or "") for row in questions]
    score_qids = [str(row.get("query_id") or "") for row in scoring]
    if not qids or any(not qid for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError("question IDs must be non-empty and unique")
    if qids != score_qids:
        raise ValueError("question and scoring rows must have identical ordered IDs")
    if any(set(row) - {"query_id", "query"} for row in questions):
        raise ValueError("Agent-visible questions may contain only query_id/query")
    atomic_jsonl(root / "questions.jsonl", questions)
    atomic_jsonl(root / "scoring.jsonl", scoring)
    manifest: dict[str, Any] = {
        "schema": "atlasnav_benchmark_bundle_v1",
        "benchmark": benchmark,
        "documents": documents,
        "questions": len(questions),
        "documents_file": "documents.parquet",
        "questions_file": "questions.jsonl",
        "scoring_file": "scoring.jsonl",
        "documents_sha256": sha256_file(root / "documents.parquet"),
        "questions_sha256": sha256_file(root / "questions.jsonl"),
        "scoring_sha256": sha256_file(root / "scoring.jsonl"),
        "agent_visible_question_fields": ["query_id", "query"],
        "evaluation_supervision_physically_separate": True,
        "construction_reads_scoring_file": False,
        "sources": sources,
    }
    manifest.update(details or {})
    atomic_json(root / "manifest.json", manifest)
    return manifest


def audit_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    questions = list(load_jsonl(root / str(manifest["questions_file"])))
    scoring = list(load_jsonl(root / str(manifest["scoring_file"])))
    documents = pq.ParquetFile(root / str(manifest["documents_file"])).metadata.num_rows
    checks = {
        "schema": manifest.get("schema") == "atlasnav_benchmark_bundle_v1",
        "documents": documents == manifest.get("documents"),
        "questions": len(questions) == len(scoring) == manifest.get("questions"),
        "ordered_query_ids": [row.get("query_id") for row in questions]
        == [row.get("query_id") for row in scoring],
        "agent_schema": all(set(row) == {"query_id", "query"} for row in questions),
        "documents_sha256": sha256_file(root / str(manifest["documents_file"]))
        == manifest.get("documents_sha256"),
        "questions_sha256": sha256_file(root / str(manifest["questions_file"]))
        == manifest.get("questions_sha256"),
        "scoring_sha256": sha256_file(root / str(manifest["scoring_file"]))
        == manifest.get("scoring_sha256"),
        "leakage_firewall": manifest.get("construction_reads_scoring_file") is False,
    }
    return {
        "schema": "atlasnav_benchmark_bundle_audit_v1",
        "benchmark": manifest.get("benchmark"),
        "passed": all(checks.values()),
        "checks": checks,
        "documents": documents,
        "questions": len(questions),
    }
