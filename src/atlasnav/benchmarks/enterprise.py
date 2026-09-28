"""Prepare EnterpriseRAG-Bench with evaluator supervision isolated."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from atlasnav.io import sha256_file

from .common import clean, finalize_bundle, new_bundle, write_documents


def _resolve_sources(raw_root: Path) -> tuple[Path, Path]:
    pairs = [
        (raw_root / "documents/test.parquet", raw_root / "questions/test.parquet"),
        (raw_root / "data/documents/test.parquet", raw_root / "data/questions/test.parquet"),
    ]
    for documents, questions in pairs:
        if documents.is_file() and questions.is_file():
            return documents, questions
    raise FileNotFoundError(pairs[0][0])


def prepare_enterprise(
    *, output: Path, raw_root: Path, expected_documents: int | None = 511_962,
    expected_questions: int | None = 500, **_: Any,
) -> dict[str, Any]:
    documents_path, questions_path = _resolve_sources(raw_root.resolve())
    document_file = pq.ParquetFile(documents_path)
    question_file = pq.ParquetFile(questions_path)
    if expected_documents is not None and document_file.metadata.num_rows != expected_documents:
        raise ValueError(
            f"expected {expected_documents} enterprise documents, "
            f"found {document_file.metadata.num_rows}"
        )
    if expected_questions is not None and question_file.metadata.num_rows != expected_questions:
        raise ValueError(
            f"expected {expected_questions} enterprise questions, "
            f"found {question_file.metadata.num_rows}"
        )

    occurrences: dict[str, int] = {}
    canonical_by_raw: dict[str, list[str]] = {}

    def documents() -> Iterator[dict[str, str]]:
        required = {"doc_id", "source_type", "title", "content"}
        if not required.issubset(document_file.schema.names):
            raise ValueError(f"enterprise documents lack fields: {sorted(required)}")
        for batch in document_file.iter_batches(batch_size=2048, columns=sorted(required)):
            for row in batch.to_pylist():
                raw_docid = str(row["doc_id"] or "").strip()
                if not raw_docid:
                    raise ValueError("enterprise corpus contains an empty doc_id")
                occurrence = occurrences.get(raw_docid, 0) + 1
                occurrences[raw_docid] = occurrence
                docid = raw_docid if occurrence == 1 else f"{raw_docid}__dup{occurrence}"
                canonical_by_raw.setdefault(raw_docid, []).append(docid)
                source_type = clean(row.get("source_type", "unknown")).casefold() or "unknown"
                title = clean(row.get("title", ""))
                body = str(row.get("content") or "").replace("\x00", " ").strip()
                yield {
                    "docid": docid,
                    "text": "\n".join(value for value in (title, body) if value),
                    # The source family remains an identity cue while avoiding
                    # a dependency on a private production hostname.
                    "url": f"https://{source_type}.enterprise.invalid/{docid}",
                }

    with new_bundle(output) as temporary:
        count = write_documents(temporary / "documents.parquet", documents())
        questions: list[dict[str, Any]] = []
        scoring: list[dict[str, Any]] = []
        for batch in question_file.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                qid = str(row.get("question_id") or "").strip()
                query = clean(row.get("question", ""))
                answer = str(row.get("gold_answer") or "").strip()
                if not qid or not query or not answer:
                    raise ValueError(f"invalid enterprise question row: {qid!r}")
                raw_gold = [str(value) for value in (row.get("expected_doc_ids") or [])]
                used: dict[str, int] = {}
                gold_docids: list[str] = []
                for raw_docid in raw_gold:
                    position = used.get(raw_docid, 0)
                    choices = canonical_by_raw.get(raw_docid, [])
                    if position >= len(choices):
                        raise ValueError(f"unknown qrel occurrence: {qid}/{raw_docid}")
                    gold_docids.append(choices[position])
                    used[raw_docid] = position + 1
                questions.append({"query_id": qid, "query": query})
                scoring.append({
                    "query_id": qid, "query": query, "answer": answer,
                    "question_type": str(row.get("question_type") or ""),
                    "source_types": list(row.get("source_types") or []),
                    "gold_docids": gold_docids,
                    "answer_facts": [str(value) for value in (row.get("answer_facts") or [])],
                })
        return finalize_bundle(
            temporary, benchmark="enterpriserag-bench", documents=count,
            questions=questions, scoring=scoring,
            sources={
                "documents": {"file": "documents/test.parquet", "sha256": sha256_file(documents_path)},
                "questions": {"file": "questions/test.parquet", "sha256": sha256_file(questions_path)},
            },
            details={
                "task_type": "enterprise_qa", "duplicate_raw_document_ids":
                sum(count - 1 for count in occurrences.values() if count > 1),
                "duplicate_policy": "stable __dupN occurrence suffix",
                "official_submission_fields": ["question_id", "answer", "document_ids"],
            },
        )
