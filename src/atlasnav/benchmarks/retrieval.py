"""Prepare BEIR-format retrieval tasks, including TREC-COVID."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from atlasnav.io import sha256_file

from .common import clean, finalize_bundle, load_jsonl, new_bundle, write_documents


def _qrel_path(raw_root: Path, split: str) -> Path:
    candidates = [raw_root / "qrels" / f"{split}.tsv", raw_root / "qrels.tsv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[0])


def _stable_subset(qids: list[str], size: int | None, seed: str) -> set[str]:
    if size is None or size >= len(qids):
        return set(qids)
    ranked = sorted(qids, key=lambda qid: hashlib.sha256(f"{seed}\0{qid}".encode()).hexdigest())
    return set(ranked[:size])


def prepare_retrieval(
    *,
    adapter: str,
    output: Path,
    raw_root: Path,
    split: str = "test",
    sample_size: int | None = None,
    seed: str = "atlasnav-retrieval-v1",
    **_: Any,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    corpus_path = raw_root / "corpus.jsonl"
    query_path = raw_root / "queries.jsonl"
    qrel_path = _qrel_path(raw_root, split)
    for path in (corpus_path, query_path, qrel_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_queries: dict[str, str] = {}
    for row in load_jsonl(query_path):
        qid = str(row.get("_id", row.get("query_id", ""))).strip()
        query = clean(row.get("text", row.get("query", "")))
        if not qid or not query or qid in raw_queries:
            raise ValueError(f"invalid or duplicate retrieval query: {qid!r}")
        raw_queries[qid] = query

    qrels: dict[str, dict[str, int]] = {}
    with qrel_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"qrel file has no header: {qrel_path}")
        for row in reader:
            qid = str(row.get("query-id", row.get("query_id", ""))).strip()
            docid = str(row.get("corpus-id", row.get("doc_id", ""))).strip()
            score = int(row.get("score", row.get("relevance", 0)) or 0)
            if score > 0:
                qrels.setdefault(qid, {})[docid] = score
    eligible = sorted(set(raw_queries) & set(qrels))
    selected = _stable_subset(eligible, sample_size, seed)

    def documents() -> Any:
        for row in load_jsonl(corpus_path):
            docid = str(row.get("_id", row.get("docid", ""))).strip()
            title = clean(row.get("title", ""))
            body = str(row.get("text", "")).replace("\x00", " ").strip()
            if not docid:
                raise ValueError("retrieval corpus contains an empty document ID")
            yield {
                "docid": docid,
                "text": "\n".join(value for value in (title, body) if value),
                "url": str((row.get("metadata") or {}).get("url", ""))
                if isinstance(row.get("metadata"), dict) else "",
            }

    questions: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    for raw_qid in sorted(selected):
        qid = f"{adapter}:{raw_qid}"
        query = raw_queries[raw_qid]
        grades = dict(sorted(qrels[raw_qid].items()))
        questions.append({"query_id": qid, "query": query})
        scoring.append({
            "query_id": qid,
            "query": query,
            "answer": "",
            "source_query_id": raw_qid,
            "gold_docids": list(grades),
            "graded_relevance": grades,
        })

    with new_bundle(output) as temporary:
        document_count = write_documents(temporary / "documents.parquet", documents())
        return finalize_bundle(
            temporary,
            benchmark=adapter,
            documents=document_count,
            questions=questions,
            scoring=scoring,
            sources={
                "corpus": {"file": corpus_path.name, "sha256": sha256_file(corpus_path)},
                "queries": {"file": query_path.name, "sha256": sha256_file(query_path)},
                "qrels": {"file": f"qrels/{qrel_path.name}", "sha256": sha256_file(qrel_path)},
            },
            details={
                "task_type": "graded_retrieval",
                "split": split,
                "sample_size": len(questions),
                "selection_seed": seed if sample_size is not None else None,
                "primary_metrics": ["ndcg@10", "recall@10", "recall@100"],
            },
        )
