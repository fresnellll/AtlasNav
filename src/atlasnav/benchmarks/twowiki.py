"""Build the shared-corpus 2WikiMultiHopQA evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from atlasnav.io import sha256_file

from .common import clean, finalize_bundle, new_bundle, write_documents


QUESTION_TYPES = ("comparison", "inference", "compositional", "bridge_comparison")


def _parse(value: object, field: str, qid: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {field} JSON for {qid}") from error


def _document(title: object, sentences: list[object]) -> dict[str, Any]:
    normalized_title = clean(title)
    normalized_sentences = [clean(sentence) for sentence in sentences]
    normalized_sentences = [sentence for sentence in normalized_sentences if sentence]
    material = normalized_title + "\0" + "\n".join(normalized_sentences)
    docid = "W2" + hashlib.sha256(material.encode()).hexdigest()[:24]
    text = "\n".join(
        [f"Title: {normalized_title}"]
        + [f"[S{index}] {sentence}" for index, sentence in enumerate(normalized_sentences)]
    )
    return {"docid": docid, "title": normalized_title, "sentences": normalized_sentences,
            "text": text, "url": ""}


def prepare_2wiki(
    *,
    output: Path,
    source: Path,
    questions_per_type: int = 100,
    seed: str = "atlasnav-2wiki-global-v1",
    **_: Any,
) -> dict[str, Any]:
    source = source.resolve()
    rows = pq.read_table(source).to_pylist()
    required = {"_id", "type", "question", "context", "supporting_facts", "answer"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"2Wiki source lacks required fields: {sorted(required)}")

    documents: dict[str, dict[str, Any]] = {}
    contexts_by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in QUESTION_TYPES}
    for row in rows:
        raw_qid = str(row["_id"])
        question_type = str(row["type"])
        if question_type not in groups:
            raise ValueError(f"unknown 2Wiki question type: {question_type}")
        groups[question_type].append(row)
        per_title: dict[str, dict[str, Any]] = {}
        for title, sentences in _parse(row["context"], "context", raw_qid):
            document = _document(title, list(sentences))
            prior = documents.get(document["docid"])
            if prior is not None and prior != document:
                raise RuntimeError(f"document hash collision: {document['docid']}")
            documents[document["docid"]] = document
            if document["title"] in per_title and per_title[document["title"]] != document:
                raise ValueError(f"conflicting title in question {raw_qid}: {document['title']}")
            per_title[document["title"]] = document
        contexts_by_qid[raw_qid] = per_title

    selected: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPES:
        ranked = sorted(groups[question_type], key=lambda row: (
            hashlib.sha256(f"{seed}\0{question_type}\0{row['_id']}".encode()).hexdigest(),
            str(row["_id"]),
        ))
        if len(ranked) < questions_per_type:
            raise ValueError(f"not enough {question_type} questions")
        selected.extend(ranked[:questions_per_type])

    questions: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    for row in selected:
        raw_qid = str(row["_id"])
        qid = f"2wiki:{raw_qid}"
        query = clean(row["question"])
        support: list[dict[str, Any]] = []
        for title, sentence_id in _parse(row["supporting_facts"], "supporting_facts", raw_qid):
            document = contexts_by_qid[raw_qid].get(clean(title))
            if document is None:
                raise ValueError(f"supporting title is absent for {raw_qid}: {title}")
            index = int(sentence_id)
            if not 0 <= index < len(document["sentences"]):
                raise ValueError(f"support sentence is out of range for {raw_qid}: {index}")
            support.append({
                "docid": document["docid"], "title": document["title"],
                "sentence_id": index, "sentence": document["sentences"][index],
            })
        questions.append({"query_id": qid, "query": query})
        scoring.append({
            "query_id": qid, "query": query, "answer": clean(row["answer"]),
            "source_query_id": raw_qid, "question_type": str(row["type"]),
            "gold_docids": sorted({item["docid"] for item in support}),
            "supporting_facts": support,
            "evidences": _parse(row.get("evidences", []), "evidences", raw_qid),
        })

    with new_bundle(output) as temporary:
        count = write_documents(
            temporary / "documents.parquet",
            ({key: row[key] for key in ("docid", "text", "url")}
             for row in sorted(documents.values(), key=lambda item: item["docid"])),
        )
        return finalize_bundle(
            temporary, benchmark="2wiki-global", documents=count,
            questions=questions, scoring=scoring,
            sources={"dev": {"file": source.name, "sha256": sha256_file(source)}},
            details={
                "task_type": "multi_hop_qa", "source_questions": len(rows),
                "selection_seed": seed, "questions_per_type": questions_per_type,
                "type_counts": dict(Counter(row["question_type"] for row in scoring)),
                "sentence_markers_preserved": True,
            },
        )
