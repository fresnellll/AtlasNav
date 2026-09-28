"""Prepare the public FanOutQA dev evidence-union closed corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlasnav.io import sha256_file

from .common import finalize_bundle, new_bundle, write_documents


def _leaves(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("evidence") is not None:
            result.append(node)
        result.extend(_leaves(node.get("decomposition") or []))
    return result


def _atoms(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [atom for key, item in value.items() for atom in _atoms(key) + _atoms(item)]
    if isinstance(value, list):
        return [atom for item in value for atom in _atoms(item)]
    if isinstance(value, bool):
        return ["yes" if value else "no"]
    return [str(value)]


def prepare_fanoutqa(
    *, output: Path, official_dev: Path, retrieval_dataset: Path, **_: Any,
) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("install atlasnav[benchmarks] to prepare FanOutQA") from error
    official_dev = official_dev.resolve()
    retrieval_dataset = retrieval_dataset.resolve()
    official_rows = json.loads(official_dev.read_text(encoding="utf-8"))
    dataset = load_from_disk(str(retrieval_dataset))
    queries = {str(row["query_id"]): row for row in dataset["queries"]}
    mappings = {str(row["query_id"]): [str(value) for value in row["doc_ids"]]
                for row in dataset["query_to_docs"]}
    corpus = {str(row["doc_id"]): row for row in dataset["corpus"]}
    if {str(row["id"]) for row in official_rows} != set(queries) or set(queries) != set(mappings):
        raise ValueError("FanOutQA official and retrieval query identities do not align")

    evidence_by_page: dict[str, dict[str, Any]] = {}
    leaves_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in official_rows:
        qid = str(row["id"])
        leaves = _leaves(row.get("decomposition") or [])
        leaves_by_query[qid] = leaves
        pageids = []
        for leaf in leaves:
            evidence = dict(leaf["evidence"])
            pageid = str(evidence["pageid"])
            pageids.append(pageid)
            prior = evidence_by_page.setdefault(pageid, evidence)
            if (prior.get("revid"), prior.get("title")) != (evidence.get("revid"), evidence.get("title")):
                raise ValueError(f"conflicting evidence identity for FanOutQA page {pageid}")
        if pageids != mappings[qid]:
            raise ValueError(f"FanOutQA evidence mapping mismatch for {qid}")
    if set(evidence_by_page) != set(corpus):
        raise ValueError("FanOutQA corpus must equal the union of dev evidence pages")

    def documents() -> Any:
        for pageid in sorted(evidence_by_page, key=int):
            evidence, source = evidence_by_page[pageid], corpus[pageid]
            if str(evidence["title"]) != str(source["title"]):
                raise ValueError(f"FanOutQA title mismatch for page {pageid}")
            docid = f"FQ{int(pageid):08d}"
            text = (
                f"Title: {evidence['title']}\nWikipedia page ID: {pageid}\n"
                f"Wikipedia revision ID: {evidence['revid']}\nURL: {source.get('url', '')}\n\n"
                f"{str(source.get('text') or '').strip()}"
            )
            yield {"docid": docid, "text": text, "url": str(source.get("url") or "")}

    questions: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    for row in official_rows:
        official_id = str(row["id"])
        qid = f"fanoutqa:{official_id}"
        query = str(row["question"])
        answer = row["answer"]
        evidence_rows: list[dict[str, Any]] = []
        for leaf in leaves_by_query[official_id]:
            evidence = leaf["evidence"]
            pageid = str(evidence["pageid"])
            evidence_rows.append({
                "docid": f"FQ{int(pageid):08d}", "pageid": int(pageid),
                "revid": int(evidence["revid"]), "title": str(evidence["title"]),
                "subquestion_id": str(leaf["id"]), "subquestion": str(leaf["question"]),
                "subanswer": leaf["answer"], "subanswer_atoms": _atoms(leaf["answer"]),
                "depends_on": list(leaf.get("depends_on") or []),
            })
        questions.append({"query_id": qid, "query": query})
        scoring.append({
            "query_id": qid, "query": query, "answer": answer,
            "answer_atoms": _atoms(answer), "categories": list(row.get("categories") or []),
            "gold_docids": sorted({item["docid"] for item in evidence_rows}),
            "evidence": evidence_rows,
        })

    with new_bundle(output) as temporary:
        count = write_documents(temporary / "documents.parquet", documents())
        return finalize_bundle(
            temporary, benchmark="fanoutqa-dev-closed-corpus", documents=count,
            questions=questions, scoring=scoring,
            sources={
                "official_dev": {"file": official_dev.name, "sha256": sha256_file(official_dev)},
                "retrieval_dataset": {"directory": retrieval_dataset.name},
            },
            details={
                "task_type": "fan_out_qa", "setting": "public dev evidence-union closed corpus",
                "official_open_book_leaderboard_comparable": False,
                "primary_metrics": ["loose_accuracy", "strict_accuracy"],
                "evidence_slots": sum(len(row["evidence"]) for row in scoring),
            },
        )
