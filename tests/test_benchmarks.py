from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atlasnav.benchmarks.common import audit_bundle
from atlasnav.benchmarks.enterprise import prepare_enterprise
from atlasnav.benchmarks.phantomwiki import prepare_phantomwiki
from atlasnav.benchmarks.results import score_official
from atlasnav.benchmarks.retrieval import prepare_retrieval
from atlasnav.benchmarks.twowiki import prepare_2wiki


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_beir_adapter_and_graded_scorer(tmp_path: Path) -> None:
    raw = tmp_path / "beir"
    _jsonl(raw / "corpus.jsonl", [
        {"_id": "d1", "title": "One", "text": "alpha"},
        {"_id": "d2", "title": "Two", "text": "beta"},
    ])
    _jsonl(raw / "queries.jsonl", [{"_id": "q1", "text": "find alpha"}])
    qrels = raw / "qrels/test.tsv"
    qrels.parent.mkdir()
    with qrels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerow(["q1", "d1", 2])
    bundle = tmp_path / "bundle"
    prepare_retrieval(adapter="trec-covid", output=bundle, raw_root=raw)
    assert audit_bundle(bundle)["passed"]
    run = tmp_path / "run/trec-covid:q1"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "final_text": "Relevant Documents\n1. Dd1.txt\n2. Dd2.txt\n"
    }), encoding="utf-8")
    report = score_official(bundle, tmp_path / "run", tmp_path / "score")
    assert report["metrics"]["ndcg_at_10"] == 1.0


def test_enterprise_duplicate_ids_are_preserved(tmp_path: Path) -> None:
    raw = tmp_path / "enterprise/data"
    (raw / "documents").mkdir(parents=True)
    (raw / "questions").mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"doc_id": "same", "source_type": "jira", "title": "A", "content": "first"},
        {"doc_id": "same", "source_type": "jira", "title": "B", "content": "second"},
    ]), raw / "documents/test.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "question_id": "q", "question": "what", "gold_answer": "answer",
        "expected_doc_ids": ["same", "same"], "answer_facts": ["fact"],
        "question_type": "multi", "source_types": ["jira"],
    }]), raw / "questions/test.parquet")
    bundle = tmp_path / "bundle"
    manifest = prepare_enterprise(
        output=bundle, raw_root=tmp_path / "enterprise",
        expected_documents=None, expected_questions=None,
    )
    assert manifest["duplicate_raw_document_ids"] == 1
    scoring = json.loads((bundle / "scoring.jsonl").read_text())
    assert scoring["gold_docids"] == ["same", "same__dup2"]
    assert audit_bundle(bundle)["passed"]


def test_2wiki_adapter_preserves_support_sentences(tmp_path: Path) -> None:
    rows = []
    for index, kind in enumerate(("comparison", "inference", "compositional", "bridge_comparison")):
        rows.append({
            "_id": f"q{index}", "type": kind, "question": f"question {index}",
            "context": json.dumps([[f"Title {index}", ["zero", "support"]]]),
            "supporting_facts": json.dumps([[f"Title {index}", 1]]),
            "evidences": json.dumps([]), "answer": "support",
        })
    source = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    bundle = tmp_path / "bundle"
    manifest = prepare_2wiki(output=bundle, source=source, questions_per_type=1)
    assert manifest["questions"] == 4
    assert audit_bundle(bundle)["passed"]
    scoring = [json.loads(line) for line in (bundle / "scoring.jsonl").read_text().splitlines()]
    assert all(row["supporting_facts"][0]["sentence"] == "support" for row in scoring)


def test_phantomwiki_nested_world_adapter(tmp_path: Path) -> None:
    corpus = tmp_path / "text.parquet"
    questions = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"title": "Alice", "article": "Alice knows Bob.", "facts": ['knows("Alice","Bob")']},
        {"title": "Bob", "article": "Bob knows Alice.", "facts": ['knows("Bob","Alice")']},
    ]), corpus)
    pq.write_table(pa.Table.from_pylist([{
        "id": "q1", "question": "Who knows Bob?", "answer": ["Alice"],
        "type": 0, "difficulty": 1, "prolog": {"query": ["x"]}, "template": ["x"],
    }]), questions)
    bundle = tmp_path / "bundle"
    manifest = prepare_phantomwiki(
        output=bundle, text_corpus_sources=[corpus], question_source=questions,
        worlds=2, generated_start_world=20, questions_per_type=1,
    )
    assert manifest["documents"] == 4
    assert audit_bundle(bundle)["passed"]
    scoring = json.loads((bundle / "scoring.jsonl").read_text())
    assert len(scoring["gold_docids"]) == 1
