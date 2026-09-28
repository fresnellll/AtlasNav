"""Prepare deterministic nested PhantomWiki corpus-scale treatments.

Worlds 0--19 reproduce the 10K/50K/100K protocol by cycling three frozen
official worlds and assigning disjoint literal namespaces. A 1M build appends
independently generated, already-namespaced world shards. The same selected
questions and core evidence are used at every scale.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from atlasnav.io import sha256_file

from .common import clean, finalize_bundle, new_bundle, write_documents


QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _docid(world: int, index: int, title: str) -> str:
    digest = hashlib.sha256(f"{world}\0{index}\0{title}".encode()).hexdigest()[:12]
    return f"PW{world:03d}{index:05d}{digest}"


def _namespace(title: str, article: str, facts: list[str], world: int) -> tuple[str, str]:
    if world == 0:
        return title, article
    prefix = f"World-{world:03d}"
    literals = {title}
    for fact in facts:
        literals.update(match.group(1) for match in QUOTED.finditer(fact))
    mapping = {value: f"{prefix} {value}" for value in literals if value not in {"male", "female"}}
    pattern = re.compile("|".join(re.escape(value) for value in sorted(mapping, key=len, reverse=True)))
    return mapping[title], pattern.sub(lambda match: mapping[match.group(0)], article)


def _rank(seed: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}\0{row['type']}\0{row['id']}".encode()).hexdigest()


def _select_questions(
    source: Path, *, questions_per_type: int, seed: str,
) -> list[dict[str, Any]]:
    rows = pq.read_table(source).to_pylist()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["type"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for question_type in sorted(grouped):
        eligible = [row for row in grouped[question_type] if 1 <= len(row.get("answer") or []) <= 10]
        if len(eligible) < questions_per_type:
            raise ValueError(f"PhantomWiki type {question_type} has too few bounded-answer questions")
        selected.extend(sorted(eligible, key=lambda row: (_rank(seed, row), str(row["id"])))[:questions_per_type])
    return sorted(selected, key=lambda row: (int(row["type"]), _rank(seed, row)))


def prepare_phantomwiki(
    *,
    output: Path,
    text_corpus_sources: list[str] | list[Path],
    question_source: Path,
    worlds: int,
    generated_shards: Path | None = None,
    generated_start_world: int = 20,
    questions_per_type: int = 4,
    seed: str = "atlasnav-phantomwiki-balanced-v1",
    **_: Any,
) -> dict[str, Any]:
    sources = [Path(value).resolve() for value in text_corpus_sources]
    question_source = question_source.resolve()
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError("one or more PhantomWiki text corpus sources are missing")
    if not question_source.is_file():
        raise FileNotFoundError(question_source)
    if worlds < 2:
        raise ValueError("PhantomWiki scale requires at least two worlds")
    if worlds > generated_start_world and generated_shards is None:
        raise ValueError("worlds above the frozen prefix require --generated-shards")

    selected = _select_questions(question_source, questions_per_type=questions_per_type, seed=seed)
    core_title_to_docid: dict[str, str] = {}

    def documents() -> Iterator[dict[str, str]]:
        for world in range(worlds):
            if world < generated_start_world:
                source = sources[world % len(sources)]
                rows = pq.read_table(source, columns=["title", "article", "facts"]).to_pylist()
                for index, row in enumerate(rows):
                    original_title = str(row["title"])
                    title, article = _namespace(
                        original_title, str(row["article"]),
                        [str(value) for value in (row.get("facts") or [])], world,
                    )
                    docid = _docid(world, index, title)
                    if world == 0:
                        if original_title in core_title_to_docid:
                            raise ValueError(f"duplicate PhantomWiki core title: {original_title}")
                        core_title_to_docid[original_title] = docid
                    yield {
                        "docid": docid,
                        "text": f"Title: {title}\n{article.strip()}",
                        "url": f"https://phantomwiki.invalid/world/{world:03d}/{docid}",
                    }
                continue
            shard = Path(generated_shards).resolve() / f"world_{world:03d}" / "data.parquet"
            if not shard.is_file():
                raise FileNotFoundError(shard)
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=4096, columns=["docid", "text", "url"]):
                yield from batch.to_pylist()

    with new_bundle(output) as temporary:
        document_count = write_documents(temporary / "documents.parquet", documents())
        questions: list[dict[str, Any]] = []
        scoring: list[dict[str, Any]] = []
        for row in selected:
            qid = f"phantomwiki:{row['id']}"
            query = clean(row["question"])
            answers = sorted(clean(value) for value in (row.get("answer") or []))
            questions.append({"query_id": qid, "query": query})
            scoring.append({
                "query_id": qid, "query": query, "answer": answers,
                "source_query_id": str(row["id"]), "question_type": int(row["type"]),
                "difficulty": int(row.get("difficulty", 0)),
                "gold_docids": sorted({core_title_to_docid[value] for value in answers
                                       if value in core_title_to_docid}),
                "prolog": row.get("prolog"), "template": row.get("template"),
            })
        source_records = {
            f"world_source_{index}": {"file": path.name, "sha256": sha256_file(path)}
            for index, path in enumerate(sources)
        }
        source_records["questions"] = {
            "file": question_source.name, "sha256": sha256_file(question_source),
        }
        return finalize_bundle(
            temporary, benchmark=f"phantomwiki-{worlds}-worlds", documents=document_count,
            questions=questions, scoring=scoring, sources=source_records,
            details={
                "task_type": "synthetic_multi_hop_qa", "worlds": worlds,
                "nested_world_prefix": True, "core_question_world": 0,
                "generated_start_world": generated_start_world,
                "selection_seed": seed, "questions_per_type": questions_per_type,
                "type_counts": dict(Counter(row["question_type"] for row in scoring)),
                "distractor_namespace": "World-NNN prefix for every non-gender quoted literal",
            },
        )
