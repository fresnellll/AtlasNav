"""Generate and independently verify grounded Router training questions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from typing import Any

import httpx

from atlasnav.io import atomic_json, atomic_jsonl, sha256_file, stable_json
from atlasnav.runtime.agent import RETRYABLE


FORBIDDEN_RE = re.compile(
    r"\b(?:source|document|file|packet|excerpt|provided text|according to the passage)\b", re.I,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _load_tasks(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    tasks = []
    with (path / "tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                tasks.append(json.loads(line))
    if manifest.get("schema") != "atlasnav_grounded_support_tasks_v1":
        raise ValueError("grounded support task manifest is required")
    if sha256_file(path / "tasks.jsonl") != manifest.get("tasks_sha256"):
        raise RuntimeError("grounded support task checksum mismatch")
    return tasks, manifest


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"], "kind": task["kind"],
        "packets": [
            {"packet_id": f"P{index}", "clues": [item["text"] for item in row["excerpts"]]}
            for index, row in enumerate(task["support_files"])
        ],
    }


def _generation_prompt(tasks: list[dict[str, Any]]) -> str:
    return (
        "Create one natural single-answer research question per item using only the supplied clue packets. "
        "For pair and triple items, every packet must contribute a necessary link in one coherent chain. "
        "Never mention files, documents, sources, packets, excerpts, retrieval, or the answer. Do not join "
        "independent subquestions. If the clues cannot support such a question, set usable=false. Return strict "
        "JSON {\"results\":[{\"task_id\":str,\"usable\":bool,\"question\":str,"
        "\"packet_roles\":[str],\"answer_type\":str}]}.\n\n"
        + json.dumps([_task_payload(task) for task in tasks], ensure_ascii=False, separators=(",", ":"))
    )


def _validation_prompt(items: list[tuple[dict[str, Any], str, dict[str, Any]]]) -> str:
    payload = []
    for task, question, decoy in items:
        value = _task_payload(task)
        value["question"] = question
        value["decoy"] = {
            "packet_id": "N0", "clues": [item["text"] for item in decoy["support_files"][0]["excerpts"]],
        }
        payload.append(value)
    return (
        "Audit each retrieval-training question using only its positive packets and decoy. Accept only when the "
        "question is natural, grounded, has one answer target, every positive packet is necessary, and the decoy "
        "is unnecessary. Reject hallucinated links, unrelated conjunctions, and source/retrieval wording. Return "
        "strict JSON {\"results\":[{\"task_id\":str,\"grounded\":bool,\"natural\":bool,"
        "\"all_positives_necessary\":bool,\"decoy_required\":bool,\"single_answer_target\":bool,"
        "\"coherent_chain\":bool,\"feedback\":str}]}.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _parse(content: str) -> dict[str, dict[str, Any]]:
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not value.startswith("{") and "{" in value and "}" in value:
        value = value[value.find("{"):value.rfind("}") + 1]
    parsed = json.loads(value)
    rows = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        raise ValueError("structured response has no results array")
    return {str(row["task_id"]): row for row in rows if isinstance(row, dict) and row.get("task_id")}


def _share_usage(usage: dict[str, int], count: int, index: int) -> dict[str, int]:
    result = {}
    for key, value in usage.items():
        quotient, remainder = divmod(int(value), count)
        result[key] = quotient + int(index < remainder)
    return result


class StructuredClient:
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int, retries: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=240.0, headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def call(self, prompt: str) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "response_format": {"type": "json_object"},
        }
        async with self.semaphore:
            for attempt in range(1, self.retries + 1):
                try:
                    response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
                    if response.status_code in RETRYABLE:
                        raise httpx.HTTPStatusError("retryable status", request=response.request, response=response)
                    response.raise_for_status()
                    raw = response.json()
                    choices = raw.get("choices") or []
                    content = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
                    usage = raw.get("usage") or {}
                    return _parse(content), {
                        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                    }
                except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                    if attempt >= self.retries:
                        raise
                    await asyncio.sleep(min(30.0, 0.8 * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2))
        raise AssertionError("unreachable")


class Cache:
    def __init__(self, path: Path, tasks: list[dict[str, Any]]) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records(task_id TEXT PRIMARY KEY,payload_sha256 TEXT NOT NULL,"
            "generation TEXT,validation TEXT,accepted INTEGER NOT NULL DEFAULT 0,"
            "input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0)"
        )
        for task in tasks:
            identity = hashlib.sha256(stable_json(task).encode()).hexdigest()
            prior = self.connection.execute(
                "SELECT payload_sha256 FROM records WHERE task_id=?", (task["task_id"],),
            ).fetchone()
            if prior and prior[0] != identity:
                raise RuntimeError(f"Router synthesis cache identity changed: {task['task_id']}")
            self.connection.execute(
                "INSERT OR IGNORE INTO records(task_id,payload_sha256) VALUES(?,?)",
                (task["task_id"], identity),
            )
        self.connection.commit()

    def generated(self, task_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT generation FROM records WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def validated(self, task_id: str) -> bool:
        row = self.connection.execute("SELECT validation FROM records WHERE task_id=?", (task_id,)).fetchone()
        return bool(row and row[0])

    def store_generation(self, task_id: str, row: dict[str, Any], usage: dict[str, int]) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE records SET generation=?,validation=NULL,accepted=0,input_tokens=input_tokens+?,"
                "output_tokens=output_tokens+? WHERE task_id=?",
                (stable_json(row), usage["input_tokens"], usage["output_tokens"], task_id),
            )

    def store_validation(self, task_id: str, row: dict[str, Any], accepted: bool,
                         usage: dict[str, int]) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE records SET validation=?,accepted=?,input_tokens=input_tokens+?,"
                "output_tokens=output_tokens+? WHERE task_id=?",
                (stable_json(row), int(accepted), usage["input_tokens"], usage["output_tokens"], task_id),
            )


def _programmatic(question: str) -> bool:
    value = " ".join(question.split())
    return 45 <= len(value) <= 900 and value.endswith("?") and not FORBIDDEN_RE.search(value)


def _decoys(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_split[str(task["split"])].append(task)
    result = {}
    for task in tasks:
        positives = {int(row["file_index"]) for row in task["support_files"]}
        values = by_split[str(task["split"])]
        start = int(_digest(f"decoy:{task['task_id']}")[:12], 16) % len(values)
        for offset in range(len(values)):
            candidate = values[(start + offset) % len(values)]
            candidate_files = {int(row["file_index"]) for row in candidate["support_files"]}
            if candidate["task_id"] != task["task_id"] and not positives & candidate_files:
                result[task["task_id"]] = candidate
                break
        if task["task_id"] not in result:
            raise RuntimeError(f"cannot select outcome-blind decoy for {task['task_id']}")
    return result


def _quota_matrix(kind_targets: dict[str, int], split_targets: dict[str, int]) -> dict[tuple[str, str], int]:
    total = sum(kind_targets.values())
    if total != sum(split_targets.values()):
        raise ValueError("Router kind and split targets differ")
    expected = {(kind, split): kind_targets[kind] * split_targets[split] / total
                for kind in kind_targets for split in split_targets}
    values = {key: int(value) for key, value in expected.items()}
    kind_remaining = {kind: kind_targets[kind] - sum(values[kind, split] for split in split_targets)
                      for kind in kind_targets}
    split_remaining = {split: split_targets[split] - sum(values[kind, split] for kind in kind_targets)
                       for split in split_targets}
    while sum(kind_remaining.values()):
        options = [
            (expected[kind, split] - values[kind, split], kind, split)
            for kind in kind_targets for split in split_targets
            if kind_remaining[kind] and split_remaining[split]
        ]
        _fraction, kind, split = max(options)
        values[kind, split] += 1
        kind_remaining[kind] -= 1
        split_remaining[split] -= 1
    return values


async def synthesize_questions(
    *,
    tasks_directory: Path,
    output: Path,
    base_url: str,
    api_key_env: str,
    model: str,
    concurrency: int = 20,
    batch_size: int = 3,
    retries: int = 8,
    kind_targets: dict[str, int] | None = None,
    split_targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    tasks_directory, output = tasks_directory.resolve(), output.resolve()
    tasks, task_manifest = _load_tasks(tasks_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Router question bundle is already finalized: {output}")
    key = os.getenv(api_key_env)
    if not key:
        raise ValueError(f"{api_key_env} is required")
    cache = Cache(output / "synthesis_cache.sqlite3", tasks)
    client = StructuredClient(base_url, key, model, concurrency, retries)
    decoys = _decoys(tasks)

    async def generate(batch: list[dict[str, Any]]) -> None:
        rows, usage = await client.call(_generation_prompt(batch))
        for index, task in enumerate(batch):
            row = rows.get(task["task_id"]) or {"task_id": task["task_id"], "usable": False, "question": ""}
            cache.store_generation(task["task_id"], row, _share_usage(usage, len(batch), index))

    async def validate(batch: list[dict[str, Any]]) -> None:
        items = [
            (task, str((cache.generated(task["task_id"]) or {}).get("question") or ""), decoys[task["task_id"]])
            for task in batch
        ]
        rows, usage = await client.call(_validation_prompt(items))
        for index, (task, question, _decoy) in enumerate(items):
            row = rows.get(task["task_id"]) or {"task_id": task["task_id"]}
            accepted = (
                _programmatic(question)
                and all(row.get(key) is True for key in (
                    "grounded", "natural", "all_positives_necessary", "single_answer_target", "coherent_chain",
                ))
                and row.get("decoy_required") is not True
            )
            cache.store_validation(task["task_id"], row, accepted, _share_usage(usage, len(items), index))

    async def run_stage(stage: str, pending: list[dict[str, Any]], callback: Any) -> None:
        batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
        for start in range(0, len(batches), concurrency):
            await asyncio.gather(*(callback(batch) for batch in batches[start:start + concurrency]))
            print(f"Router {stage}: {min(start + concurrency, len(batches))}/{len(batches)} batches", flush=True)

    try:
        # These queues must be materialized sequentially. Building both before
        # generation would omit every newly generated row from validation in
        # the same invocation.
        await run_stage(
            "generation",
            [task for task in tasks if cache.generated(task["task_id"]) is None],
            generate,
        )
        for task in tasks:
            row = cache.generated(task["task_id"]) or {}
            if cache.validated(task["task_id"]):
                continue
            if row.get("usable") is not True or not _programmatic(str(row.get("question") or "")):
                cache.store_validation(
                    task["task_id"],
                    {"task_id": task["task_id"], "programmatic_rejection": True},
                    False,
                    {"input_tokens": 0, "output_tokens": 0},
                )
        await run_stage(
            "validation",
            [
                task for task in tasks
                if not cache.validated(task["task_id"])
                and (cache.generated(task["task_id"]) or {}).get("usable") is True
            ],
            validate,
        )
    finally:
        await client.close()
    accepted: list[dict[str, Any]] = []
    for task in tasks:
        row = cache.connection.execute(
            "SELECT generation,accepted FROM records WHERE task_id=?", (task["task_id"],),
        ).fetchone()
        if not row or not row[0] or not row[1]:
            continue
        generation = json.loads(row[0])
        accepted.append({
            "query_id": str(task["task_id"]), "query": str(generation["question"]),
            "pseudoquery_id": str(task["task_id"]), "kind": task["kind"], "split": task["split"],
            "positive_file_indices": [int(value["file_index"]) for value in task["support_files"]],
            "positive_docids": [str(value["docid"]) for value in task["support_files"]],
            "positive_parents": sorted({int(value["parent"]) for value in task["support_files"]}),
            "positive_leaves": sorted({int(value["leaf"]) for value in task["support_files"]}),
            "selection_reads_evaluation_artifacts": False,
        })
    kind_targets = kind_targets or {"single": 1913, "pair": 4300, "triple": 950}
    split_targets = split_targets or {"train": 4972, "validation": 1167, "test": 1024}
    quotas = _quota_matrix(kind_targets, split_targets)
    selected: list[dict[str, Any]] = []
    for (kind, split), count in quotas.items():
        pool = sorted(
            (row for row in accepted if row["kind"] == kind and row["split"] == split),
            key=lambda row: _digest(f"final:{row['query_id']}"),
        )
        if len(pool) < count:
            raise RuntimeError(f"accepted Router questions underfill {kind}/{split}: {len(pool)} < {count}")
        selected.extend(pool[:count])
    selected.sort(key=lambda row: row["query_id"])
    questions = output / "questions.jsonl"
    atomic_jsonl(questions, selected)
    usage = cache.connection.execute(
        "SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0) FROM records"
    ).fetchone()
    cache.connection.close()
    manifest = {
        "schema": "atlasnav_verified_router_questions_v1", "finalized": True,
        "questions": len(selected), "accepted_candidates": len(accepted),
        "kind_counts": {kind: sum(row["kind"] == kind for row in selected) for kind in kind_targets},
        "split_counts": {split: sum(row["split"] == split for row in selected) for split in split_targets},
        "quota_matrix": {f"{kind}/{split}": count for (kind, split), count in quotas.items()},
        "model": model, "provider_input_tokens": int(usage[0]), "provider_output_tokens": int(usage[1]),
        "support_task_manifest_sha256": sha256_file(tasks_directory / "manifest.json"),
        "questions_sha256": sha256_file(questions),
        "selection_reads_evaluation_artifacts": False,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
