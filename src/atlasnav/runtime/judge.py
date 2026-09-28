"""Provider-neutral, durable answer judging for AtlasNav runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
import re
import time
from typing import Any

import httpx

from atlasnav.config import ModelProfile
from atlasnav.io import atomic_json
from atlasnav.runtime.agent import RETRYABLE, Usage, _usage


SYSTEM_PROMPT = """You are an exacting answer evaluator. Compare the candidate answer with the
reference answer for the question. Accept harmless formatting, capitalization, aliases, and an
equivalent value. Reject extra contradictory claims, failure to answer, and merely related facts.
Return exactly one JSON object with keys is_correct (boolean) and reason (short string)."""


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id") or row.get("id") or row.get("question_id") or "")
            question = row.get("query") or row.get("question")
            answer = row.get("answer") or row.get("reference_answer")
            if not qid or question is None or answer is None:
                raise ValueError("judge dataset requires query_id, query/question, and answer")
            if qid in rows:
                raise ValueError(f"duplicate query ID: {qid}")
            rows[qid] = {**row, "query_id": qid, "query": str(question), "answer": str(answer)}
    return rows


def _parse_json(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not value.startswith("{") and "{" in value and "}" in value:
        value = value[value.find("{"):value.rfind("}") + 1]
    result = json.loads(value)
    if not isinstance(result, dict) or not isinstance(result.get("is_correct"), bool):
        raise ValueError("judge response does not contain boolean is_correct")
    return {"is_correct": bool(result["is_correct"]), "reason": str(result.get("reason") or "")[:2000]}


class JudgeClient:
    def __init__(self, profile: ModelProfile, timeout_seconds: float = 180.0, retries: int = 8) -> None:
        if not profile.base_url or not profile.api_key:
            raise ValueError("judge base URL and API key environment variables are required")
        self.profile = profile
        self.base_url = profile.base_url.rstrip("/")
        self.retries = retries
        self.client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {profile.api_key}"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _post(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.post(f"{self.base_url}/{suffix}", json=payload)
                if response.status_code in RETRYABLE:
                    raise httpx.HTTPStatusError(
                        "retryable judge status", request=response.request, response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt >= self.retries:
                    raise
                delay = min(30.0, 0.7 * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def judge(self, question: str, reference: str, candidate: str) -> tuple[dict[str, Any], Usage]:
        prompt = (
            f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate}\n"
        )
        if self.profile.transport == "openai_chat":
            payload: dict[str, Any] = {
                "model": self.profile.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            raw = await self._post("chat/completions", payload)
            choices = raw.get("choices") or []
            content = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
        elif self.profile.transport == "openai_responses":
            payload = {
                "model": self.profile.model_id,
                "instructions": SYSTEM_PROMPT,
                "input": prompt,
                "stream": False,
                "text": {"format": {"type": "json_object"}},
            }
            raw = await self._post("responses", payload)
            content = str(raw.get("output_text") or "")
            if not content:
                pieces = []
                for item in raw.get("output") or []:
                    for part in item.get("content") or []:
                        if part.get("type") in {"output_text", "text"}:
                            pieces.append(str(part.get("text") or ""))
                content = "\n".join(pieces)
        else:
            raise ValueError(f"unsupported judge transport: {self.profile.transport}")
        return _parse_json(content), _usage(raw)


async def judge_suite(
    *,
    run_directory: Path,
    dataset: Path,
    output: Path,
    profile: ModelProfile,
    concurrency: int,
    ramp_start: int,
    ramp_seconds: float,
) -> dict[str, Any]:
    if not 1 <= ramp_start <= concurrency:
        raise ValueError("invalid judge concurrency ramp")
    run_directory, output = run_directory.resolve(), output.resolve()
    references = load_dataset(dataset.resolve())
    output.mkdir(parents=True, exist_ok=True)
    query_ids = sorted(references)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for qid in query_ids:
        destination = output / qid / "judge_result.json"
        if destination.is_file():
            continue
        result = run_directory / qid / "result.json"
        if not result.is_file():
            continue
        queue.put_nowait(qid)
    scheduled = queue.qsize()
    for _ in range(concurrency):
        queue.put_nowait(None)
    client = JudgeClient(profile)
    started = time.monotonic()

    async def worker(index: int) -> None:
        if index >= ramp_start and ramp_seconds > 0:
            await asyncio.sleep(
                ramp_seconds * (index - ramp_start + 1) / max(1, concurrency - ramp_start + 1)
            )
        while True:
            qid = await queue.get()
            try:
                if qid is None:
                    return
                destination = output / qid
                destination.mkdir(parents=True, exist_ok=True)
                run = json.loads((run_directory / qid / "result.json").read_text(encoding="utf-8"))
                candidate = str(run.get("final_text") or "")
                if not candidate:
                    record = {
                        "schema": "atlasnav_judge_result_v1", "query_id": qid,
                        "is_correct": False, "reason": "invalid or empty agent terminal",
                        "judge_usage": {**Usage().__dict__, "cost_total": 0.0,
                                        "currency": profile.pricing.currency},
                        "model_profile": profile.profile_id,
                    }
                else:
                    verdict, usage = await client.judge(
                        references[qid]["query"], references[qid]["answer"], candidate,
                    )
                    cost = profile.pricing.cost(
                        input_tokens=max(0, usage.input_tokens - usage.cached_input_tokens),
                        cached_input_tokens=usage.cached_input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_tokens=usage.cache_creation_tokens,
                    )
                    record = {
                        "schema": "atlasnav_judge_result_v1", "query_id": qid,
                        **verdict,
                        "judge_usage": {**usage.__dict__, "cost_total": cost,
                                        "currency": profile.pricing.currency},
                        "model_profile": profile.profile_id,
                    }
                atomic_json(destination / "judge_result.json", record)
                print(f"{qid}: judge={record['is_correct']}", flush=True)
            except Exception as error:
                atomic_json(destination / "judge_failure.json", {
                    "schema": "atlasnav_judge_failure_v1", "query_id": qid,
                    "error_type": type(error).__name__, "error": str(error),
                })
                print(f"{qid}: judge failed {type(error).__name__}: {error}", flush=True)
            finally:
                queue.task_done()

    try:
        async with asyncio.TaskGroup() as group:
            for index in range(concurrency):
                group.create_task(worker(index))
    finally:
        await client.close()
    rows = []
    for qid in query_ids:
        path = output / qid / "judge_result.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "schema": "atlasnav_judge_run_v1", "profile": profile.profile_id,
        "dataset_queries": len(query_ids), "scheduled_queries": scheduled,
        "durable_results": len(rows), "correct": sum(row["is_correct"] for row in rows),
        "recorded_judge_cost": sum(float(row["judge_usage"]["cost_total"]) for row in rows),
        "currency": profile.pricing.currency, "wall_time_seconds": time.monotonic() - started,
        "concurrency": concurrency, "ramp_start": ramp_start, "ramp_seconds": ramp_seconds,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
