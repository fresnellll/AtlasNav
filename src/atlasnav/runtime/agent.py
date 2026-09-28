"""Provider-neutral finite-budget Atlas agent with incremental trajectories."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

import httpx

from atlasnav.config import ModelProfile
from atlasnav.io import atomic_json, stable_json


RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
FINAL_RE = re.compile(r"<FINAL>(.*?)</FINAL>", re.I | re.S)


TOOL_SPECIFICATIONS = [
    ("overview", "Page through Atlas parent regions.", {
        "type": "object", "properties": {"offset": {"type": "integer"}, "limit": {"type": "integer"}},
    }),
    ("expand", "Expand a parent or leaf address.", {
        "type": "object", "properties": {"address": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["address"],
    }),
    ("files", "List query-ranked files in a region.", {
        "type": "object", "properties": {"address": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["address"],
    }),
    ("locate", "Find the Atlas address of a canonical handle.", {
        "type": "object", "properties": {"handle": {"type": "string"}}, "required": ["handle"],
    }),
    ("search", "Search the full canonical corpus, optionally within a region.", {
        "type": "object", "properties": {"query": {"type": "string"}, "cluster": {"type": "string"}, "mode": {"type": "string", "enum": ["all", "any", "phrase", "raw"]}, "limit": {"type": "integer"}}, "required": ["query"],
    }),
    ("route", "Project a new lexical clue over Atlas regions.", {
        "type": "object", "properties": {"query": {"type": "string"}, "mode": {"type": "string", "enum": ["all", "any", "phrase", "raw"]}, "limit": {"type": "integer"}}, "required": ["query"],
    }),
    ("leads", "Show the persistent mechanical lead ledger.", {
        "type": "object", "properties": {"limit": {"type": "integer"}, "include_resolved": {"type": "boolean"}},
    }),
    ("open", "Open canonical document content, optionally around literal terms.", {
        "type": "object", "properties": {"handle": {"type": "string"}, "find": {"type": "string"}, "start": {"type": "integer"}, "lines": {"type": "integer"}, "context": {"type": "integer"}}, "required": ["handle"],
    }),
    ("dismiss", "Mark a lead checked and irrelevant.", {
        "type": "object", "properties": {"handle": {"type": "string"}, "reason": {"type": "string"}}, "required": ["handle"],
    }),
]


def chat_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
        for name, description, parameters in TOOL_SPECIFICATIONS
    ]


def response_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "name": name, "description": description, "parameters": parameters, "strict": False}
        for name, description, parameters in TOOL_SPECIFICATIONS
    ]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class AgentTurn:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    usage: Usage
    chat_message: dict[str, Any] | None = None
    response_items: list[dict[str, Any]] = field(default_factory=list)
    response_id: str | None = None


def _usage(payload: dict[str, Any]) -> Usage:
    value = payload.get("usage") or {}
    input_tokens = int(value.get("input_tokens") or value.get("prompt_tokens") or 0)
    output_tokens = int(value.get("output_tokens") or value.get("completion_tokens") or 0)
    input_details = value.get("input_tokens_details") or value.get("prompt_tokens_details") or {}
    cached = int(input_details.get("cached_tokens") or input_details.get("cache_read_tokens") or 0)
    creation = int(input_details.get("cache_creation_tokens") or 0)
    return Usage(input_tokens, cached, output_tokens, creation)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class AgentClient:
    def __init__(self, profile: ModelProfile, timeout_seconds: float = 180.0, retries: int = 8) -> None:
        if not profile.base_url or not profile.api_key:
            raise ValueError("agent base URL and API key environment variables are required")
        self.profile = profile
        self.base_url = profile.base_url.rstrip("/")
        self.retries = retries
        self.client = httpx.AsyncClient(timeout=timeout_seconds, headers={"Authorization": f"Bearer {profile.api_key}"})

    async def close(self) -> None:
        await self.client.aclose()

    async def _post(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.post(f"{self.base_url}/{suffix}", json=payload)
                if response.status_code in RETRYABLE:
                    raise httpx.HTTPStatusError("retryable provider status", request=response.request, response=response)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(30.0, 0.7 * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2))
        raise AssertionError("unreachable")

    async def chat(self, messages: list[dict[str, Any]], tools_enabled: bool) -> AgentTurn:
        payload: dict[str, Any] = {
            "model": self.profile.model_id,
            "messages": messages,
            "stream": False,
        }
        if tools_enabled:
            payload.update({"tools": chat_tools(), "tool_choice": "auto"})
        if self.profile.reasoning_effort:
            payload["reasoning_effort"] = self.profile.reasoning_effort
        raw = await self._post("chat/completions", payload)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("agent response has no choices")
        message = choices[0].get("message") or {}
        calls = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            calls.append(ToolCall(str(item.get("id") or ""), str(function.get("name") or ""), _arguments(function.get("arguments"))))
        return AgentTurn(
            content=str(message.get("content") or ""),
            reasoning=str(message.get("reasoning_content") or ""),
            tool_calls=calls,
            usage=_usage(raw),
            chat_message={
                "role": "assistant", "content": message.get("content"),
                **({"tool_calls": message.get("tool_calls")} if message.get("tool_calls") else {}),
            },
        )

    async def responses(self, items: list[dict[str, Any]], tools_enabled: bool) -> AgentTurn:
        payload: dict[str, Any] = {"model": self.profile.model_id, "input": items, "stream": False}
        if tools_enabled:
            payload["tools"] = response_tools()
        if self.profile.reasoning_effort:
            payload["reasoning"] = {"effort": self.profile.reasoning_effort}
        raw = await self._post("responses", payload)
        output = raw.get("output") or []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in output:
            if item.get("type") == "function_call":
                calls.append(ToolCall(str(item.get("call_id") or item.get("id") or ""), str(item.get("name") or ""), _arguments(item.get("arguments"))))
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        content_parts.append(str(part.get("text") or ""))
            elif item.get("type") == "reasoning":
                for part in item.get("summary") or []:
                    reasoning_parts.append(str(part.get("text") or ""))
        if not content_parts and raw.get("output_text"):
            content_parts.append(str(raw["output_text"]))
        return AgentTurn(
            content="\n".join(content_parts), reasoning="\n".join(reasoning_parts),
            tool_calls=calls, usage=_usage(raw), response_items=output,
            response_id=str(raw.get("id") or "") or None,
        )


def _tool_command(name: str, arguments: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-m", "atlasnav.runtime.atlas_tool", name]
    if name in {"expand", "files"}:
        command.append(str(arguments.get("address") or ""))
    elif name in {"locate", "open", "dismiss"}:
        command.append(str(arguments.get("handle") or ""))
    elif name in {"search", "route"}:
        command.append(str(arguments.get("query") or ""))
    options = {
        "offset": "--offset", "limit": "--limit", "cluster": "--cluster",
        "mode": "--mode", "find": "--find", "start": "--start",
        "lines": "--lines", "context": "--context", "reason": "--reason",
    }
    for key, option in options.items():
        value = arguments.get(key)
        if value is not None and value != "":
            command.extend((option, str(value)))
    if name == "leads" and arguments.get("include_resolved") is True:
        command.append("--all")
    return command


async def execute_tool(workspace: Path, call: ToolCall, timeout_seconds: float) -> dict[str, Any]:
    known = {name for name, _, _ in TOOL_SPECIFICATIONS}
    if call.name not in known:
        return {"call_id": call.call_id, "name": call.name, "arguments": call.arguments,
                "is_error": True, "output": "unknown tool"}
    try:
        process = await asyncio.to_thread(
            subprocess.run, _tool_command(call.name, call.arguments), cwd=workspace,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_seconds, check=False,
        )
        output = process.stdout[-120_000:]
        return {"call_id": call.call_id, "name": call.name, "arguments": call.arguments,
                "is_error": process.returncode != 0, "output": output}
    except subprocess.TimeoutExpired:
        return {"call_id": call.call_id, "name": call.name, "arguments": call.arguments,
                "is_error": True, "output": f"tool exceeded {timeout_seconds:.0f}s timeout"}


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stable_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _final_text(content: str) -> str:
    match = FINAL_RE.search(content)
    return (match.group(1) if match else content).strip()


async def run_query(
    workspace: Path,
    output: Path,
    profile: ModelProfile,
    client: AgentClient,
    system_prompt: str,
    safe_release_prompt: str,
    tool_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("run_status") == "completed":
            return prior
    query = (workspace / "QUERY.txt").read_text(encoding="utf-8").strip()
    route = (workspace / "ROUTE.tsv").read_text(encoding="utf-8")
    user_prompt = f"Question:\n{query}\n\nInitial Atlas viewport:\n{route}"
    chat_history: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_history: list[dict[str, Any]] = list(chat_history)
    events = output / "events.jsonl"
    if events.exists():
        events.rename(output / f"events.incomplete-{int(time.time())}.jsonl")
    _append_event(events, {"type": "input", "message_id": 0, "role": "system", "content": system_prompt})
    _append_event(events, {"type": "input", "message_id": 1, "role": "user", "content": user_prompt})
    total = Usage()
    cumulative_cost = 0.0
    safe_release_sent = False
    final = ""
    started = time.monotonic()
    repairs = 0
    turn = 0
    while turn < profile.safe_release.max_turns:
        turn += 1
        tools_enabled = not safe_release_sent or profile.safe_release.allow_tools
        if profile.transport == "openai_chat":
            response = await client.chat(chat_history, tools_enabled)
        elif profile.transport == "openai_responses":
            response = await client.responses(response_history, tools_enabled)
        else:
            raise ValueError(f"unsupported agent transport: {profile.transport}")
        total = Usage(
            total.input_tokens + response.usage.input_tokens,
            total.cached_input_tokens + response.usage.cached_input_tokens,
            total.output_tokens + response.usage.output_tokens,
            total.cache_creation_tokens + response.usage.cache_creation_tokens,
        )
        incremental = profile.pricing.cost(
            input_tokens=max(0, response.usage.input_tokens - response.usage.cached_input_tokens),
            cached_input_tokens=response.usage.cached_input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=response.usage.cache_creation_tokens,
        )
        cumulative_cost += incremental
        _append_event(events, {
            "type": "assistant", "turn": turn, "content": response.content,
            "reasoning": response.reasoning, "tool_calls": [call.__dict__ for call in response.tool_calls],
            "usage": response.usage.__dict__, "incremental_cost": incremental,
            "cumulative_cost": cumulative_cost,
        })
        if profile.transport == "openai_chat" and response.chat_message is not None:
            chat_history.append(response.chat_message)
        elif profile.transport == "openai_responses":
            response_history.extend(response.response_items)
        if response.tool_calls:
            observations = await asyncio.gather(*[
                execute_tool(workspace, call, tool_timeout_seconds) for call in response.tool_calls
            ])
            for observation in observations:
                _append_event(events, {"type": "tool", "turn": turn, **observation})
                if profile.transport == "openai_chat":
                    chat_history.append({
                        "role": "tool", "tool_call_id": observation["call_id"],
                        "content": observation["output"],
                    })
                else:
                    response_history.append({
                        "type": "function_call_output", "call_id": observation["call_id"],
                        "output": observation["output"],
                    })
        elif response.content.strip():
            final = _final_text(response.content)
            if final:
                break
        else:
            repairs += 1
            if repairs > 2:
                break
            repair = "Return the best available answer now as <FINAL>answer</FINAL>."
            if profile.transport == "openai_chat":
                chat_history.append({"role": "user", "content": repair})
            else:
                response_history.append({"role": "user", "content": repair})
            _append_event(events, {"type": "repair", "turn": turn, "content": repair})
        threshold = profile.safe_release.cost_threshold
        if profile.safe_release.enabled and not safe_release_sent and (
            (threshold is not None and cumulative_cost >= threshold)
            or turn >= profile.safe_release.fallback_turn
        ):
            safe_release_sent = True
            if profile.transport == "openai_chat":
                chat_history.append({"role": "user", "content": safe_release_prompt})
            else:
                response_history.append({"role": "user", "content": safe_release_prompt})
            _append_event(events, {
                "type": "safe_release", "turn": turn, "content": safe_release_prompt,
                "cost_threshold": threshold, "cumulative_cost": cumulative_cost,
            })
    result = {
        "schema": "atlasnav_live_result_v1",
        "query_id": workspace.name,
        "query": query,
        "final_text": final,
        "run_status": "completed" if final else "invalid_terminal",
        "terminal_valid": bool(final),
        "turn_count": turn,
        "safe_release_sent": safe_release_sent,
        "agent_usage": {
            **total.__dict__, "cost_total": cumulative_cost,
            "currency": profile.pricing.currency,
        },
        "wall_time_seconds": time.monotonic() - started,
        "model_profile": profile.profile_id,
        "transport": profile.transport,
        "provider_account_metadata_recorded": False,
    }
    atomic_json(output / "conversation.json", {
        "schema": "atlasnav_conversation_v1",
        "transport": profile.transport,
        "messages": chat_history if profile.transport == "openai_chat" else response_history,
    })
    atomic_json(result_path, result)
    return result


async def run_suite(
    *,
    runtime: Path,
    output: Path,
    profile: ModelProfile,
    system_prompt: Path,
    safe_release_prompt: Path,
    concurrency: int,
    ramp_start: int,
    ramp_seconds: float,
    tool_timeout_seconds: float = 30.0,
    limit: int | None = None,
) -> dict[str, Any]:
    if not 1 <= ramp_start <= concurrency:
        raise ValueError("invalid agent concurrency ramp")
    runtime, output = runtime.resolve(), output.resolve()
    workspaces = sorted((runtime / "workspaces").iterdir(), key=lambda path: path.name)
    if limit is not None:
        workspaces = workspaces[:limit]
    output.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[Path | None] = asyncio.Queue()
    for workspace in workspaces:
        result = output / workspace.name / "result.json"
        if result.is_file() and json.loads(result.read_text(encoding="utf-8")).get("run_status") == "completed":
            continue
        queue.put_nowait(workspace)
    pending_count = queue.qsize()
    for _ in range(concurrency):
        queue.put_nowait(None)
    client = AgentClient(profile)
    prompt = system_prompt.read_text(encoding="utf-8").strip()
    release = safe_release_prompt.read_text(encoding="utf-8").strip()
    started = time.monotonic()

    async def worker(index: int) -> None:
        if index >= ramp_start and ramp_seconds > 0:
            await asyncio.sleep(ramp_seconds * (index - ramp_start + 1) / max(1, concurrency - ramp_start + 1))
        while True:
            workspace = await queue.get()
            try:
                if workspace is None:
                    return
                try:
                    result = await run_query(
                        workspace, output / workspace.name, profile, client, prompt, release,
                        tool_timeout_seconds,
                    )
                    print(f"{workspace.name}: {result['run_status']} turns={result['turn_count']} cost={result['agent_usage']['cost_total']:.6f}", flush=True)
                except Exception as error:
                    atomic_json(output / workspace.name / "failure.json", {
                        "schema": "atlasnav_run_failure_v1", "query_id": workspace.name,
                        "error_type": type(error).__name__, "error": str(error),
                    })
                    print(f"{workspace.name}: failed {type(error).__name__}: {error}", flush=True)
            finally:
                queue.task_done()

    try:
        async with asyncio.TaskGroup() as group:
            for index in range(concurrency):
                group.create_task(worker(index))
    finally:
        await client.close()
    rows = []
    for workspace in workspaces:
        path = output / workspace.name / "result.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "schema": "atlasnav_live_run_v1", "profile": profile.profile_id,
        "requested_queries": len(workspaces), "scheduled_queries": pending_count,
        "durable_results": len(rows),
        "completed": sum(row.get("run_status") == "completed" for row in rows),
        "invalid_terminals": sum(row.get("run_status") == "invalid_terminal" for row in rows),
        "recorded_agent_cost": sum(float((row.get("agent_usage") or {}).get("cost_total") or 0.0) for row in rows),
        "currency": profile.pricing.currency,
        "wall_time_seconds": time.monotonic() - started,
        "concurrency": concurrency, "ramp_start": ramp_start, "ramp_seconds": ramp_seconds,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
