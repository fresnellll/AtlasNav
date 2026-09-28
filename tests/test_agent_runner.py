from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlasnav.config import ModelProfile, Pricing, SafeRelease
from atlasnav.runtime.agent import AgentTurn, Usage, run_query


class FakeClient:
    async def chat(self, messages, tools_enabled):
        assert tools_enabled is True
        return AgentTurn(
            content="<FINAL>Cedar</FINAL>", reasoning="", tool_calls=[],
            usage=Usage(input_tokens=100, cached_input_tokens=20, output_tokens=10),
            chat_message={"role": "assistant", "content": "<FINAL>Cedar</FINAL>"},
        )


@pytest.mark.asyncio
async def test_agent_writes_incremental_and_final_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "runtime/workspaces/q1"
    workspace.mkdir(parents=True)
    (workspace / "QUERY.txt").write_text("What is the answer?\n", encoding="utf-8")
    (workspace / "ROUTE.tsv").write_text("rank\tanchors\n1\tD42 Cedar\n", encoding="utf-8")
    profile = ModelProfile(
        profile_id="test", model_id="test", transport="openai_chat",
        base_url_env="UNUSED", api_key_env="UNUSED", context_window=10000,
        reasoning_effort=None,
        pricing=Pricing("CNY", 1.0, 0.2, 2.0),
        safe_release=SafeRelease(True, 2.75, 300, 295, False),
    )
    output = tmp_path / "run/q1"
    result = await run_query(
        workspace, output, profile, FakeClient(), "system", "release", 30.0,
    )
    assert result["final_text"] == "Cedar"
    assert result["run_status"] == "completed"
    assert result["agent_usage"]["cost_total"] > 0
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["input", "input", "assistant"]
    assert (output / "conversation.json").is_file()
