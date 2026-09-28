from __future__ import annotations

import asyncio
from pathlib import Path

from atlasnav.io import atomic_json, atomic_jsonl, sha256_file
from atlasnav.router import synthesize as module


def test_generation_is_validated_in_same_invocation(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    rows = []
    for index in range(2):
        rows.append({
            "schema": "atlasnav_grounded_support_task_v1",
            "task_id": f"single:train:{index:06d}", "kind": "single", "split": "train",
            "support_files": [{
                "file_index": index, "docid": str(index), "parent": index, "leaf": index,
                "excerpts": [{"text": f"Entity {index} has the unique verification value alpha-{index}."}],
            }],
        })
    task_file = tasks / "tasks.jsonl"
    atomic_jsonl(task_file, rows)
    atomic_json(tasks / "manifest.json", {
        "schema": "atlasnav_grounded_support_tasks_v1", "finalized": True,
        "tasks": 2, "tasks_sha256": sha256_file(task_file),
    })

    class FakeClient:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def close(self) -> None:
            pass

        async def call(self, prompt: str):
            FakeClient.calls += 1
            if prompt.startswith("Create"):
                return ({row["task_id"]: {
                    "task_id": row["task_id"], "usable": True,
                    "question": f"Which entity has the unique verification value alpha-{index} in this account?",
                } for index, row in enumerate(rows)}, {"input_tokens": 20, "output_tokens": 10})
            return ({row["task_id"]: {
                "task_id": row["task_id"], "grounded": True, "natural": True,
                "all_positives_necessary": True, "decoy_required": False,
                "single_answer_target": True, "coherent_chain": True,
            } for row in rows}, {"input_tokens": 20, "output_tokens": 10})

    monkeypatch.setattr(module, "StructuredClient", FakeClient)
    monkeypatch.setenv("ATLASNAV_SYNTHESIS_TEST_KEY", "test-only")
    output = tmp_path / "questions"
    result = asyncio.run(module.synthesize_questions(
        tasks_directory=tasks, output=output, base_url="https://example.invalid/v1",
        api_key_env="ATLASNAV_SYNTHESIS_TEST_KEY", model="synthetic",
        concurrency=1, batch_size=2, kind_targets={"single": 2},
        split_targets={"train": 2},
    ))
    assert FakeClient.calls == 2
    assert result["questions"] == 2
    assert result["accepted_candidates"] == 2
