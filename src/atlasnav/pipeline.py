"""Declarative end-to-end orchestration with durable stage boundaries."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tomllib
from typing import Any

from atlasnav.atlas.build import build_atlas
from atlasnav.config import load_model_profile
from atlasnav.corpus.prepare import build_fulltext_index, prepare_corpus
from atlasnav.embeddings.build import build_embeddings
from atlasnav.embeddings.client import EmbeddingClient
from atlasnav.embeddings.queries import build_query_embeddings
from atlasnav.evaluation.evaluate import evaluate_suite
from atlasnav.evaluation.report import render_report
from atlasnav.io import atomic_json
from atlasnav.router.train import train
from atlasnav.runtime.agent import run_suite
from atlasnav.runtime.build import audit_runtime, build_runtime
from atlasnav.runtime.judge import judge_suite


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _required(table: dict[str, Any], key: str) -> Any:
    if key not in table:
        raise ValueError(f"pipeline runbook is missing {key}")
    return table[key]


async def build_pipeline(runbook: Path) -> dict[str, Any]:
    """Build corpus, Atlas, Router, query bundle, and runtime in order."""
    runbook = runbook.resolve()
    root = runbook.parent
    cfg = _load(runbook)
    paths = cfg.get("paths") or {}
    embedding = cfg.get("embedding") or {}
    atlas_cfg = cfg.get("atlas") or {}
    runtime_cfg = cfg.get("runtime") or {}
    corpus_input = _path(root, _required(paths, "corpus_input"))
    corpus = _path(root, _required(paths, "corpus"))
    fulltext = _path(root, _required(paths, "fulltext_index"))
    document_embeddings = _path(root, _required(paths, "document_embeddings"))
    atlas = _path(root, _required(paths, "atlas"))
    dataset = _path(root, _required(paths, "dataset"))
    query_embeddings = _path(root, _required(paths, "query_embeddings"))
    router_model = _path(root, _required(paths, "router_model"))
    runtime = _path(root, _required(paths, "runtime"))
    state = _path(root, _required(paths, "state"))
    stages: dict[str, Any] = {}
    if not (corpus / "manifest.json").is_file():
        stages["corpus"] = prepare_corpus(corpus_input, corpus, int(cfg.get("corpus", {}).get("batch_size", 1024)))
    else:
        stages["corpus"] = {"status": "existing"}
    if not fulltext.is_file():
        stages["fulltext"] = build_fulltext_index(corpus, fulltext)
    else:
        stages["fulltext"] = {"status": "existing"}
    key_env = str(embedding.get("api_key_env", "ATLASNAV_EMBEDDING_API_KEY"))
    key = os.getenv(key_env)
    if (not (document_embeddings / "manifest.json").is_file()
            or not (query_embeddings / "manifest.json").is_file()) and not key:
        raise ValueError(f"{key_env} is required for unfinished embedding stages")
    client = EmbeddingClient(
        str(_required(embedding, "base_url")), key or "not-used",
        str(embedding.get("model", "qwen3.7-text-embedding")),
        float(embedding.get("timeout", 180.0)), int(embedding.get("maximum_retries", 12)),
    )
    try:
        common = {
            "dimensions": int(embedding.get("dimensions", 2560)),
            "batch_size": int(embedding.get("batch_size", 20)),
            "maximum_concurrency": int(embedding.get("concurrency", 30)),
            "ramp_start": int(embedding.get("ramp_start", 2)),
            "ramp_seconds": float(embedding.get("ramp_seconds", 180.0)),
        }
        if not (document_embeddings / "manifest.json").is_file():
            stages["document_embeddings"] = await build_embeddings(corpus, document_embeddings, client, **common)
        else:
            stages["document_embeddings"] = {"status": "existing"}
        if not (query_embeddings / "manifest.json").is_file():
            stages["query_embeddings"] = await build_query_embeddings(dataset, query_embeddings, client, **common)
        else:
            stages["query_embeddings"] = {"status": "existing"}
    finally:
        await client.close()
    if not (atlas / "manifest.json").is_file():
        stages["atlas"] = build_atlas(
            document_embeddings, atlas,
            pca_dimensions=int(atlas_cfg.get("pca_dimensions", 192)),
            pca_training_rows=int(atlas_cfg.get("pca_training_rows", 50_000)),
            neighbors=int(atlas_cfg.get("neighbors", 48)),
            parent_workers=int(atlas_cfg.get("parent_workers", 1)),
            child_workers=int(atlas_cfg.get("child_workers", 1)),
        )
    else:
        stages["atlas"] = {"status": "existing"}
    arrays_value = paths.get("router_arrays")
    if not router_model.is_file():
        if not arrays_value:
            raise ValueError("router_model is absent and paths.router_arrays was not supplied")
        arrays_path = _path(root, str(arrays_value))
        if not (arrays_path / "manifest.json").is_file():
            router_cfg = cfg.get("router") or {}
            required_router_paths = {
                name: _path(root, _required(paths, name))
                for name in ("router_tasks", "router_questions", "router_query_embeddings")
            }
            from atlasnav.router.tasks import build_support_tasks
            from atlasnav.router.synthesize import synthesize_questions
            from atlasnav.router.arrays import build_training_arrays

            task_path = required_router_paths["router_tasks"]
            question_path = required_router_paths["router_questions"]
            router_query_embeddings = required_router_paths["router_query_embeddings"]
            if not (task_path / "manifest.json").is_file():
                stages["router_tasks"] = build_support_tasks(
                    corpus=corpus, atlas=atlas, output=task_path,
                    tasks=int(router_cfg.get("candidate_tasks", 30_000)),
                    candidate_per_leaf=int(router_cfg.get("candidate_per_leaf", 72)),
                    single_fraction=float(router_cfg.get("single_fraction", 0.25)),
                    pair_fraction=float(router_cfg.get("pair_fraction", 0.50)),
                    seed=int(router_cfg.get("seed", 41_200)),
                )
            else:
                stages["router_tasks"] = {"status": "existing"}
            if not (question_path / "manifest.json").is_file():
                synthesis_key_env = str(router_cfg.get("synthesis_api_key_env", "ATLASNAV_SYNTHESIS_API_KEY"))
                stages["router_questions"] = await synthesize_questions(
                    tasks_directory=task_path, output=question_path,
                    base_url=str(_required(router_cfg, "synthesis_base_url")),
                    api_key_env=synthesis_key_env,
                    model=str(_required(router_cfg, "synthesis_model")),
                    concurrency=int(router_cfg.get("synthesis_concurrency", 20)),
                    batch_size=int(router_cfg.get("synthesis_batch_size", 3)),
                    retries=int(router_cfg.get("synthesis_maximum_retries", 8)),
                )
            else:
                stages["router_questions"] = {"status": "existing"}
            if not (router_query_embeddings / "manifest.json").is_file():
                embedding_key = os.getenv(key_env)
                if not embedding_key:
                    raise ValueError(f"{key_env} is required for Router question embeddings")
                router_client = EmbeddingClient(
                    str(_required(embedding, "base_url")), embedding_key,
                    str(embedding.get("model", "qwen3.7-text-embedding")),
                    float(embedding.get("timeout", 180.0)), int(embedding.get("maximum_retries", 12)),
                )
                try:
                    stages["router_query_embeddings"] = await build_query_embeddings(
                        question_path / "questions.jsonl", router_query_embeddings, router_client,
                        dimensions=int(embedding.get("dimensions", 2560)),
                        batch_size=int(embedding.get("batch_size", 20)),
                        maximum_concurrency=int(embedding.get("concurrency", 30)),
                        ramp_start=int(embedding.get("ramp_start", 2)),
                        ramp_seconds=float(embedding.get("ramp_seconds", 180.0)),
                    )
                finally:
                    await router_client.close()
            else:
                stages["router_query_embeddings"] = {"status": "existing"}
            stages["router_arrays"] = build_training_arrays(
                questions=question_path, query_embeddings=router_query_embeddings,
                atlas=atlas, fulltext_index=fulltext, output=arrays_path,
                candidates_per_query=int(router_cfg.get("candidates_per_query", 257)),
            )
        from argparse import Namespace
        stages["router"] = train(Namespace(
            array_dir=arrays_path, output_dir=router_model.parent,
            maxiter=int((cfg.get("router") or {}).get("maxiter", 100)),
        ))
        produced = router_model.parent / "router_model.npz"
        if produced != router_model:
            raise RuntimeError(f"trained Router was written to {produced}, runbook expects {router_model}")
    else:
        stages["router"] = {"status": "existing"}
    if not (runtime / "manifest.json").is_file():
        stages["runtime"] = build_runtime(
            corpus_directory=corpus, fulltext_index=fulltext, atlas_directory=atlas,
            query_bundle=query_embeddings, router_model=router_model, dataset=dataset,
            output_directory=runtime, state_directory=state,
            initial_parents=int(runtime_cfg.get("initial_parents", 10)),
            anchors_per_parent=int(runtime_cfg.get("anchors_per_parent", 3)),
        )
    else:
        stages["runtime"] = audit_runtime(runtime)
    report = {"schema": "atlasnav_build_pipeline_v1", "runbook": str(runbook), "stages": stages}
    report_path = _path(root, str(paths.get("pipeline_report", "pipeline-build.json")))
    atomic_json(report_path, report)
    return report


async def run_pipeline(runbook: Path) -> dict[str, Any]:
    """Run, judge, evaluate, and report a built Atlas runtime."""
    runbook = runbook.resolve()
    root = runbook.parent
    cfg = _load(runbook)
    paths = cfg.get("paths") or {}
    execution = cfg.get("execution") or {}
    runtime = _path(root, _required(paths, "runtime"))
    dataset = _path(root, _required(paths, "dataset"))
    runs = _path(root, _required(paths, "runs"))
    judgments = _path(root, _required(paths, "judgments"))
    evaluation = _path(root, _required(paths, "evaluation"))
    agent_profile = load_model_profile(_path(root, _required(paths, "agent_profile")))
    judge_profile = load_model_profile(_path(root, _required(paths, "judge_profile")))
    resources = Path(__file__).resolve().parent / "resources"
    run_result = await run_suite(
        runtime=runtime, output=runs, profile=agent_profile,
        system_prompt=_path(root, str(paths.get("system_prompt", resources / "atlasnav_system.txt"))),
        safe_release_prompt=_path(root, str(paths.get("safe_release_prompt", resources / "safe_release.txt"))),
        concurrency=int(execution.get("agent_concurrency", 20)),
        ramp_start=int(execution.get("agent_ramp_start", 2)),
        ramp_seconds=float(execution.get("agent_ramp_seconds", 120.0)),
        tool_timeout_seconds=float(execution.get("tool_timeout", 30.0)),
    )
    judge_result = await judge_suite(
        run_directory=runs, dataset=dataset, output=judgments, profile=judge_profile,
        concurrency=int(execution.get("judge_concurrency", 30)),
        ramp_start=int(execution.get("judge_ramp_start", 5)),
        ramp_seconds=float(execution.get("judge_ramp_seconds", 60.0)),
    )
    qrels = _path(root, str(paths["qrels"])) if paths.get("qrels") else None
    corpus = _path(root, str(paths["corpus"])) if paths.get("corpus") else None
    evaluate_result = evaluate_suite(
        run_directory=runs, judge_directory=judgments, output=evaluation,
        qrels=qrels, corpus=corpus,
        turn_checkpoints=list(map(int, execution.get("turn_checkpoints", [15, 30, 60, 120, 300]))),
        cost_checkpoints=list(map(float, execution.get("cost_checkpoints", []))),
    )
    markdown = _path(root, str(paths.get("markdown_report", "evaluation.md")))
    report_result = render_report(evaluation, markdown)
    result = {
        "schema": "atlasnav_execution_pipeline_v1", "run": run_result,
        "judge": judge_result, "evaluation": evaluate_result, "report": report_result,
    }
    atomic_json(_path(root, str(paths.get("execution_report", "pipeline-run.json"))), result)
    return result
