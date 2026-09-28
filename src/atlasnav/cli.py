"""AtlasNav command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tomllib
from pathlib import Path

from .artifacts import download_artifacts, verify_artifact_root
from .config import load_model_profile
from .doctor import doctor
from .errors import AtlasNavError


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _repository_root() -> Path:
    # `doctor` audits a release checkout, not the interpreter's site-packages.
    # Source/editable installs often make those the same tree; wheels do not.
    return Path.cwd().resolve()


def _prompt(name: str) -> Path:
    return Path(__file__).resolve().parent / "resources" / name


def command_doctor(args: argparse.Namespace) -> None:
    root = Path(args.repository_root).resolve()
    profiles = sorted((root / "configs/paper").glob("*.toml"))
    report = doctor(root, profiles)
    _json(report)
    if not report["valid_public_tree"]:
        raise SystemExit(2)


def command_profile(args: argparse.Namespace) -> None:
    value = load_model_profile(args.path)
    _json(
        {
            "id": value.profile_id,
            "model_id": value.model_id,
            "transport": value.transport,
            "context_window": value.context_window,
            "reasoning_effort": value.reasoning_effort,
            "pricing": value.pricing.__dict__,
            "safe_release": value.safe_release.__dict__,
            "base_url_configured": bool(value.base_url),
            "api_key_configured": bool(value.api_key),
        }
    )


def command_artifact_download(args: argparse.Namespace) -> None:
    repo_id = args.repo_id or os.getenv("ATLASNAV_ARTIFACT_REPO")
    if not repo_id:
        raise AtlasNavError("--repo-id or ATLASNAV_ARTIFACT_REPO is required")
    path = download_artifacts(repo_id, args.output, args.revision)
    _json({"schema": "atlasnav_artifact_download_v1", "local_dir": path})


def command_artifact_verify(args: argparse.Namespace) -> None:
    report = verify_artifact_root(args.artifact_root, deep=args.deep)
    _json(report)
    if not report["valid"]:
        raise SystemExit(2)


def command_corpus_prepare(args: argparse.Namespace) -> None:
    from .corpus.prepare import prepare_corpus

    _json(prepare_corpus(args.input, args.output, args.batch_size))


def command_corpus_index(args: argparse.Namespace) -> None:
    from .corpus.prepare import build_fulltext_index

    _json(build_fulltext_index(args.corpus, args.output))


def _benchmark_options(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    path = path.resolve()
    with path.open("rb") as stream:
        value = tomllib.load(stream) if path.suffix.casefold() == ".toml" else json.load(stream)
    if not isinstance(value, dict):
        raise AtlasNavError("benchmark config must contain one object/table")
    options = dict(value.get("benchmark", value))
    path_keys = {
        "raw_root", "source", "question_source", "generated_shards",
        "official_dev", "retrieval_dataset", "official_repository",
    }
    for key in path_keys & options.keys():
        options[key] = Path(str(options[key])).expanduser()
    if "text_corpus_sources" in options:
        options["text_corpus_sources"] = [Path(str(item)).expanduser()
                                            for item in options["text_corpus_sources"]]
    return options


def command_benchmark_prepare(args: argparse.Namespace) -> None:
    from .benchmarks import prepare_benchmark

    _json(prepare_benchmark(args.adapter, args.output, **_benchmark_options(args.config)))


def command_benchmark_audit(args: argparse.Namespace) -> None:
    from .benchmarks.common import audit_bundle

    report = audit_bundle(args.bundle)
    _json(report)
    if not report["passed"]:
        raise SystemExit(2)


def command_phantomwiki_generate(args: argparse.Namespace) -> None:
    from .benchmarks.phantomwiki_generate import generate_world_shards

    _json(generate_world_shards(
        official_repository=args.official_repository, output=args.output,
        world_start=args.world_start, world_stop=args.world_stop,
        expected_revision=args.expected_revision, trees_per_world=args.trees_per_world,
    ))


def command_benchmark_export(args: argparse.Namespace) -> None:
    from .benchmarks.results import export_predictions

    _json(export_predictions(args.bundle, args.run_dir, args.output))


def command_benchmark_score(args: argparse.Namespace) -> None:
    from .benchmarks.results import score_official

    _json(score_official(args.bundle, args.run_dir, args.output))


def command_enterprise_evaluate(args: argparse.Namespace) -> None:
    from .benchmarks.enterprise_eval import evaluate_frozen, evaluate_with_selector

    common = dict(questions=args.questions, answers=args.answers, results=args.results, output=args.output)
    if args.api_key_env:
        key = os.getenv(args.api_key_env)
        if not key:
            raise AtlasNavError(f"environment variable {args.api_key_env} is required")
        if args.audit is None or args.catalog is None or not args.base_url or not args.model:
            raise AtlasNavError("LLM document-selection mode requires --audit, --catalog, --base-url, and --model")
        value = evaluate_with_selector(
            **common, audit=args.audit, catalog=args.catalog, base_url=args.base_url,
            api_key=key, model=args.model, workers=args.workers,
            candidate_limit=args.candidate_limit,
        )
    else:
        value = evaluate_frozen(**common, selection_results=args.selection_results)
    _json(value)


def command_build_embeddings(args: argparse.Namespace) -> None:
    from .embeddings.build import build_embeddings
    from .embeddings.client import EmbeddingClient

    key = os.getenv(args.api_key_env)
    if not key:
        raise AtlasNavError(f"environment variable {args.api_key_env} is required")
    client = EmbeddingClient(args.base_url, key, args.model, args.timeout, args.maximum_retries)

    async def run() -> object:
        try:
            return await build_embeddings(
                args.corpus,
                args.output,
                client,
                dimensions=args.dimensions,
                batch_size=args.batch_size,
                maximum_concurrency=args.concurrency,
                ramp_start=args.ramp_start,
                ramp_seconds=args.ramp_seconds,
            )
        finally:
            await client.close()

    _json(asyncio.run(run()))


def command_build_atlas(args: argparse.Namespace) -> None:
    from .atlas.build import build_atlas

    _json(build_atlas(
        args.embeddings,
        args.output,
        pca_dimensions=args.pca_dimensions,
        pca_training_rows=args.pca_training_rows,
        neighbors=args.neighbors,
        parent_workers=args.parent_workers,
        child_workers=args.child_workers,
    ))


def command_build_query_embeddings(args: argparse.Namespace) -> None:
    from .embeddings.client import EmbeddingClient
    from .embeddings.queries import build_query_embeddings

    key = os.getenv(args.api_key_env)
    if not key:
        raise AtlasNavError(f"environment variable {args.api_key_env} is required")
    client = EmbeddingClient(args.base_url, key, args.model, args.timeout, args.maximum_retries)

    async def run() -> object:
        try:
            return await build_query_embeddings(
                args.dataset, args.output, client,
                dimensions=args.dimensions, batch_size=args.batch_size,
                maximum_concurrency=args.concurrency, ramp_start=args.ramp_start,
                ramp_seconds=args.ramp_seconds,
            )
        finally:
            await client.close()

    _json(asyncio.run(run()))


def command_router_train(args: argparse.Namespace) -> None:
    from .router.train import audit, train

    namespace = argparse.Namespace(
        array_dir=args.arrays,
        output_dir=args.output,
        maxiter=args.maxiter,
        audit_only=args.audit_only,
    )
    _json(audit(args.output.resolve()) if args.audit_only else train(namespace))


def command_router_build_tasks(args: argparse.Namespace) -> None:
    from .router.tasks import build_support_tasks

    _json(build_support_tasks(
        corpus=args.corpus, atlas=args.atlas, output=args.output,
        tasks=args.tasks, candidate_per_leaf=args.candidate_per_leaf,
        single_fraction=args.single_fraction, pair_fraction=args.pair_fraction,
        seed=args.seed,
    ))


def command_router_synthesize(args: argparse.Namespace) -> None:
    from .router.synthesize import synthesize_questions

    _json(asyncio.run(synthesize_questions(
        tasks_directory=args.tasks, output=args.output, base_url=args.base_url,
        api_key_env=args.api_key_env, model=args.model,
        concurrency=args.concurrency, batch_size=args.batch_size, retries=args.maximum_retries,
    )))


def command_router_build_arrays(args: argparse.Namespace) -> None:
    from .router.arrays import audit_training_arrays, build_training_arrays

    if args.audit_only:
        report = audit_training_arrays(args.output)
    else:
        report = build_training_arrays(
            questions=args.questions, query_embeddings=args.query_embeddings,
            atlas=args.atlas, fulltext_index=args.fulltext_index, output=args.output,
            candidates_per_query=args.candidates,
        )
    _json(report)
    if not report.get("passed", True):
        raise SystemExit(2)


def command_runtime_build(args: argparse.Namespace) -> None:
    from .runtime.build import build_runtime

    _json(build_runtime(
        corpus_directory=args.corpus,
        fulltext_index=args.fulltext_index,
        atlas_directory=args.atlas,
        query_bundle=args.query_embeddings,
        router_model=args.router_model,
        frozen_ranking=args.frozen_ranking,
        dataset=args.dataset,
        output_directory=args.output,
        state_directory=args.state,
        initial_parents=args.initial_parents,
        anchors_per_parent=args.anchors_per_parent,
    ))


def command_runtime_audit(args: argparse.Namespace) -> None:
    from .runtime.build import audit_runtime

    report = audit_runtime(args.output)
    _json(report)
    if not report["passed"]:
        raise SystemExit(2)


def command_reproduce(args: argparse.Namespace) -> None:
    from .reproduce import reproduce

    _json(reproduce(args.artifact_root, args.output, args.suite))


def command_run(args: argparse.Namespace) -> None:
    from .runtime.agent import run_suite

    profile = load_model_profile(args.profile)
    _json(asyncio.run(run_suite(
        runtime=args.runtime, output=args.output, profile=profile,
        system_prompt=args.system_prompt, safe_release_prompt=args.safe_release_prompt,
        concurrency=args.concurrency, ramp_start=args.ramp_start,
        ramp_seconds=args.ramp_seconds, tool_timeout_seconds=args.tool_timeout,
        limit=args.limit,
    )))


def command_judge(args: argparse.Namespace) -> None:
    from .runtime.judge import judge_suite

    profile = load_model_profile(args.profile)
    _json(asyncio.run(judge_suite(
        run_directory=args.run_dir, dataset=args.dataset, output=args.output,
        profile=profile, concurrency=args.concurrency, ramp_start=args.ramp_start,
        ramp_seconds=args.ramp_seconds,
    )))


def command_evaluate(args: argparse.Namespace) -> None:
    from .evaluation.evaluate import evaluate_suite

    _json(evaluate_suite(
        run_directory=args.run_dir, judge_directory=args.judge_dir,
        output=args.output, qrels=args.qrels, corpus=args.corpus,
        turn_checkpoints=args.turn_checkpoints, cost_checkpoints=args.cost_checkpoints,
    ))


def command_report(args: argparse.Namespace) -> None:
    from .evaluation.report import render_report

    _json(render_report(args.evaluation, args.output))


def command_pipeline_build(args: argparse.Namespace) -> None:
    from .pipeline import build_pipeline

    _json(asyncio.run(build_pipeline(args.config)))


def command_pipeline_run(args: argparse.Namespace) -> None:
    from .pipeline import run_pipeline

    _json(asyncio.run(run_pipeline(args.config)))


def parser() -> argparse.ArgumentParser:
    root = _repository_root()
    result = argparse.ArgumentParser(prog="atlasnav", description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="audit environment and public release tree")
    doctor_parser.add_argument("--repository-root", default=root)
    doctor_parser.set_defaults(handler=command_doctor)

    profile_parser = sub.add_parser("profile", help="validate and display a model profile")
    profile_parser.add_argument("path", type=Path)
    profile_parser.set_defaults(handler=command_profile)

    artifact = sub.add_parser("artifacts", help="download and verify frozen research artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    download = artifact_sub.add_parser("download")
    download.add_argument("--repo-id")
    download.add_argument("--revision")
    download.add_argument("--output", type=Path, default=Path("artifacts/release"))
    download.set_defaults(handler=command_artifact_download)
    verify = artifact_sub.add_parser("verify")
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--deep", action="store_true", help="scan archive structure and sensitive markers")
    verify.set_defaults(handler=command_artifact_verify)

    corpus = sub.add_parser("corpus", help="prepare canonical documents and indexes")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    prepare = corpus_sub.add_parser("prepare", help="normalize JSONL or Parquet documents")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=1024)
    prepare.set_defaults(handler=command_corpus_prepare)
    index = corpus_sub.add_parser("build-index", help="build a full-corpus SQLite FTS index")
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.set_defaults(handler=command_corpus_index)

    benchmarks = sub.add_parser(
        "benchmarks", help="prepare leakage-isolated auxiliary benchmark bundles"
    )
    benchmarks_sub = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_prepare = benchmarks_sub.add_parser(
        "prepare", help="adapt upstream benchmark data to the shared AtlasNav contract"
    )
    benchmark_prepare.add_argument(
        "--adapter", required=True,
        choices=("phantomwiki", "enterpriserag", "fanoutqa", "2wiki", "trec-covid",
                 "scifact", "arguana", "beir"),
    )
    benchmark_prepare.add_argument("--config", type=Path, required=True)
    benchmark_prepare.add_argument("--output", type=Path, required=True)
    benchmark_prepare.set_defaults(handler=command_benchmark_prepare)
    benchmark_audit = benchmarks_sub.add_parser("audit", help="audit a prepared benchmark bundle")
    benchmark_audit.add_argument("--bundle", type=Path, required=True)
    benchmark_audit.set_defaults(handler=command_benchmark_audit)
    phantom_generate = benchmarks_sub.add_parser(
        "generate-phantomwiki", help="generate resumable independent distractor worlds"
    )
    phantom_generate.add_argument("--official-repository", type=Path, required=True)
    phantom_generate.add_argument("--output", type=Path, required=True)
    phantom_generate.add_argument("--world-start", type=int, default=20)
    phantom_generate.add_argument("--world-stop", type=int, default=200)
    phantom_generate.add_argument(
        "--expected-revision", default="5542d6b2ba9a76cc62bd107b71934f68339ade34"
    )
    phantom_generate.add_argument("--trees-per-world", type=int, default=100)
    phantom_generate.set_defaults(handler=command_phantomwiki_generate)
    benchmark_export = benchmarks_sub.add_parser(
        "export", help="export final answers and successfully opened canonical document IDs"
    )
    benchmark_export.add_argument("--bundle", type=Path, required=True)
    benchmark_export.add_argument("--run-dir", type=Path, required=True)
    benchmark_export.add_argument("--output", type=Path, required=True)
    benchmark_export.set_defaults(handler=command_benchmark_export)
    benchmark_score = benchmarks_sub.add_parser(
        "score", help="compute deterministic official retrieval or FanOutQA metrics"
    )
    benchmark_score.add_argument("--bundle", type=Path, required=True)
    benchmark_score.add_argument("--run-dir", type=Path, required=True)
    benchmark_score.add_argument("--output", type=Path, required=True)
    benchmark_score.set_defaults(handler=command_benchmark_score)

    enterprise = sub.add_parser("enterprise", help="evaluate EnterpriseRAG-Bench frozen answers or document selection")
    enterprise_sub = enterprise.add_subparsers(dest="enterprise_command", required=True)
    enterprise_eval = enterprise_sub.add_parser("evaluate")
    enterprise_eval.add_argument("--questions", type=Path, required=True)
    enterprise_eval.add_argument("--answers", type=Path, required=True)
    enterprise_eval.add_argument("--results", type=Path, required=True)
    enterprise_eval.add_argument("--output", type=Path, required=True)
    enterprise_eval.add_argument("--selection-results", type=Path, help="use stored document metrics without an API")
    enterprise_eval.add_argument("--audit", type=Path)
    enterprise_eval.add_argument("--catalog", type=Path)
    enterprise_eval.add_argument("--base-url")
    enterprise_eval.add_argument("--model")
    enterprise_eval.add_argument("--api-key-env", help="set to enable LLM-backed document selection")
    enterprise_eval.add_argument("--workers", type=int, default=8)
    enterprise_eval.add_argument("--candidate-limit", type=int, default=40)
    enterprise_eval.set_defaults(handler=command_enterprise_evaluate)

    build = sub.add_parser("build", help="construct embeddings and the persistent Atlas")
    build_sub = build.add_subparsers(dest="build_command", required=True)
    embeddings = build_sub.add_parser("embeddings", help="encode four document views")
    embeddings.add_argument("--corpus", type=Path, required=True)
    embeddings.add_argument("--output", type=Path, required=True)
    embeddings.add_argument("--base-url", required=True)
    embeddings.add_argument("--api-key-env", default="ATLASNAV_EMBEDDING_API_KEY")
    embeddings.add_argument("--model", default="qwen3.7-text-embedding")
    embeddings.add_argument("--dimensions", type=int, default=2560)
    embeddings.add_argument("--batch-size", type=int, default=20)
    embeddings.add_argument("--concurrency", type=int, default=30)
    embeddings.add_argument("--ramp-start", type=int, default=2)
    embeddings.add_argument("--ramp-seconds", type=float, default=180.0)
    embeddings.add_argument("--timeout", type=float, default=180.0)
    embeddings.add_argument("--maximum-retries", type=int, default=12)
    embeddings.set_defaults(handler=command_build_embeddings)
    atlas = build_sub.add_parser("atlas", help="build sparse view graphs and a multiplex hierarchy")
    atlas.add_argument("--embeddings", type=Path, required=True)
    atlas.add_argument("--output", type=Path, required=True)
    atlas.add_argument("--pca-dimensions", type=int, default=192)
    atlas.add_argument("--pca-training-rows", type=int, default=50_000)
    atlas.add_argument("--neighbors", type=int, default=48)
    atlas.add_argument("--parent-workers", type=int, default=1)
    atlas.add_argument("--child-workers", type=int, default=1)
    atlas.set_defaults(handler=command_build_atlas)
    query_embeddings = build_sub.add_parser(
        "query-embeddings", help="encode four query views for deployment"
    )
    query_embeddings.add_argument("--dataset", type=Path, required=True)
    query_embeddings.add_argument("--output", type=Path, required=True)
    query_embeddings.add_argument("--base-url", required=True)
    query_embeddings.add_argument("--api-key-env", default="ATLASNAV_EMBEDDING_API_KEY")
    query_embeddings.add_argument("--model", default="qwen3.7-text-embedding")
    query_embeddings.add_argument("--dimensions", type=int, default=2560)
    query_embeddings.add_argument("--batch-size", type=int, default=20)
    query_embeddings.add_argument("--concurrency", type=int, default=30)
    query_embeddings.add_argument("--ramp-start", type=int, default=2)
    query_embeddings.add_argument("--ramp-seconds", type=float, default=60.0)
    query_embeddings.add_argument("--timeout", type=float, default=180.0)
    query_embeddings.add_argument("--maximum-retries", type=int, default=12)
    query_embeddings.set_defaults(handler=command_build_query_embeddings)

    router = sub.add_parser("router", help="construct or train query-adaptive routing")
    router_sub = router.add_subparsers(dest="router_command", required=True)
    tasks = router_sub.add_parser("build-tasks", help="construct corpus-only single/pair/triple support tasks")
    tasks.add_argument("--corpus", type=Path, required=True)
    tasks.add_argument("--atlas", type=Path, required=True)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--tasks", type=int, default=30_000)
    tasks.add_argument("--candidate-per-leaf", type=int, default=72)
    tasks.add_argument("--single-fraction", type=float, default=0.25)
    tasks.add_argument("--pair-fraction", type=float, default=0.50)
    tasks.add_argument("--seed", type=int, default=41_200)
    tasks.set_defaults(handler=command_router_build_tasks)
    synthesize = router_sub.add_parser(
        "synthesize", help="generate and independently verify grounded Router questions"
    )
    synthesize.add_argument("--tasks", type=Path, required=True)
    synthesize.add_argument("--output", type=Path, required=True)
    synthesize.add_argument("--base-url", required=True)
    synthesize.add_argument("--api-key-env", default="ATLASNAV_SYNTHESIS_API_KEY")
    synthesize.add_argument("--model", required=True)
    synthesize.add_argument("--concurrency", type=int, default=20)
    synthesize.add_argument("--batch-size", type=int, default=3)
    synthesize.add_argument("--maximum-retries", type=int, default=8)
    synthesize.set_defaults(handler=command_router_synthesize)
    arrays = router_sub.add_parser(
        "build-arrays", help="build 816-D multi-positive Router training arrays"
    )
    arrays.add_argument("--questions", type=Path, required=True)
    arrays.add_argument("--query-embeddings", type=Path, required=True)
    arrays.add_argument("--atlas", type=Path, required=True)
    arrays.add_argument("--fulltext-index", type=Path, required=True)
    arrays.add_argument("--output", type=Path, required=True)
    arrays.add_argument("--candidates", type=int, default=257)
    arrays.add_argument("--audit-only", action="store_true")
    arrays.set_defaults(handler=command_router_build_arrays)
    train = router_sub.add_parser("train")
    train.add_argument("--arrays", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--maxiter", type=int, default=100)
    train.add_argument("--audit-only", action="store_true")
    train.set_defaults(handler=command_router_train)

    runtime = sub.add_parser("runtime", help="build or audit query workspaces")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_build = runtime_sub.add_parser("build")
    runtime_build.add_argument("--corpus", type=Path, required=True)
    runtime_build.add_argument("--fulltext-index", type=Path, required=True)
    runtime_build.add_argument("--atlas", type=Path, required=True)
    ranking_mode = runtime_build.add_mutually_exclusive_group(required=True)
    ranking_mode.add_argument("--query-embeddings", type=Path)
    ranking_mode.add_argument(
        "--frozen-ranking", type=Path,
        help="reuse an exact checksummed post-Router score bundle",
    )
    runtime_build.add_argument("--router-model", type=Path)
    runtime_build.add_argument("--dataset", type=Path, required=True)
    runtime_build.add_argument("--output", type=Path, required=True)
    runtime_build.add_argument("--state", type=Path, required=True)
    runtime_build.add_argument("--initial-parents", type=int, default=10)
    runtime_build.add_argument("--anchors-per-parent", type=int, default=3)
    runtime_build.set_defaults(handler=command_runtime_build)
    runtime_audit = runtime_sub.add_parser("audit")
    runtime_audit.add_argument("--output", type=Path, required=True)
    runtime_audit.set_defaults(handler=command_runtime_audit)

    reproduce_parser = sub.add_parser("reproduce", help="recompute frozen paper results without API calls")
    reproduce_parser.add_argument("--suite", choices=("paper", "browsecomp_plus"), default="paper")
    reproduce_parser.add_argument("--artifact-root", type=Path, required=True)
    reproduce_parser.add_argument("--output", type=Path, default=Path("reproduced"))
    reproduce_parser.set_defaults(handler=command_reproduce)

    run = sub.add_parser("run", help="run the AtlasNav agent with durable incremental trajectories")
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--system-prompt", type=Path, default=_prompt("atlasnav_system.txt"))
    run.add_argument("--safe-release-prompt", type=Path, default=_prompt("safe_release.txt"))
    run.add_argument("--concurrency", type=int, default=20)
    run.add_argument("--ramp-start", type=int, default=2)
    run.add_argument("--ramp-seconds", type=float, default=120.0)
    run.add_argument("--tool-timeout", type=float, default=30.0)
    run.add_argument("--limit", type=int)
    run.set_defaults(handler=command_run)

    judge = sub.add_parser("judge", help="judge durable agent terminals")
    judge.add_argument("--run-dir", type=Path, required=True)
    judge.add_argument("--dataset", type=Path, required=True)
    judge.add_argument("--profile", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--concurrency", type=int, default=30)
    judge.add_argument("--ramp-start", type=int, default=5)
    judge.add_argument("--ramp-seconds", type=float, default=60.0)
    judge.set_defaults(handler=command_judge)

    evaluate = sub.add_parser("evaluate", help="compute endpoint, checkpoints, and Evidence Blindness")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--judge-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--qrels", type=Path)
    evaluate.add_argument("--corpus", type=Path)
    evaluate.add_argument("--turn-checkpoints", type=int, nargs="*", default=[15, 30, 60, 120, 300])
    evaluate.add_argument("--cost-checkpoints", type=float, nargs="*", default=[])
    evaluate.set_defaults(handler=command_evaluate)

    report = sub.add_parser("report", help="render a stable Markdown evaluation report")
    report.add_argument("--evaluation", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=command_report)

    pipeline = sub.add_parser("pipeline", help="execute a declarative end-to-end pipeline")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command", required=True)
    pipeline_build = pipeline_sub.add_parser("build", help="build all offline assets and runtime")
    pipeline_build.add_argument("--config", type=Path, required=True)
    pipeline_build.set_defaults(handler=command_pipeline_build)
    pipeline_run = pipeline_sub.add_parser("run", help="run, judge, evaluate, and report")
    pipeline_run.add_argument("--config", type=Path, required=True)
    pipeline_run.set_defaults(handler=command_pipeline_run)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.handler(args)
    except AtlasNavError as error:
        raise SystemExit(f"atlasnav: {error}") from error
