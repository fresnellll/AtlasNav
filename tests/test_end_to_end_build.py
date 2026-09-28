from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from atlasnav.atlas.build import build_atlas
from atlasnav.atlas.signatures import VIEWS
from atlasnav.corpus.prepare import build_fulltext_index, prepare_corpus
from atlasnav.io import atomic_json, sha256_file, stable_json
from atlasnav.router.features import FEATURE_DIMENSIONS
from atlasnav.router.arrays import audit_training_arrays, build_training_arrays
from atlasnav.runtime.build import audit_runtime, build_runtime


def test_synthetic_corpus_to_runtime(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for index in range(48):
            stream.write(stable_json({
                "docid": str(index),
                "text": (
                    f"Entity {index} participated in Episode {index % 7} in year {1900 + index}. "
                    f"It has relation group {index % 5} and topic family {index % 6}. "
                    f"The canonical verification code is value-{index}."
                ),
                "url": f"https://example.invalid/{index}",
            }) + "\n")
    corpus = tmp_path / "corpus"
    prepare_corpus(source, corpus)
    fulltext = tmp_path / "fulltext.sqlite3"
    build_fulltext_index(corpus, fulltext)

    embedding_directory = tmp_path / "embeddings"
    embedding_directory.mkdir()
    with gzip.open(corpus / "catalog.jsonl.gz", "rt", encoding="utf-8") as source_catalog:
        catalog_rows = [json.loads(line) for line in source_catalog]
    catalog = embedding_directory / "catalog.jsonl.gz"
    with gzip.open(catalog, "wt", encoding="utf-8") as stream:
        for row in catalog_rows:
            stream.write(stable_json(row) + "\n")
    rng = np.random.default_rng(41041)
    arrays, hashes = {}, {}
    for view in VIEWS:
        path = embedding_directory / f"{view}.f16.npy"
        np.save(path, rng.normal(size=(48, 192)).astype(np.float16))
        arrays[view], hashes[view] = path.name, sha256_file(path)
    atomic_json(embedding_directory / "manifest.json", {
        "schema": "atlasnav_four_view_embeddings_v1", "finalized": True,
        "documents": 48, "views": list(VIEWS), "model": "synthetic",
        "dimensions_per_view": 192, "vectors_concatenated_or_averaged": False,
        "construction_reads_evaluation_artifacts": False,
        "corpus_sha256": json.loads((corpus / "manifest.json").read_text())["documents_sha256"],
        "catalog": catalog.name, "catalog_sha256": sha256_file(catalog),
        "view_embeddings": arrays, "view_embedding_sha256": hashes,
    })
    atlas = tmp_path / "atlas"
    build_atlas(
        embedding_directory, atlas, pca_dimensions=192, pca_training_rows=48,
        neighbors=6, parent_count_prior=6,
        parent_resolutions=(0.1, 0.5, 1.0, 2.0),
        child_resolution_multipliers=(0.5, 1.0, 2.0), representatives=3,
    )

    dataset = tmp_path / "questions.jsonl"
    query_row = {
        "query_id": "q1", "query": "What is the canonical verification code for Entity 7?",
        "answer": "value-7",
        "kind": "single", "split": "train", "positive_file_indices": [7],
    }
    dataset.write_text(stable_json(query_row) + "\n", encoding="utf-8")
    query_directory = tmp_path / "query_embeddings"
    query_directory.mkdir()
    query_arrays, query_hashes = {}, {}
    for view in VIEWS:
        path = query_directory / f"{view}.f16.npy"
        np.save(path, rng.normal(size=(1, 192)).astype(np.float16))
        query_arrays[view], query_hashes[view] = path.name, sha256_file(path)
    query_catalog = query_directory / "catalog.jsonl.gz"
    with gzip.open(query_catalog, "wt", encoding="utf-8") as stream:
        stream.write(stable_json({
            "index": 0, "query_id": "q1",
            "query": "What is the canonical verification code for Entity 7?",
        }) + "\n")
    atomic_json(query_directory / "manifest.json", {
        "schema": "atlasnav_four_view_query_embeddings_v1", "finalized": True,
        "queries": 1, "views": list(VIEWS), "model": "synthetic",
        "dimensions_per_view": 192, "dataset_sha256": sha256_file(dataset),
        "query_catalog": query_catalog.name, "query_catalog_sha256": sha256_file(query_catalog),
        "query_embeddings": query_arrays, "query_embedding_sha256": query_hashes,
    })
    router_model = tmp_path / "router_model.npz"
    np.savez_compressed(
        router_model,
        mean=np.zeros(FEATURE_DIMENSIONS, dtype=np.float32),
        scale=np.ones(FEATURE_DIMENSIONS, dtype=np.float32),
        facet_coefficients=np.zeros((4, FEATURE_DIMENSIONS), dtype=np.float32),
        facet_intercept=np.zeros(4, dtype=np.float32),
        eta_coefficients=np.zeros(FEATURE_DIMENSIONS, dtype=np.float32),
        eta_intercept=np.float32(0),
        confidence_coefficients=np.zeros(FEATURE_DIMENSIONS, dtype=np.float32),
        confidence_intercept=np.float32(0),
        facet_temperature=np.float32(1), eta_temperature=np.float32(1),
        confidence_temperature=np.float32(1), confidence_bias=np.float32(0),
    )
    runtime, state = tmp_path / "runtime", tmp_path / "state"
    build_runtime(
        corpus_directory=corpus, fulltext_index=fulltext, atlas_directory=atlas,
        query_bundle=query_directory, router_model=router_model, dataset=dataset,
        output_directory=runtime, state_directory=state,
        initial_parents=3, anchors_per_parent=2,
    )
    assert audit_runtime(runtime)["passed"] is True
    assert (runtime / "workspaces/q1/ROUTE.tsv").is_file()
    assert (runtime / "workspaces/q1/.atlas_query.json").is_file()

    frozen = tmp_path / "frozen_ranking"
    frozen.mkdir()
    frozen_scores = frozen / "query_file_scores.f32.npy"
    frozen_scores.write_bytes((runtime / "query_file_scores.f32.npy").read_bytes())
    frozen_audit = frozen / "retrieval_audit.jsonl.gz"
    frozen_audit.write_bytes((runtime / "retrieval_audit.jsonl.gz").read_bytes())
    route = (runtime / "workspaces/q1/ROUTE.tsv").read_text(encoding="utf-8")
    route_metadata = json.loads(
        (runtime / "workspaces/q1/.atlas_query.json").read_text(encoding="utf-8")
    )
    frozen_routes = frozen / "workspace_routes.jsonl.gz"
    with gzip.open(frozen_routes, "wt", encoding="utf-8") as stream:
        stream.write(stable_json({
            "query_id": "q1", "route_tsv": route,
            "route_sha256": sha256_file(runtime / "workspaces/q1/ROUTE.tsv"),
            "parent_order": route_metadata["parent_order"],
            "initial_leads": route_metadata["initial_leads"],
        }) + "\n")
    frozen_catalog = frozen / "query_catalog.ids.jsonl.gz"
    with gzip.open(frozen_catalog, "wt", encoding="utf-8") as stream:
        stream.write(stable_json({"index": 0, "query_id": "q1"}) + "\n")
    atomic_json(frozen / "manifest.json", {
        "schema": "atlasnav_frozen_ranking_v1", "finalized": True,
        "queries": 1, "documents": 48,
        "query_scores": frozen_scores.name,
        "query_scores_sha256": sha256_file(frozen_scores),
        "retrieval_audit": frozen_audit.name,
        "retrieval_audit_sha256": sha256_file(frozen_audit),
        "workspace_routes": frozen_routes.name,
        "workspace_routes_sha256": sha256_file(frozen_routes),
        "query_catalog": frozen_catalog.name,
        "query_catalog_sha256": sha256_file(frozen_catalog),
        "router_model_sha256": sha256_file(router_model),
        "reads_answers_qrels_judgments_or_trajectories": False,
    })
    frozen_runtime, frozen_state = tmp_path / "frozen_runtime", tmp_path / "frozen_state"
    build_runtime(
        corpus_directory=corpus, fulltext_index=fulltext, atlas_directory=atlas,
        query_bundle=None, router_model=None, frozen_ranking=frozen, dataset=dataset,
        output_directory=frozen_runtime, state_directory=frozen_state,
        initial_parents=3, anchors_per_parent=2,
    )
    assert audit_runtime(frozen_runtime)["passed"] is True
    assert sha256_file(frozen_runtime / "query_file_scores.f32.npy") == sha256_file(
        runtime / "query_file_scores.f32.npy"
    )
    assert (frozen_runtime / "workspaces/q1/ROUTE.tsv").read_bytes() == (
        runtime / "workspaces/q1/ROUTE.tsv"
    ).read_bytes()

    router_questions = tmp_path / "router_questions"
    router_questions.mkdir()
    question_file = router_questions / "questions.jsonl"
    question_file.write_text(dataset.read_text(encoding="utf-8"), encoding="utf-8")
    atomic_json(router_questions / "manifest.json", {
        "schema": "atlasnav_verified_router_questions_v1", "finalized": True,
        "questions": 1, "questions_sha256": sha256_file(question_file),
        "selection_reads_evaluation_artifacts": False,
    })
    router_arrays = tmp_path / "router_arrays"
    build_training_arrays(
        questions=router_questions, query_embeddings=query_directory,
        atlas=atlas, fulltext_index=fulltext, output=router_arrays,
    )
    assert audit_training_arrays(router_arrays)["passed"] is True
    assert np.load(router_arrays / "features.f32.npy").shape == (1, FEATURE_DIMENSIONS)
