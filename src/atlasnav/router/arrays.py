"""Build outcome-blind multi-positive arrays for AtlasNav Router training."""

from __future__ import annotations

from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from atlasnav.atlas.graph import normalize_rows
from atlasnav.atlas.signatures import VIEWS
from atlasnav.io import atomic_json, sha256_file, stable_json
from atlasnav.router.features import (
    FEATURE_DIMENSIONS,
    assemble_features,
    calibration_indexes,
    view_meta_features,
)


CHANNELS = (*VIEWS, "bm25")
RRF_K = 60
DEFAULT_CANDIDATES = 257
LEXICAL_TOP = 256
SEED = 41_202
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"query_id", "query", "kind", "split", "positive_file_indices"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"invalid Router question at line {line_number}")
            positives = [int(value) for value in row["positive_file_indices"]]
            if not positives or len(positives) > 3 or len(set(positives)) != len(positives):
                raise ValueError(f"invalid positive set at line {line_number}")
            rows.append({**row, "positive_file_indices": positives})
    if not rows or len({str(row["query_id"]) for row in rows}) != len(rows):
        raise ValueError("Router questions must be non-empty with unique query IDs")
    return rows


def _asset(root: Path, manifest: dict[str, Any], key: str) -> Path:
    value = (manifest.get("arrays") or {}).get(key)
    if not value:
        matches = sorted(root.glob(f"{key}.*.npy"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Atlas array is missing: {key}")
    path = root / str(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _project(raw: np.ndarray, transform_path: Path) -> np.ndarray:
    parameters = np.load(transform_path)
    transform = np.asarray(parameters["transform"], dtype=np.float32)
    bias = np.asarray(parameters["bias"], dtype=np.float32)
    projected = np.asarray(raw, dtype=np.float32) @ transform.T
    if bias.size:
        projected += bias
    return normalize_rows(projected)


def _expression(query: str) -> str | None:
    values: list[str] = []
    for token in TOKEN_RE.findall(query):
        normalized = token.casefold()
        if len(normalized) < 2 or normalized in values:
            continue
        values.append(normalized)
        if len(values) == 10:
            break
    if not values:
        return None
    return " OR ".join('"' + value.replace('"', '""') + '"' for value in values)


def _lexical_hits(connection: sqlite3.Connection, query: str, top: int = LEXICAL_TOP) -> list[int]:
    expression = _expression(query)
    if not expression:
        return []
    rows = connection.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? "
        "ORDER BY bm25(documents_fts,1.0),rowid LIMIT ?",
        (expression, top),
    ).fetchall()
    return [int(row[0]) - 1 for row in rows]


def _deterministic(values: Iterable[int] | np.ndarray, seed: int) -> list[int]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.int64)
    if not len(array):
        return []
    rng = np.random.default_rng(seed)
    return array[rng.permutation(len(array))].astype(int).tolist()


def _graph_neighbours(atlas: Path, sources: np.ndarray, documents: int) -> dict[int, list[int]]:
    accumulated: dict[int, dict[int, float]] = {int(source): {} for source in sources}
    for view in VIEWS:
        graph = np.load(atlas / f"{view}_graph.npz")
        edges = np.asarray(graph["edges"], dtype=np.int32)
        weights = np.asarray(graph["weights"], dtype=np.float32)
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        columns = np.concatenate((edges[:, 1], edges[:, 0]))
        adjacency = sparse.csr_matrix(
            (np.concatenate((weights, weights)), (rows, columns)), shape=(documents, documents),
        )
        for source in sources:
            start, stop = adjacency.indptr[int(source):int(source) + 2]
            indexes, values = adjacency.indices[start:stop], adjacency.data[start:stop]
            target = accumulated[int(source)]
            for position in np.argsort(-values, kind="stable")[:64]:
                index = int(indexes[position])
                target[index] = max(target.get(index, -math.inf), float(values[position]))
        del adjacency, graph, edges, weights, rows, columns
    return {
        source: [index for index, _score in sorted(values.items(), key=lambda item: (-item[1], item[0]))]
        for source, values in accumulated.items()
    }


def _candidate_rows(
    rows: list[dict[str, Any]], atlas: Path, atlas_manifest: dict[str, Any],
    lexical: list[list[int]], documents: int, candidates_per_query: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    file_to_leaf = np.load(_asset(atlas, atlas_manifest, "file_to_leaf"), mmap_mode="r")
    file_to_parent = np.load(_asset(atlas, atlas_manifest, "file_to_parent"), mmap_mode="r")
    leaf_members = np.load(_asset(atlas, atlas_manifest, "leaf_members"), mmap_mode="r")
    leaf_offsets = np.load(_asset(atlas, atlas_manifest, "leaf_offsets"), mmap_mode="r")
    parent_members = np.load(_asset(atlas, atlas_manifest, "parent_members"), mmap_mode="r")
    parent_offsets = np.load(_asset(atlas, atlas_manifest, "parent_offsets"), mmap_mode="r")
    sources = np.asarray(sorted({value for row in rows for value in row["positive_file_indices"]}), dtype=np.int32)
    if np.any(sources < 0) or np.any(sources >= documents):
        raise ValueError("Router positive file index lies outside the Atlas")
    neighbours = _graph_neighbours(atlas, sources, documents)
    width = min(candidates_per_query, documents)
    candidates = np.empty((len(rows), width), dtype=np.int32)
    positive_mask = np.zeros((len(rows), width), dtype=bool)
    categories: Counter[str] = Counter()
    universe = np.arange(documents, dtype=np.int32)
    for query_index, row in enumerate(rows):
        positives = list(map(int, row["positive_file_indices"]))
        selected, selected_set = list(positives), set(positives)
        positive_mask[query_index, :len(positives)] = True

        def add(pool: Iterable[int] | np.ndarray, limit: int, category: str) -> None:
            added = 0
            for raw in pool:
                value = int(raw)
                if value in selected_set:
                    continue
                selected.append(value)
                selected_set.add(value)
                categories[category] += 1
                added += 1
                if added >= limit or len(selected) >= width:
                    return

        for source in positives:
            leaf, parent = int(file_to_leaf[source]), int(file_to_parent[source])
            same_leaf = np.asarray(leaf_members[leaf_offsets[leaf]:leaf_offsets[leaf + 1]])
            same_parent = np.asarray(parent_members[parent_offsets[parent]:parent_offsets[parent + 1]])
            same_parent = same_parent[np.asarray(file_to_leaf[same_parent]) != leaf]
            add(_deterministic(same_leaf, SEED + source * 17 + 1), 16, "same_leaf")
            add(_deterministic(same_parent, SEED + source * 17 + 2), 16, "same_parent_other_leaf")
            add(neighbours[source], 24, "graph")
        add(lexical[query_index], 64, "bm25")
        rng = np.random.default_rng(SEED + query_index * 1_000_003)
        sample = rng.choice(universe, size=min(1024, documents), replace=False)
        add(_deterministic(sample, SEED + query_index), width, "random")
        if len(selected) < width:
            add(_deterministic(universe, SEED + query_index + 7), width, "fallback")
        candidates[query_index] = np.asarray(selected[:width], dtype=np.int32)
    audit = {
        "candidates_per_query": width,
        "positive_cardinality": dict(Counter(len(row["positive_file_indices"]) for row in rows)),
        "negative_category_slots": dict(categories),
        "unique_per_row": bool(all(len(set(map(int, row))) == len(row) for row in candidates)),
    }
    return candidates, positive_mask, audit


def approximate_rrf(
    candidate_similarity: np.ndarray, calibration_similarity: np.ndarray, documents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate corpus ranks against a fixed outcome-blind calibration sample."""
    ranks = np.empty(candidate_similarity.shape, dtype=np.int32)
    for index in range(len(candidate_similarity)):
        ordered = np.sort(calibration_similarity[index])
        below = np.searchsorted(ordered, candidate_similarity[index], side="right")
        tail = len(ordered) - below
        ranks[index] = np.clip(
            1 + np.rint(tail * (documents - 1) / max(len(ordered), 1)).astype(np.int32),
            1, documents,
        )
    return 1.0 / (RRF_K + ranks), ranks


def channel_targets(ranks: np.ndarray, positive_mask: np.ndarray, rrf: np.ndarray) -> dict[str, np.ndarray]:
    """Derive corpus-only supervision for facets, lexical mix, and confidence."""
    queries = len(ranks)
    utility = np.empty((queries, len(CHANNELS)), dtype=np.float32)
    facet = np.empty((queries, len(VIEWS)), dtype=np.float32)
    eta = np.empty(queries, dtype=np.float32)
    confidence = np.empty(queries, dtype=np.float32)
    baseline_loss = np.empty(queries, dtype=np.float32)
    oracle_loss = np.empty(queries, dtype=np.float32)
    for query in range(queries):
        positive, negative = positive_mask[query], ~positive_mask[query]
        for channel in range(len(CHANNELS)):
            discount = 1.0 / np.log2(1.0 + ranks[query, positive, channel])
            utility[query, channel] = 0.5 * float(np.mean(discount)) + 0.5 * float(np.min(discount))
        dense = utility[query, :len(VIEWS)]
        values = np.exp((dense - np.max(dense)) / 0.06)
        facet[query] = values / np.sum(values)
        semantic, lexical = float(np.sum(facet[query] * dense)), float(utility[query, -1])
        eta[query] = float(np.clip(semantic / max(semantic + lexical, 1e-8), 0.20, 0.80))
        baseline_weights = np.asarray([0.25, 0.25, 0.25, 0.25, 1.0], dtype=np.float32)
        oracle_weights = np.concatenate((
            2.0 * eta[query] * facet[query], np.asarray([2.0 * (1.0 - eta[query])], dtype=np.float32),
        ))

        def loss(weights: np.ndarray) -> float:
            score = rrf[query] @ weights
            maximum = float(np.max(score[negative]))
            negative_anchor = float(
                np.log(np.sum(np.exp((score[negative] - maximum) / 0.002))) + maximum / 0.002
            )
            return float(np.mean(np.logaddexp(0.0, negative_anchor - score[positive] / 0.002)))

        baseline_loss[query], oracle_loss[query] = loss(baseline_weights), loss(oracle_weights)
        confidence[query] = float(np.clip((baseline_loss[query] - oracle_loss[query]) / 0.35, 0.0, 1.0))
    return {
        "channel_utility": utility, "facet_targets": facet, "eta_targets": eta,
        "confidence_targets": confidence, "baseline_loss": baseline_loss, "oracle_loss": oracle_loss,
    }


def build_training_arrays(
    *, questions: Path, query_embeddings: Path, atlas: Path, fulltext_index: Path,
    output: Path, candidates_per_query: int = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Build the exact deployable 816-D features and multi-positive candidate loss arrays."""
    questions, query_embeddings, atlas, fulltext_index, output = (
        Path(value).resolve() for value in (questions, query_embeddings, atlas, fulltext_index, output)
    )
    if output.exists():
        raise FileExistsError(f"refusing to replace Router training arrays: {output}")
    question_file = questions / "questions.jsonl"
    question_manifest = _read_json(questions / "manifest.json")
    embedding_manifest = _read_json(query_embeddings / "manifest.json")
    atlas_manifest = _read_json(atlas / "manifest.json")
    if question_manifest.get("schema") != "atlasnav_verified_router_questions_v1":
        raise RuntimeError("verified Router questions are required")
    if embedding_manifest.get("schema") != "atlasnav_four_view_query_embeddings_v1":
        raise RuntimeError("four-view Router query embeddings are required")
    if atlas_manifest.get("schema") not in {"atlasnav_multiplex_atlas_v1", "atlasnav_multiplex_atlas_frozen_v1"}:
        raise RuntimeError("a finalized Atlas is required")
    if sha256_file(question_file) != question_manifest.get("questions_sha256"):
        raise RuntimeError("Router question checksum mismatch")
    if embedding_manifest.get("dataset_sha256") != sha256_file(question_file):
        raise RuntimeError("Router questions and query embeddings differ")
    if question_manifest.get("selection_reads_evaluation_artifacts") is not False:
        raise RuntimeError("Router question leakage firewall failed")
    rows = _read_questions(question_file)
    documents = int(atlas_manifest["documents"])
    if len(rows) != int(embedding_manifest["queries"]):
        raise RuntimeError("Router question and embedding cardinalities differ")
    split_parents: dict[str, set[int]] = {name: set() for name in ("train", "validation", "test")}
    file_to_parent = np.load(_asset(atlas, atlas_manifest, "file_to_parent"), mmap_mode="r")
    file_to_leaf = np.load(_asset(atlas, atlas_manifest, "file_to_leaf"), mmap_mode="r")
    for row in rows:
        parents = {int(file_to_parent[index]) for index in row["positive_file_indices"]}
        split_parents[str(row["split"])].update(parents)
    if any(
        split_parents[left] & split_parents[right]
        for position, left in enumerate(("train", "validation", "test"))
        for right in ("train", "validation", "test")[position + 1:]
    ):
        raise RuntimeError("parent-group split leakage in Router questions")

    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    connection = sqlite3.connect(f"file:{fulltext_index}?mode=ro&immutable=1", uri=True)
    lexical: list[list[int]] = []
    try:
        for index, row in enumerate(rows):
            lexical.append(_lexical_hits(connection, str(row["query"])))
            if (index + 1) % 500 == 0:
                print(f"Router BM25 {index + 1:,}/{len(rows):,}", flush=True)
    finally:
        connection.close()
    candidates, positive_mask, candidate_audit = _candidate_rows(
        rows, atlas, atlas_manifest, lexical, documents, candidates_per_query,
    )
    np.save(temporary / "candidates.i32.npy", candidates)
    np.save(temporary / "positive_mask.bool.npy", positive_mask)
    calibration = calibration_indexes(documents)
    np.save(temporary / "calibration_files.i32.npy", calibration)
    dense_rrf = np.empty((len(rows), candidates.shape[1], len(VIEWS)), dtype=np.float32)
    dense_ranks = np.empty((len(rows), candidates.shape[1], len(VIEWS)), dtype=np.int32)
    meta_by_view: dict[str, np.ndarray] = {}
    query_vectors: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(VIEWS):
        pca = atlas_manifest["pca"]["assets"][view]
        raw_query = np.load(query_embeddings / embedding_manifest["query_embeddings"][view], mmap_mode="r")
        query_vectors[view] = _project(raw_query, atlas / pca["transform"])
        file_vectors = normalize_rows(
            np.asarray(np.load(atlas / pca["reduced"], mmap_mode="r"), dtype=np.float32)
        )
        graph = np.load(atlas / f"{view}_graph.npz")
        edges = np.asarray(graph["edges"], dtype=np.int64)
        edge_keys = np.sort(edges[:, 0] * documents + edges[:, 1])
        meta = np.empty((len(rows), 10), dtype=np.float32)
        for start in range(0, len(rows), 32):
            stop = min(start + 32, len(rows))
            calibration_similarity = query_vectors[view][start:stop] @ file_vectors[calibration].T
            candidate_similarity = np.einsum(
                "bd,bkd->bk", query_vectors[view][start:stop], file_vectors[candidates[start:stop]], optimize=True,
            )
            block_rrf, block_ranks = approximate_rrf(candidate_similarity, calibration_similarity, documents)
            dense_rrf[start:stop, :, view_index], dense_ranks[start:stop, :, view_index] = block_rrf, block_ranks
            for local, query_index in enumerate(range(start, stop)):
                lexical_ids = np.asarray(lexical[query_index][:64], dtype=np.int32)
                lexical_similarity = (
                    file_vectors[lexical_ids] @ query_vectors[view][query_index]
                    if len(lexical_ids) else np.empty(0, dtype=np.float32)
                )
                meta[query_index] = view_meta_features(
                    calibration_similarity[local], calibration, lexical_ids, lexical_similarity,
                    file_to_parent, file_to_leaf, edge_keys, documents,
                )
            if stop % 512 == 0 or stop == len(rows):
                print(f"Router {view} features {stop:,}/{len(rows):,}", flush=True)
        meta_by_view[view] = meta
        del file_vectors, graph, edges, edge_keys

    features = np.empty((len(rows), FEATURE_DIMENSIONS), dtype=np.float32)
    for index, row in enumerate(rows):
        features[index] = assemble_features(
            {view: query_vectors[view][index] for view in VIEWS},
            {view: meta_by_view[view][index] for view in VIEWS}, str(row["query"]),
        )
    np.save(temporary / "features.f32.npy", features)
    channel_rrf = np.zeros((len(rows), candidates.shape[1], len(CHANNELS)), dtype=np.float32)
    channel_ranks = np.full((len(rows), candidates.shape[1], len(CHANNELS)), documents, dtype=np.int32)
    channel_rrf[:, :, :len(VIEWS)], channel_ranks[:, :, :len(VIEWS)] = dense_rrf, dense_ranks
    for query_index, values in enumerate(lexical):
        rank_by_file = {int(value): rank for rank, value in enumerate(values, 1)}
        for position, candidate in enumerate(candidates[query_index]):
            rank = rank_by_file.get(int(candidate))
            if rank is not None:
                channel_ranks[query_index, position, -1] = rank
                channel_rrf[query_index, position, -1] = 1.0 / (RRF_K + rank)
    np.save(temporary / "channel_rrf.f32.npy", channel_rrf)
    np.save(temporary / "channel_ranks.i32.npy", channel_ranks)
    targets = channel_targets(channel_ranks, positive_mask, channel_rrf)
    for name, value in targets.items():
        np.save(temporary / f"{name}.f32.npy", value)
    split = np.asarray([{"train": 0, "validation": 1, "test": 2}[str(row["split"])] for row in rows], dtype=np.int8)
    kind = np.asarray([{"single": 1, "pair": 2, "triple": 3}[str(row["kind"])] for row in rows], dtype=np.int8)
    np.save(temporary / "split.i8.npy", split)
    np.save(temporary / "kind.i8.npy", kind)
    leaf_sizes = np.diff(np.load(_asset(atlas, atlas_manifest, "leaf_offsets"), mmap_mode="r"))
    rare_threshold = float(np.quantile(leaf_sizes, 0.25))
    rare_query = np.asarray([
        any(leaf_sizes[int(file_to_leaf[index])] <= rare_threshold for index in row["positive_file_indices"])
        for row in rows
    ], dtype=bool)
    np.save(temporary / "rare_query.bool.npy", rare_query)
    positive_frequency = Counter(index for row in rows for index in row["positive_file_indices"])
    sample_weight = np.asarray([
        np.mean([1.0 / math.sqrt(positive_frequency[index]) for index in row["positive_file_indices"]])
        for row in rows
    ], dtype=np.float32)
    sample_weight /= np.mean(sample_weight[split == 0])
    np.save(temporary / "sample_weight.f32.npy", sample_weight)
    with gzip.open(temporary / "query_ids.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(stable_json({"index": index, "pseudoquery_id": str(row["query_id"])}) + "\n")
    manifest = {
        "schema": "atlasnav_router_training_arrays_v1", "finalized": True,
        "queries": len(rows), "documents": documents, "features": FEATURE_DIMENSIONS,
        "channels": list(CHANNELS), "candidate_audit": candidate_audit,
        "split_counts": {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))},
        "kind_counts": {name: int(np.sum(kind == code)) for name, code in (("single", 1), ("pair", 2), ("triple", 3))},
        "rare_leaf_definition": "at least one positive in the lowest leaf-size quartile",
        "rare_leaf_size_threshold": rare_threshold,
        "questions_manifest_sha256": sha256_file(questions / "manifest.json"),
        "embedding_manifest_sha256": sha256_file(query_embeddings / "manifest.json"),
        "atlas_manifest_sha256": sha256_file(atlas / "manifest.json"),
        "fulltext_sha256": sha256_file(fulltext_index),
        "construction_reads_answers_qrels_trajectories_or_correctness": False,
        "wall_time_seconds": time.monotonic() - started, "assets": {},
    }
    for path in temporary.iterdir():
        if path.is_file() and path.name != "manifest.json":
            manifest["assets"][path.name] = sha256_file(path)
    atomic_json(temporary / "manifest.json", manifest)
    temporary.replace(output)
    return manifest


def audit_training_arrays(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = _read_json(output / "manifest.json")
    errors = []
    if manifest.get("schema") != "atlasnav_router_training_arrays_v1" or manifest.get("finalized") is not True:
        errors.append("manifest identity mismatch")
    for name, expected in (manifest.get("assets") or {}).items():
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"asset checksum mismatch: {name}")
    if manifest.get("construction_reads_answers_qrels_trajectories_or_correctness") is not False:
        errors.append("evaluation isolation marker failed")
    return {
        "schema": "atlasnav_router_training_arrays_audit_v1", "passed": not errors,
        "errors": errors, "queries": manifest.get("queries"), "candidate_audit": manifest.get("candidate_audit"),
    }
