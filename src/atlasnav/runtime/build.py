"""Build query workspaces from a frozen Atlas and query-adaptive Router."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from atlasnav.atlas.graph import normalize_rows
from atlasnav.atlas.signatures import VIEWS
from atlasnav.embeddings.queries import load_queries
from atlasnav.io import atomic_json, sha256_file, stable_json
from atlasnav.router.features import assemble_features, calibration_indexes, view_meta_features
from atlasnav.router.model import outputs_from_model


RRF_K = 60
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _link_or_copy(source: Path, destination: Path) -> str:
    """Avoid duplicating a large immutable index when the filesystem permits."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _asset(atlas: Path, manifest: dict[str, Any], name: str) -> Path:
    mapping = manifest.get("arrays") or manifest.get("assets") or {}
    value = mapping.get(name)
    if value is None:
        aliases = {
            "file_to_parent": "file_to_parent.i32.npy",
            "file_to_leaf": "file_to_leaf.i32.npy",
            "leaf_to_parent": "leaf_to_parent.i32.npy",
            "leaf_local_index": "leaf_local_index.i32.npy",
        }
        value = aliases.get(name)
    if not value or not (atlas / str(value)).is_file():
        raise FileNotFoundError(f"Atlas asset is missing: {name}")
    return atlas / str(value)


def _card_assets(atlas: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    cards = manifest.get("cards") or {}
    parent = cards.get("parents") or (manifest.get("assets") or {}).get("parent_cards") or "parent_cards.jsonl.gz"
    leaf = cards.get("leaves") or (manifest.get("assets") or {}).get("child_cards") or "child_cards.jsonl.gz"
    return atlas / str(parent), atlas / str(leaf)


def _labels(atlas: Path, parents: list[dict[str, Any]], leaves: list[dict[str, Any]]) -> None:
    path = atlas / "labels.deepseek.json"
    if path.is_file():
        value = _read_json(path)
        parent_labels = value.get("parents") or {}
        leaf_labels = value.get("children") or {}
        for index, card in enumerate(parents):
            identity = str(card.get("cluster_id") or f"C{index:03d}")
            label = parent_labels.get(identity) or {}
            if label.get("label"):
                card["llm_label"] = str(label["label"])
                card["llm_summary"] = str(label.get("summary") or "")
        for index, card in enumerate(leaves):
            parent = int(card.get("parent_index", 0))
            local = int(card.get("local_index", index))
            identity = str(card.get("child_id") or f"C{parent:03d}/S{local:02d}")
            label = leaf_labels.get(identity) or {}
            if label.get("label"):
                card["llm_label"] = str(label["label"])
                card["facet_hint"] = str(label.get("facet_hint") or "")


def _label(card: dict[str, Any], limit: int = 180) -> str:
    value = card.get("llm_label") or card.get("machine_label")
    if not value:
        representatives = card.get("representatives") or []
        value = " | ".join(str(row.get("title") or "") for row in representatives[:3])
    return " ".join(str(value or "unlabeled region").split())[:limit]


def _project(raw: np.ndarray, transform_path: Path) -> np.ndarray:
    parameters = np.load(transform_path)
    transform = np.asarray(parameters["transform"], dtype=np.float32)
    bias = np.asarray(parameters["bias"], dtype=np.float32)
    values = np.asarray(raw, dtype=np.float32) @ transform.T
    if bias.size:
        values += bias
    return normalize_rows(values)


def _expression(query: str) -> str:
    values: list[str] = []
    for token in TOKEN_RE.findall(query):
        normalized = token.casefold()
        if len(normalized) < 2 or normalized in values:
            continue
        values.append(normalized)
        if len(values) >= 10:
            break
    if not values:
        raise ValueError("query contains no full-text terms")
    return " OR ".join('"' + value.replace('"', '""') + '"' for value in values)


def _lexical(connection: sqlite3.Connection, query: str) -> tuple[list[sqlite3.Row], str]:
    expression = _expression(query)
    rows = connection.execute(
        "SELECT documents_fts.rowid rowid,bm25(documents_fts,1.0) score "
        "FROM documents_fts WHERE documents_fts MATCH ? ORDER BY score,rowid",
        (expression,),
    ).fetchall()
    return rows, expression


def _preview(connection: sqlite3.Connection, index: int, expression: str) -> str:
    row = connection.execute(
        "SELECT snippet(documents_fts,0,'[',']',' … ',26) FROM documents_fts "
        "WHERE rowid=? AND documents_fts MATCH ?",
        (index + 1, expression),
    ).fetchone()
    if row is None or not str(row[0] or "").strip():
        row = connection.execute("SELECT substr(text,1,500) FROM documents WHERE rowid=?", (index + 1,)).fetchone()
    return " ".join(str(row[0] if row else "").split())[:260]


def _leaf_local(leaf_to_parent: np.ndarray) -> np.ndarray:
    result = np.empty(len(leaf_to_parent), dtype=np.int32)
    seen: dict[int, int] = {}
    for leaf, parent_value in enumerate(leaf_to_parent):
        parent = int(parent_value)
        result[leaf] = seen.get(parent, 0)
        seen[parent] = int(result[leaf]) + 1
    return result


def _route(
    scores: np.ndarray,
    parent: np.ndarray,
    leaf: np.ndarray,
    leaf_to_parent: np.ndarray,
    leaf_local: np.ndarray,
    parents: list[dict[str, Any]],
    leaves: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    connection: sqlite3.Connection,
    expression: str,
    initial_parents: int,
    anchors_per_parent: int,
    members_by_leaf: tuple[np.ndarray, ...],
    leaves_by_parent: tuple[tuple[int, ...], ...],
) -> tuple[list[int], list[dict[str, Any]], str]:
    leaf_best: dict[int, int] = {}
    for leaf_id in range(len(leaves)):
        members = members_by_leaf[leaf_id]
        order = np.lexsort((members, -scores[members]))
        leaf_best[leaf_id] = int(members[order[0]])
    parent_order = sorted(
        range(len(parents)),
        key=lambda parent_id: (
            -max(float(scores[leaf_best[leaf_id]]) for leaf_id in leaves_by_parent[parent_id]),
            parent_id,
        ),
    )
    leads: list[dict[str, Any]] = []
    lines = ["rank\tparent\tcorpus_files\tregion_label\tfacet_palette\tanchors"]
    for route_rank, parent_id in enumerate(parent_order[:initial_parents], 1):
        ranked_leaves = sorted(
            leaves_by_parent[parent_id],
            key=lambda leaf_id: (-float(scores[leaf_best[leaf_id]]), leaf_id),
        )
        anchors: list[str] = []
        for anchor_rank, leaf_id in enumerate(ranked_leaves[:anchors_per_parent], 1):
            file_index = leaf_best[leaf_id]
            row = catalog[file_index]
            address = f"C{parent_id:03d}/S{int(leaf_local[leaf_id]):02d}"
            preview = _preview(connection, file_index, expression)
            anchors.append(
                f"{address} {_label(leaves[leaf_id], 90)} :: D{row['docid']} "
                f"{' '.join(str(row.get('title') or '').split())[:140]} :: {preview}"
            )
            leads.append({
                "handle": f"D{row['docid']}", "title": row.get("title"),
                "map_address": address,
                "rank": (route_rank - 1) * anchors_per_parent + anchor_rank,
                "preview": preview,
            })
        members = int(np.sum(parent == parent_id))
        palette = " / ".join(_label(leaves[leaf_id], 70) for leaf_id in ranked_leaves[:5])
        lines.append(
            f"{route_rank}\tC{parent_id:03d}\t{members}\t{_label(parents[parent_id])}\t"
            f"{palette}\t{' ; '.join(anchors)}"
        )
    lines.append(f"# First viewport over a complete {len(parent):,}-file Atlas; no candidate boundary.")
    return parent_order, leads, "\n".join(lines) + "\n"


def build_runtime(
    *,
    corpus_directory: Path,
    fulltext_index: Path,
    atlas_directory: Path,
    query_bundle: Path | None,
    router_model: Path | None,
    frozen_ranking: Path | None = None,
    dataset: Path,
    output_directory: Path,
    state_directory: Path,
    initial_parents: int = 10,
    anchors_per_parent: int = 3,
) -> dict[str, Any]:
    paths = [corpus_directory, fulltext_index, atlas_directory, dataset]
    corpus_directory, fulltext_index, atlas_directory, dataset = [
        Path(path).resolve() for path in paths
    ]
    query_bundle = Path(query_bundle).resolve() if query_bundle is not None else None
    router_model = Path(router_model).resolve() if router_model is not None else None
    frozen_ranking = Path(frozen_ranking).resolve() if frozen_ranking is not None else None
    if (query_bundle is None) == (frozen_ranking is None):
        raise ValueError("select exactly one of query embeddings or a frozen ranking")
    if query_bundle is not None and router_model is None:
        raise ValueError("a Router model is required with query embeddings")
    if frozen_ranking is not None and router_model is not None:
        raise ValueError("a frozen ranking already contains the Router output")
    output_directory, state_directory = output_directory.resolve(), state_directory.resolve()
    if output_directory.exists() or state_directory.exists():
        raise FileExistsError("refusing to replace a runtime or its persistent state")
    corpus_manifest = _read_json(corpus_directory / "manifest.json")
    atlas_manifest = _read_json(atlas_directory / "manifest.json")
    query_manifest = _read_json(query_bundle / "manifest.json") if query_bundle else None
    ranking_manifest = _read_json(frozen_ranking / "manifest.json") if frozen_ranking else None
    if corpus_manifest.get("schema") != "atlasnav_canonical_corpus_v1":
        raise RuntimeError("canonical corpus is required")
    if atlas_manifest.get("schema") not in {"atlasnav_multiplex_atlas_v1", "atlasnav_multiplex_atlas_frozen_v1"}:
        raise RuntimeError("a finalized AtlasNav Atlas is required")
    if query_manifest is not None and query_manifest.get("schema") != "atlasnav_four_view_query_embeddings_v1":
        raise RuntimeError("four-view query embeddings are required")
    if ranking_manifest is not None and ranking_manifest.get("schema") != "atlasnav_frozen_ranking_v1":
        raise RuntimeError("a finalized AtlasNav frozen ranking is required")
    if ranking_manifest is not None and ranking_manifest.get("finalized") is not True:
        raise RuntimeError("the AtlasNav frozen ranking is not finalized")
    queries = load_queries(dataset)
    catalog = _read_gzip(atlas_directory / str(atlas_manifest["catalog"]))
    documents = int(atlas_manifest["documents"])
    if len(catalog) != documents or int(corpus_manifest["documents"]) != documents:
        raise RuntimeError("corpus and Atlas cardinality differ")
    corpus_catalog = _read_gzip(corpus_directory / str(corpus_manifest["catalog_file"]))
    if [str(row["docid"]) for row in corpus_catalog] != [str(row["docid"]) for row in catalog]:
        raise RuntimeError("canonical corpus order differs from the frozen Atlas")
    bundle = query_bundle if query_bundle is not None else frozen_ranking
    bundle_manifest = query_manifest if query_manifest is not None else ranking_manifest
    assert bundle is not None and bundle_manifest is not None
    query_catalog_path = bundle / str(bundle_manifest["query_catalog"])
    if (
        ranking_manifest is not None
        and sha256_file(query_catalog_path) != ranking_manifest.get("query_catalog_sha256")
    ):
        raise RuntimeError("frozen query catalog checksum mismatch")
    query_catalog = _read_gzip(query_catalog_path)
    if [row["query_id"] for row in query_catalog] != [row["query_id"] for row in queries]:
        raise RuntimeError("query dataset differs from its ranking bundle")
    parent_path = _asset(atlas_directory, atlas_manifest, "file_to_parent")
    leaf_path = _asset(atlas_directory, atlas_manifest, "file_to_leaf")
    leaf_parent_path = _asset(atlas_directory, atlas_manifest, "leaf_to_parent")
    parent = np.load(parent_path, mmap_mode="r")
    leaf = np.load(leaf_path, mmap_mode="r")
    leaf_to_parent = np.load(leaf_parent_path, mmap_mode="r")
    try:
        leaf_local_path = _asset(atlas_directory, atlas_manifest, "leaf_local_index")
        leaf_local = np.load(leaf_local_path, mmap_mode="r")
    except FileNotFoundError:
        leaf_local = _leaf_local(leaf_to_parent)
    parent_cards_path, leaf_cards_path = _card_assets(atlas_directory, atlas_manifest)
    parent_cards, leaf_cards = _read_gzip(parent_cards_path), _read_gzip(leaf_cards_path)
    _labels(atlas_directory, parent_cards, leaf_cards)
    ordered_files = np.argsort(leaf, kind="stable")
    leaf_counts = np.bincount(leaf, minlength=len(leaf_cards))
    leaf_offsets = np.concatenate(([0], np.cumsum(leaf_counts)))
    members_by_leaf = tuple(
        ordered_files[leaf_offsets[index]:leaf_offsets[index + 1]]
        for index in range(len(leaf_cards))
    )
    leaves_by_parent = tuple(
        tuple(index for index, value in enumerate(leaf_to_parent) if int(value) == parent_id)
        for parent_id in range(len(parent_cards))
    )
    temporary = output_directory.with_name(f".{output_directory.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    score_path = temporary / "query_file_scores.f32.npy"
    retrieval_audit: list[dict[str, Any]] = []
    frozen_routes: dict[str, dict[str, Any]] | None = None
    connection = sqlite3.connect(f"file:{fulltext_index}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if frozen_ranking is not None:
            assert ranking_manifest is not None
            source_scores = frozen_ranking / str(ranking_manifest["query_scores"])
            if sha256_file(source_scores) != ranking_manifest.get("query_scores_sha256"):
                raise RuntimeError("frozen ranking checksum mismatch")
            if int(ranking_manifest.get("queries") or 0) != len(queries):
                raise RuntimeError("frozen ranking query cardinality mismatch")
            if int(ranking_manifest.get("documents") or 0) != documents:
                raise RuntimeError("frozen ranking document cardinality mismatch")
            _link_or_copy(source_scores, score_path)
            scores = np.load(score_path, mmap_mode="r")
            if scores.shape != (len(queries), documents) or not np.isfinite(scores).all():
                raise RuntimeError("frozen ranking shape or values are invalid")
            audit_path = frozen_ranking / str(ranking_manifest["retrieval_audit"])
            if sha256_file(audit_path) != ranking_manifest.get("retrieval_audit_sha256"):
                raise RuntimeError("frozen retrieval audit checksum mismatch")
            retrieval_audit = _read_gzip(audit_path)
            if [row["query_id"] for row in retrieval_audit] != [row["query_id"] for row in queries]:
                raise RuntimeError("frozen retrieval audit differs from the query dataset")
            route_asset = ranking_manifest.get("workspace_routes")
            if route_asset:
                route_path = frozen_ranking / str(route_asset)
                if sha256_file(route_path) != ranking_manifest.get("workspace_routes_sha256"):
                    raise RuntimeError("frozen workspace route checksum mismatch")
                route_rows = _read_gzip(route_path)
                if [str(row["query_id"]) for row in route_rows] != [row["query_id"] for row in queries]:
                    raise RuntimeError("frozen workspace routes differ from the query dataset")
                frozen_routes = {str(row["query_id"]): row for row in route_rows}
        else:
            assert query_bundle is not None and query_manifest is not None and router_model is not None
            file_vectors: dict[str, np.ndarray] = {}
            query_vectors: dict[str, np.ndarray] = {}
            edge_keys: dict[str, np.ndarray] = {}
            for view in VIEWS:
                pca = atlas_manifest["pca"]["assets"][view]
                file_vectors[view] = normalize_rows(
                    np.asarray(np.load(atlas_directory / pca["reduced"], mmap_mode="r"), dtype=np.float32)
                )
                raw = np.load(query_bundle / query_manifest["query_embeddings"][view], mmap_mode="r")
                query_vectors[view] = _project(raw, atlas_directory / pca["transform"])
                graph = np.load(atlas_directory / f"{view}_graph.npz")
                edges = np.asarray(graph["edges"], dtype=np.int64)
                edge_keys[view] = np.sort(edges[:, 0] * documents + edges[:, 1])
            model = np.load(router_model)
            calibration = calibration_indexes(documents)
            scores = open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(queries), documents))
            for query_index, query in enumerate(queries):
                similarities = {
                    view: file_vectors[view] @ query_vectors[view][query_index] for view in VIEWS
                }
                lexical, _ = _lexical(connection, query["query"])
                lexical_ids = np.asarray([int(row["rowid"]) - 1 for row in lexical[:64]], dtype=np.int32)
                meta = {
                    view: view_meta_features(
                        similarities[view][calibration], calibration, lexical_ids,
                        similarities[view][lexical_ids] if len(lexical_ids) else np.empty(0, dtype=np.float32),
                        parent, leaf, edge_keys[view], documents,
                    )
                    for view in VIEWS
                }
                feature = assemble_features(
                    {view: query_vectors[view][query_index] for view in VIEWS}, meta, query["query"]
                )
                output = outputs_from_model(feature[None, :], model)
                channel = output["channel"][0]
                fused = np.zeros(documents, dtype=np.float32)
                ids = np.arange(documents, dtype=np.int32)
                for view_index, view in enumerate(VIEWS):
                    order = np.lexsort((ids, -similarities[view]))
                    ranks = np.empty(documents, dtype=np.int32)
                    ranks[order] = np.arange(1, documents + 1, dtype=np.int32)
                    fused += float(channel[view_index]) / (RRF_K + ranks)
                for rank, row in enumerate(lexical, 1):
                    fused[int(row["rowid"]) - 1] += float(channel[-1]) / (RRF_K + rank)
                scores[query_index] = fused
                retrieval_audit.append({
                    "query_id": query["query_id"],
                    "channel_weights": {name: float(channel[index]) for index, name in enumerate((*VIEWS, "bm25"))},
                    "facet_weights": {name: float(output["facet"][0][index]) for index, name in enumerate(VIEWS)},
                    "semantic_mass": float(2.0 * output["eta"][0]),
                    "bm25_mass": float(2.0 * (1.0 - output["eta"][0])),
                    "router_confidence": float(output["confidence"][0]),
                    "scored_documents_per_semantic_view": documents,
                    "topk_before_atlas_aggregation": None,
                    "lexical_matching_documents": len(lexical),
                    "reads_answers_qrels_judgments_or_trajectories": False,
                })
                print(f"ranked {query_index + 1}/{len(queries)}", flush=True)
            scores.flush()
        for source, name in (
            (parent_path, "file_to_parent.i32.npy"),
            (leaf_path, "file_to_leaf.i32.npy"),
            (leaf_parent_path, "leaf_to_parent.i32.npy"),
        ):
            shutil.copy2(source, temporary / name)
        np.save(temporary / "leaf_local_index.i32.npy", np.asarray(leaf_local, dtype=np.int32))
        for name, rows in (("parent_cards.jsonl.gz", parent_cards), ("child_cards.jsonl.gz", leaf_cards), ("catalog.jsonl.gz", catalog)):
            with gzip.open(temporary / name, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(stable_json(row) + "\n")
        fulltext_materialization = _link_or_copy(fulltext_index, temporary / "fulltext.sqlite3")
        wrapper = temporary / "atlas.py"
        wrapper.write_text(
            "#!/usr/bin/env python3\nfrom atlasnav.runtime.atlas_tool import main\nmain()\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        records: dict[str, Any] = {}
        for query_index, query in enumerate(queries):
            frozen_route = (frozen_routes or {}).get(query["query_id"])
            if frozen_route is not None:
                route = str(frozen_route["route_tsv"])
                if hashlib.sha256(route.encode()).hexdigest() != frozen_route.get("route_sha256"):
                    raise RuntimeError(f"frozen route checksum mismatch: {query['query_id']}")
                parent_order = [int(value) for value in frozen_route["parent_order"]]
                leads = list(frozen_route["initial_leads"])
            else:
                _, expression = _lexical(connection, query["query"])
                parent_order, leads, route = _route(
                    np.asarray(scores[query_index]), parent, leaf, leaf_to_parent, leaf_local,
                    parent_cards, leaf_cards, catalog, connection, expression,
                    initial_parents, anchors_per_parent, members_by_leaf, leaves_by_parent,
                )
            workspace = temporary / "workspaces" / query["query_id"]
            workspace.mkdir(parents=True)
            (workspace / "QUERY.txt").write_text(query["query"].rstrip() + "\n", encoding="utf-8")
            (workspace / "ROUTE.tsv").write_text(route, encoding="utf-8")
            final_workspace = output_directory / "workspaces" / query["query_id"]
            atomic_json(workspace / ".atlas_query.json", {
                "schema": "atlasnav_query_workspace_v1", "query_id": query["query_id"],
                "query_index": query_index, "parent_order": parent_order,
                "initial_leads": leads, "scored_documents": documents,
                "topk_before_atlas_aggregation": None,
                "reads_answers_qrels_judgments_or_trajectories": False,
            })
            atomic_json(workspace / "WORKSPACE.json", {
                "schema": "atlasnav_workspace_v1", "query_id": query["query_id"],
                "arm": "atlasnav", "workspace": str(final_workspace),
                "atlas_runtime": str(output_directory),
                "state_dir": str(state_directory / query["query_id"]),
            })
            os.symlink("../../atlas.py", workspace / "atlas.py")
            records[query["query_id"]] = {
                "route_bytes": len(route.encode()),
                "route_sha256": hashlib.sha256(route.encode()).hexdigest(),
                "initial_leads": len(leads),
            }
    finally:
        connection.close()
    retrieval_path = temporary / "retrieval_audit.jsonl.gz"
    with gzip.open(retrieval_path, "wt", encoding="utf-8") as stream:
        for row in retrieval_audit:
            stream.write(stable_json(row) + "\n")
    manifest = {
        "schema": "atlasnav_runtime_v1", "finalized": True,
        "queries": len(queries), "documents": documents,
        "parents": len(parent_cards), "leaves": len(leaf_cards),
        "initial_parents": initial_parents, "anchors_per_parent": anchors_per_parent,
        "candidate_boundary": False, "full_corpus_reachable": True,
        "complete_dense_rank_per_semantic_view": True,
        "rrf_k": RRF_K,
        "fulltext_materialization": fulltext_materialization,
        "query_scores_sha256": sha256_file(score_path),
        "retrieval_audit_sha256": sha256_file(retrieval_path),
        "router_model_sha256": (
            sha256_file(router_model) if router_model is not None
            else str((ranking_manifest or {}).get("router_model_sha256") or "")
        ),
        "atlas_manifest_sha256": sha256_file(atlas_directory / "manifest.json"),
        "query_manifest_sha256": (
            sha256_file(query_bundle / "manifest.json") if query_bundle is not None else None
        ),
        "frozen_ranking_manifest_sha256": (
            sha256_file(frozen_ranking / "manifest.json") if frozen_ranking is not None else None
        ),
        "workspace_records": records,
        "reads_answers_qrels_judgments_or_trajectories": False,
    }
    atomic_json(temporary / "manifest.json", manifest)
    temporary.replace(output_directory)
    state_directory.mkdir(parents=True)
    atomic_json(state_directory / "manifest.json", {
        "schema": "atlasnav_runtime_state_v1",
        "runtime_manifest_sha256": sha256_file(output_directory / "manifest.json"),
        "queries": len(queries),
    })
    return {**manifest, "workspace_records": len(records)}


def audit_runtime(output_directory: Path) -> dict[str, Any]:
    output_directory = output_directory.resolve()
    manifest = _read_json(output_directory / "manifest.json")
    errors: list[str] = []
    if manifest.get("schema") != "atlasnav_runtime_v1" or manifest.get("finalized") is not True:
        errors.append("runtime manifest is not finalized")
    queries = int(manifest.get("queries", -1))
    documents = int(manifest.get("documents", -1))
    scores = np.load(output_directory / "query_file_scores.f32.npy", mmap_mode="r")
    if scores.shape != (queries, documents) or not np.isfinite(scores).all():
        errors.append("query score matrix shape or values are invalid")
    if sha256_file(output_directory / "query_file_scores.f32.npy") != manifest.get("query_scores_sha256"):
        errors.append("query score checksum mismatch")
    workspaces = output_directory / "workspaces"
    values = [path for path in workspaces.iterdir() if path.is_dir()] if workspaces.is_dir() else []
    if len(values) != queries:
        errors.append("workspace cardinality mismatch")
    for workspace in values:
        required = ("QUERY.txt", "ROUTE.tsv", "WORKSPACE.json", ".atlas_query.json", "atlas.py")
        if any(not (workspace / name).exists() for name in required):
            errors.append(f"incomplete workspace: {workspace.name}")
        if (workspace / "CANDIDATES.tsv").exists():
            errors.append(f"candidate boundary leaked into workspace: {workspace.name}")
    return {
        "schema": "atlasnav_runtime_audit_v1",
        "passed": not errors,
        "errors": errors,
        "queries": len(values),
        "documents": documents,
        "full_corpus_reachable": manifest.get("full_corpus_reachable") is True,
    }
