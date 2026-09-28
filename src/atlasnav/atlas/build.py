"""Build a complete hierarchical multi-view Corpus Atlas."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from atlasnav.atlas.graph import (
    SparseLayer,
    cross_links,
    edge_jaccard,
    hierarchical_partition,
    knn_layer,
    normalize_rows,
    train_pca,
)
from atlasnav.atlas.signatures import VIEWS
from atlasnav.io import atomic_json, sha256_file, stable_json


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(int(row.get("index", -1)) != index for index, row in enumerate(rows)):
        raise RuntimeError("embedding catalog is not canonically ordered")
    return rows


def _save_layer(path: Path, layer: SparseLayer) -> None:
    np.savez(path, name=np.asarray(layer.name), vertices=np.asarray(layer.vertices, dtype=np.int64),
             edges=layer.edges.astype(np.int32), weights=layer.weights.astype(np.float32))


def _centroid(matrix: np.ndarray, members: np.ndarray) -> np.ndarray:
    return normalize_rows(np.mean(matrix[members], axis=0, keepdims=True))[0]


def _representatives(
    members: np.ndarray,
    matrices: dict[str, np.ndarray],
    views: Sequence[str],
    catalog: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    centroids = {view: _centroid(matrices[view], members) for view in views}
    scores = {view: matrices[view][members] @ centroids[view] for view in views}
    combined = np.mean(np.stack([scores[view] for view in views]), axis=0)
    pool = np.lexsort((members, -combined))[: min(len(members), max(80, count * 20))]
    chosen: list[int] = []
    for _ in range(min(count, len(pool))):
        winner: tuple[float, int] | None = None
        for position in pool:
            index = int(members[int(position)])
            if index in chosen:
                continue
            redundancy = max(
                (float(np.mean([matrices[view][index] @ matrices[view][prior] for view in views])) for prior in chosen),
                default=0.0,
            )
            candidate = (float(combined[int(position)]) - 0.18 * redundancy, -index)
            if winner is None or candidate > winner:
                winner = candidate
        if winner is None:
            break
        chosen.append(-winner[1])

    def render(index: int, score: float) -> dict[str, Any]:
        row = catalog[index]
        return {
            "file_index": index,
            "handle": f"D{row['docid']}",
            "title": " ".join(str(row.get("title") or "").split())[:300],
            "domain": " ".join(str(row.get("domain") or "").split())[:120],
            "centrality": score,
        }

    position_by_index = {int(index): position for position, index in enumerate(members)}
    representatives = [render(index, float(combined[position_by_index[index]])) for index in chosen]
    palettes: dict[str, list[dict[str, Any]]] = {}
    for view in views:
        leaders = np.lexsort((members, -scores[view]))[: min(3, len(members))]
        palettes[view] = [render(int(members[int(position)]), float(scores[view][int(position)])) for position in leaders]
    return representatives, palettes


def _ordered_memberships(assignment: np.ndarray, communities: int) -> tuple[np.ndarray, np.ndarray]:
    members = np.argsort(assignment, kind="stable").astype(np.int32)
    counts = np.bincount(assignment, minlength=communities)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return members, offsets


def build_atlas(
    embedding_directory: Path,
    output_directory: Path,
    *,
    pca_dimensions: int = 192,
    pca_training_rows: int = 50_000,
    pca_eigen_power: float = -0.25,
    neighbors: int = 48,
    hnsw_m: int = 32,
    ef_construction: int = 160,
    ef_search: int = 128,
    parent_views: Sequence[str] = ("topic", "identity"),
    child_views: Sequence[str] = ("episode", "relation"),
    parent_layer_weights: Sequence[float] = (1.0, 0.75),
    child_layer_weights: Sequence[float] = (1.0, 1.0),
    parent_count_prior: int = 100,
    parent_resolutions: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
    child_resolution_multipliers: Sequence[float] = (0.35, 0.6, 1.0, 1.7, 2.8),
    representatives: int = 8,
    bridge_links: int = 3,
    seed: int = 42,
    parent_workers: int = 1,
    child_workers: int = 1,
) -> dict[str, Any]:
    embedding_directory, output_directory = embedding_directory.resolve(), output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to replace Atlas: {output_directory}")
    manifest = json.loads((embedding_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "atlasnav_four_view_embeddings_v1" or manifest.get("finalized") is not True:
        raise RuntimeError("finalized four-view embeddings are required")
    if tuple(manifest.get("views") or []) != VIEWS or manifest.get("construction_reads_evaluation_artifacts") is not False:
        raise RuntimeError("embedding leakage firewall or view identity failed")
    if set(parent_views) | set(child_views) != set(VIEWS) or set(parent_views) & set(child_views):
        raise ValueError("parent and child views must be complementary two-view partitions")
    catalog_path = embedding_directory / str(manifest["catalog"])
    if sha256_file(catalog_path) != manifest["catalog_sha256"]:
        raise RuntimeError("embedding catalog checksum mismatch")
    catalog = _load_catalog(catalog_path)
    documents = int(manifest["documents"])
    if len(catalog) != documents:
        raise RuntimeError("embedding catalog cardinality mismatch")
    temporary = output_directory.with_name(f".{output_directory.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    matrices: dict[str, np.ndarray] = {}
    layers: dict[str, SparseLayer] = {}
    pca_assets: dict[str, Any] = {}
    graph_assets: dict[str, Any] = {}
    try:
        for view_index, view in enumerate(VIEWS):
            source = embedding_directory / str(manifest["view_embeddings"][view])
            if sha256_file(source) != manifest["view_embedding_sha256"][view]:
                raise RuntimeError(f"{view} embedding checksum mismatch")
            raw = np.load(source, mmap_mode="r")
            reduced, parameters = train_pca(
                raw,
                output_dimensions=min(pca_dimensions, raw.shape[1]),
                training_rows=pca_training_rows,
                seed=seed + view_index * 101,
                eigen_power=pca_eigen_power,
            )
            matrices[view] = reduced
            reduced_path = temporary / f"{view}_reduced.f16.npy"
            np.save(reduced_path, reduced.astype(np.float16))
            transform_path = temporary / f"{view}_pca.npz"
            np.savez(transform_path, **{key: np.asarray(value) for key, value in parameters.items()})
            layer = knn_layer(view, reduced, neighbors=neighbors, hnsw_m=hnsw_m,
                              ef_construction=ef_construction, ef_search=ef_search, single_thread=True)
            if documents > 1 and np.any(np.bincount(layer.edges.ravel(), minlength=documents) == 0):
                raise RuntimeError(f"{view} graph contains isolated documents")
            layers[view] = layer
            graph_path = temporary / f"{view}_graph.npz"
            _save_layer(graph_path, layer)
            pca_assets[view] = {
                "reduced": reduced_path.name, "reduced_sha256": sha256_file(reduced_path),
                "transform": transform_path.name, "transform_sha256": sha256_file(transform_path),
            }
            graph_assets[view] = {
                "path": graph_path.name, "sha256": sha256_file(graph_path), "undirected_edges": len(layer.edges),
            }
            print(f"{view}: {len(layer.edges):,} undirected edges", flush=True)
        overlaps = {
            f"{left}:{right}": edge_jaccard(layers[left], layers[right])
            for index, left in enumerate(VIEWS) for right in VIEWS[index + 1:]
        }
        if max(overlaps.values(), default=0.0) >= 0.90:
            raise RuntimeError(f"multi-view topology collapsed: {overlaps}")
        parent, leaf, partition_audit = hierarchical_partition(
            [layers[view] for view in parent_views],
            [layers[view] for view in child_views],
            parent_layer_weights=parent_layer_weights,
            child_layer_weights=child_layer_weights,
            parent_target=parent_count_prior,
            parent_resolutions=parent_resolutions,
            child_resolution_multipliers=child_resolution_multipliers,
            seeds=(seed, seed + 1),
            child_workers=child_workers,
            parent_candidate_workers=parent_workers,
        )
        parent_count = int(parent.max(initial=-1)) + 1
        leaf_count = int(leaf.max(initial=-1)) + 1
        parent_members, parent_offsets = _ordered_memberships(parent, parent_count)
        leaf_members, leaf_offsets = _ordered_memberships(leaf, leaf_count)
        leaf_to_parent = np.empty(leaf_count, dtype=np.int32)
        for leaf_id in range(leaf_count):
            values = np.unique(parent[leaf == leaf_id])
            if len(values) != 1:
                raise RuntimeError("leaf crosses a parent boundary")
            leaf_to_parent[leaf_id] = values[0]
        arrays = {
            "file_to_parent": parent,
            "file_to_leaf": leaf,
            "parent_members": parent_members,
            "parent_offsets": parent_offsets,
            "leaf_members": leaf_members,
            "leaf_offsets": leaf_offsets,
            "leaf_to_parent": leaf_to_parent,
        }
        array_assets: dict[str, str] = {}
        for name, value in arrays.items():
            path = temporary / f"{name}.{value.dtype.name}.npy"
            np.save(path, value)
            array_assets[name] = path.name
        links = cross_links([layers[view] for view in child_views], leaf, maximum_per_leaf=bridge_links)
        parent_cards: list[dict[str, Any]] = []
        for parent_id in range(parent_count):
            members = np.flatnonzero(parent == parent_id).astype(np.int32)
            reps, palettes = _representatives(members, matrices, parent_views, catalog, representatives)
            parent_cards.append({"parent_id": f"P{parent_id:03d}", "size": len(members),
                                 "representatives": reps, "view_palettes": palettes})
        leaf_cards: list[dict[str, Any]] = []
        for leaf_id in range(leaf_count):
            members = np.flatnonzero(leaf == leaf_id).astype(np.int32)
            reps, palettes = _representatives(members, matrices, child_views, catalog, representatives)
            leaf_cards.append({"leaf_id": f"L{leaf_id:04d}", "parent_id": f"P{int(leaf_to_parent[leaf_id]):03d}",
                               "size": len(members), "representatives": reps, "view_palettes": palettes,
                               "bridges": links.get(leaf_id, [])})
        for name, rows in (("parent_cards.jsonl.gz", parent_cards), ("leaf_cards.jsonl.gz", leaf_cards)):
            with gzip.open(temporary / name, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(stable_json(row) + "\n")
        shutil.copy2(catalog_path, temporary / "catalog.jsonl.gz")
        result = {
            "schema": "atlasnav_multiplex_atlas_v1",
            "finalized": True,
            "documents": documents,
            "parent_regions": parent_count,
            "leaf_regions": leaf_count,
            "parent_views": list(parent_views),
            "child_views": list(child_views),
            "candidate_boundary": False,
            "construction_reads_evaluation_artifacts": False,
            "complete_primary_addressing": bool(len(parent) == documents and len(leaf) == documents),
            "pca": {"dimensions": pca_dimensions, "training_rows": pca_training_rows,
                    "eigen_power": pca_eigen_power, "assets": pca_assets},
            "graphs": {"neighbors": neighbors, "hnsw_m": hnsw_m,
                       "ef_construction": ef_construction, "ef_search": ef_search,
                       "layers": graph_assets, "exact_edge_jaccard": overlaps},
            "hierarchy": {"parent_count_prior": parent_count_prior, "audit": partition_audit},
            "arrays": array_assets,
            "cards": {"parents": "parent_cards.jsonl.gz", "leaves": "leaf_cards.jsonl.gz"},
            "catalog": "catalog.jsonl.gz",
            "wall_time_seconds": time.monotonic() - started,
        }
        atomic_json(temporary / "manifest.json", result)
        temporary.replace(output_directory)
        return result
    except BaseException:
        # The temporary tree is intentionally retained for forensic recovery.
        raise

