#!/usr/bin/env python3
"""Reusable graph and multiplex-community primitives for the AtlasNav atlas."""

from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import faiss
import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class SparseLayer:
    name: str
    vertices: int
    edges: np.ndarray
    weights: np.ndarray

    def graph(self) -> ig.Graph:
        graph = ig.Graph(n=self.vertices, edges=self.edges.tolist(), directed=False)
        graph.es["weight"] = self.weights.astype(float).tolist()
        return graph


@dataclass(frozen=True)
class PartitionCandidate:
    resolution: float
    seed: int
    membership: np.ndarray
    improvement: float
    communities: int
    maximum_size: int
    singleton_fraction: float


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("AtlasNav graph input contains zero or non-finite vectors")
    return values / norms


def train_pca(
    matrix: np.ndarray,
    *,
    output_dimensions: int,
    training_rows: int,
    seed: int,
    eigen_power: float = -0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    values = np.asarray(matrix, dtype=np.float32)
    if output_dimensions >= values.shape[1]:
        reduced = normalize_rows(values)
        return reduced, {
            "input_dimensions": values.shape[1],
            "output_dimensions": values.shape[1],
            "mean": np.zeros(values.shape[1], dtype=np.float32),
            "transform": np.eye(values.shape[1], dtype=np.float32),
            "bias": np.zeros(values.shape[1], dtype=np.float32),
            "eigenvalues": np.ones(values.shape[1], dtype=np.float32),
            "eigen_power": 0.0,
        }
    rng = np.random.default_rng(seed)
    sample_size = min(training_rows, len(values))
    sample_indices = np.sort(rng.choice(len(values), size=sample_size, replace=False))
    transform = faiss.PCAMatrix(values.shape[1], output_dimensions, eigen_power, False)
    transform.train(np.ascontiguousarray(values[sample_indices]))
    if not transform.is_trained:
        raise RuntimeError("FAISS PCA did not train")
    reduced = normalize_rows(transform.apply_py(np.ascontiguousarray(values)))
    matrix_a = faiss.vector_to_array(transform.A).reshape(output_dimensions, values.shape[1])
    # FAISS serialisation is handled by the build pipeline. A, b and
    # eigenvalues are exposed here only for deterministic diagnostics.
    return reduced, {
        "input_dimensions": values.shape[1],
        "output_dimensions": output_dimensions,
        "transform": matrix_a.astype(np.float32),
        "bias": faiss.vector_to_array(transform.b).astype(np.float32),
        "eigenvalues": faiss.vector_to_array(transform.eigenvalues).astype(np.float32),
        "eigen_power": eigen_power,
    }


def knn_layer(
    name: str,
    vectors: np.ndarray,
    *,
    neighbors: int = 48,
    hnsw_m: int = 32,
    ef_construction: int = 160,
    ef_search: int = 128,
    single_thread: bool = True,
) -> SparseLayer:
    values = normalize_rows(vectors)
    count, dimensions = values.shape
    if count < 2:
        return SparseLayer(name, count, np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32))
    k = min(neighbors, count - 1)
    if single_thread:
        faiss.omp_set_num_threads(1)
    if count <= 4_096:
        index: faiss.Index = faiss.IndexFlatIP(dimensions)
    else:
        index = faiss.IndexHNSWFlat(dimensions, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = max(ef_search, k + 1)
    index.add(np.ascontiguousarray(values))
    similarities, neighbors_found = index.search(np.ascontiguousarray(values), k + 1)

    directed: list[list[tuple[int, float]]] = []
    local_scales = np.empty(count, dtype=np.float32)
    for source in range(count):
        rows: list[tuple[int, float]] = []
        for target, similarity in zip(neighbors_found[source], similarities[source]):
            target = int(target)
            if target < 0 or target == source:
                continue
            rows.append((target, float(similarity)))
            if len(rows) >= k:
                break
        directed.append(rows)
        local_scales[source] = max(1e-4, 1.0 - rows[-1][1]) if rows else 1.0

    accumulated: dict[tuple[int, int], list[float]] = {}
    for source, rows in enumerate(directed):
        for rank_index, (target, similarity) in enumerate(rows, start=1):
            left, right = sorted((source, target))
            distance = max(0.0, 1.0 - similarity)
            scale = math.sqrt(float(local_scales[source]) * float(local_scales[target]))
            local_affinity = math.exp(-distance / max(scale, 1e-6))
            rank_discount = 1.0 / math.log2(rank_index + 2.0)
            accumulated.setdefault((left, right), []).append(local_affinity * rank_discount)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for edge, observations in sorted(accumulated.items()):
        # Mutual edges retain their mean strength. A one-way edge is kept at
        # half strength so sparse rare modes do not become unreachable.
        weight = statistics_mean(observations) * (1.0 if len(observations) >= 2 else 0.5)
        if weight > 0:
            edges.append(edge)
            weights.append(weight)
    return SparseLayer(
        name=name,
        vertices=count,
        edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        weights=np.asarray(weights, dtype=np.float32),
    )


def statistics_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def edge_jaccard(left: SparseLayer, right: SparseLayer) -> float:
    """Exact topology overlap without constructing Python sets of millions of edges."""
    if left.vertices != right.vertices:
        raise ValueError("AtlasNav graph layers have different vertex sets")
    vertices = max(left.vertices, 1)
    left_ids = (
        left.edges[:, 0].astype(np.int64) * vertices + left.edges[:, 1].astype(np.int64)
    )
    right_ids = (
        right.edges[:, 0].astype(np.int64) * vertices + right.edges[:, 1].astype(np.int64)
    )
    if len(np.unique(left_ids)) != len(left_ids) or len(np.unique(right_ids)) != len(right_ids):
        raise RuntimeError("AtlasNav sparse layer contains duplicate undirected edges")
    intersection = len(np.intersect1d(left_ids, right_ids, assume_unique=True))
    union = len(left_ids) + len(right_ids) - intersection
    return float(intersection / union) if union else 1.0


def canonicalize_membership(membership: Sequence[int]) -> np.ndarray:
    values = np.asarray(membership, dtype=np.int32)
    groups: dict[int, list[int]] = {}
    for index, community in enumerate(values):
        groups.setdefault(int(community), []).append(index)
    order = sorted(groups, key=lambda community: (-len(groups[community]), groups[community][0]))
    mapping = {community: new for new, community in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int32)


def partition_once(
    layers: Sequence[SparseLayer],
    *,
    layer_weights: Sequence[float],
    resolution: float,
    seed: int,
    n_iterations: int = 4,
    graphs: Sequence[ig.Graph] | None = None,
) -> PartitionCandidate:
    if not layers or len(layers) != len(layer_weights):
        raise ValueError("AtlasNav multiplex layers and weights must be non-empty and aligned")
    vertices = {layer.vertices for layer in layers}
    if len(vertices) != 1:
        raise ValueError("AtlasNav multiplex layers have different vertex sets")
    graph_values = list(graphs) if graphs is not None else [layer.graph() for layer in layers]
    if len(graph_values) != len(layers):
        raise ValueError("AtlasNav multiplex graph cache is not aligned with layers")
    membership, improvement = la.find_partition_multiplex(
        graph_values,
        la.RBConfigurationVertexPartition,
        layer_weights=list(layer_weights),
        n_iterations=n_iterations,
        seed=seed,
        weights="weight",
        resolution_parameter=resolution,
    )
    canonical = canonicalize_membership(membership)
    sizes = np.bincount(canonical)
    return PartitionCandidate(
        resolution=resolution,
        seed=seed,
        membership=canonical,
        improvement=float(improvement),
        communities=len(sizes),
        maximum_size=int(sizes.max(initial=0)),
        singleton_fraction=float(np.mean(sizes == 1)) if len(sizes) else 0.0,
    )


_PARTITION_CANDIDATE_CONTEXT: dict[str, object] | None = None


def _partition_candidate_job(task: tuple[float, int]) -> PartitionCandidate:
    if _PARTITION_CANDIDATE_CONTEXT is None:
        raise RuntimeError("partition candidate worker started without shared context")
    return partition_once(
        _PARTITION_CANDIDATE_CONTEXT["layers"],
        layer_weights=_PARTITION_CANDIDATE_CONTEXT["layer_weights"],
        resolution=float(task[0]),
        seed=int(task[1]),
        graphs=_PARTITION_CANDIDATE_CONTEXT["graphs"],
    )


def select_resolution(
    layers: Sequence[SparseLayer],
    *,
    layer_weights: Sequence[float],
    target_communities: int,
    resolutions: Iterable[float],
    seeds: Sequence[int] = (42, 43),
    candidate_workers: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    if target_communities < 1:
        raise ValueError("target communities must be positive")
    all_candidates: list[PartitionCandidate] = []
    by_resolution: dict[float, list[PartitionCandidate]] = {}
    # Constructing multi-million-edge igraph objects dominates a resolution
    # scan if repeated. Membership optimisation may vary by resolution/seed;
    # the immutable graph layers do not, so build them exactly once.
    graphs = [layer.graph() for layer in layers]
    resolution_values = [float(value) for value in resolutions]
    tasks = [(resolution, int(seed)) for resolution in resolution_values for seed in seeds]
    if candidate_workers < 1:
        raise ValueError("candidate_workers must be positive")
    if candidate_workers == 1:
        candidate_values = [
            partition_once(
                layers,
                layer_weights=layer_weights,
                resolution=resolution,
                seed=seed,
                graphs=graphs,
            )
            for resolution, seed in tasks
        ]
    else:
        global _PARTITION_CANDIDATE_CONTEXT
        _PARTITION_CANDIDATE_CONTEXT = {
            "layers": tuple(layers),
            "layer_weights": tuple(layer_weights),
            "graphs": tuple(graphs),
        }
        with ProcessPoolExecutor(
            max_workers=min(candidate_workers, len(tasks)),
            mp_context=mp.get_context("fork"),
        ) as executor:
            candidate_values = list(executor.map(_partition_candidate_job, tasks))
        _PARTITION_CANDIDATE_CONTEXT = None
    cursor = 0
    for resolution in resolution_values:
        candidates = candidate_values[cursor:cursor + len(seeds)]
        cursor += len(seeds)
        all_candidates.extend(candidates)
        by_resolution[resolution] = candidates

    first_by_resolution = {
        resolution: candidates[0] for resolution, candidates in by_resolution.items()
    }
    ordered_resolutions = sorted(first_by_resolution)
    scored: list[tuple[float, float, PartitionCandidate, dict[str, float]]] = []
    vertex_count = layers[0].vertices
    for resolution, candidates in by_resolution.items():
        first = candidates[0]
        stability = (
            statistics_mean([
                adjusted_rand_score(first.membership, candidate.membership)
                for candidate in candidates[1:]
            ])
            if len(candidates) > 1
            else 1.0
        )
        sizes = np.bincount(first.membership)
        probabilities = sizes.astype(np.float64) / max(vertex_count, 1)
        entropy = (
            float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(sizes)))
            if len(sizes) > 1
            else 0.0
        )
        layer_gains: list[float] = []
        for layer, layer_weight in zip(layers, layer_weights):
            total_weight = float(np.sum(layer.weights))
            if total_weight <= 0:
                continue
            same = first.membership[layer.edges[:, 0]] == first.membership[layer.edges[:, 1]]
            observed = float(np.sum(layer.weights[same])) / total_weight
            expected = float(np.sum(probabilities * probabilities))
            gain = (observed - expected) / max(1e-9, 1.0 - expected)
            layer_gains.append(float(layer_weight) * gain)
        edge_gain = sum(layer_gains) / max(1e-9, sum(layer_weights))
        position = ordered_resolutions.index(resolution)
        adjacent = []
        if position > 0:
            adjacent.append(adjusted_rand_score(
                first.membership, first_by_resolution[ordered_resolutions[position - 1]].membership
            ))
        if position + 1 < len(ordered_resolutions):
            adjacent.append(adjusted_rand_score(
                first.membership, first_by_resolution[ordered_resolutions[position + 1]].membership
            ))
        plateau_stability = statistics_mean(adjacent) if adjacent else 1.0

        # The desired count is only a weak navigability prior. Structural edge
        # gain, seed reproducibility and resolution-plateau persistence carry
        # most of the objective, so K=100 and K=10 are not hard outcomes.
        count_penalty = 0.20 * abs(math.log(max(first.communities, 1) / target_communities))
        maximum_share = first.maximum_size / max(vertex_count, 1)
        navigable_share = max(0.08, 4.0 / max(target_communities, 1))
        oversized_penalty = max(0.0, maximum_share - navigable_share) * 3.0
        singleton_penalty = first.singleton_fraction * 2.0
        instability_penalty = (1.0 - stability) * 1.25
        plateau_penalty = (1.0 - plateau_stability) * 0.35
        edge_penalty = (1.0 - edge_gain) * 1.0
        entropy_penalty = (1.0 - entropy) * 0.35
        score = (
            count_penalty + oversized_penalty + singleton_penalty + instability_penalty
            + plateau_penalty + edge_penalty + entropy_penalty
        )
        scored.append((score, resolution, first, {
            "count_penalty": count_penalty,
            "oversized_penalty": oversized_penalty,
            "singleton_penalty": singleton_penalty,
            "stability_adjusted_rand": stability,
            "instability_penalty": instability_penalty,
            "plateau_stability_adjusted_rand": plateau_stability,
            "plateau_penalty": plateau_penalty,
            "multiplex_edge_gain_over_size_null": edge_gain,
            "edge_penalty": edge_penalty,
            "normalized_size_entropy": entropy,
            "entropy_penalty": entropy_penalty,
        }))
    score, resolution, winner, components = min(scored, key=lambda row: (row[0], row[1]))
    audit = {
        "selected_resolution": resolution,
        "target_communities": target_communities,
        "selected_communities": winner.communities,
        "selected_maximum_size": winner.maximum_size,
        "selected_singleton_fraction": winner.singleton_fraction,
        "selected_score": score,
        "selected_components": components,
        "candidates": [
            {
                "resolution": candidate.resolution,
                "seed": candidate.seed,
                "communities": candidate.communities,
                "maximum_size": candidate.maximum_size,
                "singleton_fraction": candidate.singleton_fraction,
                "improvement": candidate.improvement,
            }
            for candidate in all_candidates
        ],
        "resolution_scores": [
            {
                "resolution": candidate.resolution,
                "communities": candidate.communities,
                "score": candidate_score,
                **candidate_components,
            }
            for candidate_score, _resolution, candidate, candidate_components in scored
        ],
    }
    return winner.membership, audit


def induced_layer(layer: SparseLayer, members: np.ndarray) -> SparseLayer:
    members = np.asarray(members, dtype=np.int32)
    reverse = np.full(layer.vertices, -1, dtype=np.int32)
    reverse[members] = np.arange(len(members), dtype=np.int32)
    left = reverse[layer.edges[:, 0]]
    right = reverse[layer.edges[:, 1]]
    mask = (left >= 0) & (right >= 0)
    edges = np.column_stack((left[mask], right[mask])).astype(np.int32)
    return SparseLayer(layer.name, len(members), edges, layer.weights[mask].astype(np.float32))


def induced_layers_by_partition(
    layers: Sequence[SparseLayer], membership: np.ndarray
) -> list[list[SparseLayer]]:
    """Materialize every parent-induced layer with one edge scan per view."""
    parent = np.asarray(membership, dtype=np.int32)
    parents = int(parent.max(initial=-1)) + 1
    local_index = np.empty(len(parent), dtype=np.int32)
    parent_sizes = np.bincount(parent, minlength=parents)
    for parent_id in range(parents):
        members = np.flatnonzero(parent == parent_id)
        local_index[members] = np.arange(len(members), dtype=np.int32)
    result: list[list[SparseLayer]] = []
    for layer in layers:
        edge_parent = parent[layer.edges[:, 0]]
        internal = edge_parent == parent[layer.edges[:, 1]]
        parent_ids = edge_parent[internal]
        order = np.argsort(parent_ids, kind="stable")
        ordered_parent = parent_ids[order]
        ordered_edges = layer.edges[internal][order]
        ordered_weights = layer.weights[internal][order]
        counts = np.bincount(ordered_parent, minlength=parents)
        offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        prepared: list[SparseLayer] = []
        for parent_id in range(parents):
            start, end = int(offsets[parent_id]), int(offsets[parent_id + 1])
            edges = local_index[ordered_edges[start:end]].astype(np.int32, copy=True)
            prepared.append(SparseLayer(
                layer.name,
                int(parent_sizes[parent_id]),
                edges.reshape(-1, 2),
                ordered_weights[start:end].astype(np.float32, copy=True),
            ))
        result.append(prepared)
    return result


def child_target(parent_size: int) -> int:
    if parent_size < 24:
        return 1
    return max(2, min(24, int(round(math.sqrt(parent_size / 10.0)))))


_CHILD_PARTITION_CONTEXT: dict[str, object] | None = None


def _partition_child_parent(parent_id: int) -> tuple[int, np.ndarray, np.ndarray, dict[str, object]]:
    """Partition one parent using fork-shared immutable graph arrays."""
    if _CHILD_PARTITION_CONTEXT is None:
        raise RuntimeError("child partition worker started without shared context")
    parent = np.asarray(_CHILD_PARTITION_CONTEXT["parent"], dtype=np.int32)
    prepared_children = _CHILD_PARTITION_CONTEXT["prepared_children"]
    child_layer_weights = _CHILD_PARTITION_CONTEXT["child_layer_weights"]
    child_resolution_multipliers = _CHILD_PARTITION_CONTEXT["child_resolution_multipliers"]
    seeds = _CHILD_PARTITION_CONTEXT["seeds"]
    overrides = _CHILD_PARTITION_CONTEXT["resolution_overrides"]
    assert isinstance(prepared_children, list)
    assert isinstance(overrides, dict)

    members = np.flatnonzero(parent == parent_id).astype(np.int32)
    target = child_target(len(members))
    if target == 1:
        return parent_id, members, np.zeros(len(members), dtype=np.int32), {
            "target_communities": 1,
            "selected_communities": 1,
            "selected_resolution": None,
        }
    local_layers = [prepared[parent_id] for prepared in prepared_children]
    if not any(len(layer.edges) for layer in local_layers):
        return parent_id, members, np.zeros(len(members), dtype=np.int32), {
            "target_communities": target,
            "selected_communities": 1,
            "selected_resolution": None,
            "fallback": "no_episode_or_relation_edges_inside_parent",
        }
    if parent_id in overrides:
        resolutions = [float(overrides[parent_id])]
        reused_resolution = True
    else:
        base = max(0.02, target / max(len(members), 1))
        resolutions = [
            base * float(multiplier) * 10
            for multiplier in child_resolution_multipliers
        ]
        reused_resolution = False
    local, audit = select_resolution(
        local_layers,
        layer_weights=child_layer_weights,
        target_communities=target,
        resolutions=resolutions,
        seeds=tuple(int(seed) + parent_id * 1009 for seed in seeds),
    )
    audit["selected_resolution_reused_from_corpus_only_scan"] = reused_resolution
    return parent_id, members, local, audit


def hierarchical_partition(
    parent_layers: Sequence[SparseLayer],
    child_layers: Sequence[SparseLayer],
    *,
    parent_layer_weights: Sequence[float] = (1.0, 0.75),
    child_layer_weights: Sequence[float] = (1.0, 1.0),
    parent_target: int = 100,
    parent_resolutions: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
    child_resolution_multipliers: Sequence[float] = (0.35, 0.6, 1.0, 1.7, 2.8),
    seeds: Sequence[int] = (42, 43),
    child_resolution_overrides: dict[int, float] | None = None,
    child_workers: int = 1,
    parent_candidate_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    parent, parent_audit = select_resolution(
        parent_layers,
        layer_weights=parent_layer_weights,
        target_communities=parent_target,
        resolutions=parent_resolutions,
        seeds=seeds,
        candidate_workers=parent_candidate_workers,
    )
    if child_workers < 1:
        raise ValueError("child_workers must be positive")
    leaf = np.full(len(parent), -1, dtype=np.int32)
    child_audits: dict[str, object] = {}
    next_leaf = 0
    prepared_children = induced_layers_by_partition(child_layers, parent)
    parent_ids = list(range(int(parent.max(initial=-1)) + 1))
    global _CHILD_PARTITION_CONTEXT
    _CHILD_PARTITION_CONTEXT = {
        "parent": parent,
        "prepared_children": prepared_children,
        "child_layer_weights": tuple(child_layer_weights),
        "child_resolution_multipliers": tuple(child_resolution_multipliers),
        "seeds": tuple(seeds),
        "resolution_overrides": dict(child_resolution_overrides or {}),
    }
    if child_workers == 1:
        results = [_partition_child_parent(parent_id) for parent_id in parent_ids]
    else:
        # Linux fork shares the large immutable NumPy graph arrays copy-on-write;
        # only compact memberships/audits return through the worker pipes.
        with ProcessPoolExecutor(
            max_workers=min(child_workers, len(parent_ids)),
            mp_context=mp.get_context("fork"),
        ) as executor:
            results = list(executor.map(_partition_child_parent, parent_ids))
    _CHILD_PARTITION_CONTEXT = None
    for parent_id, members, local, audit in results:
        leaf[members] = local + next_leaf
        child_audits[f"C{parent_id:03d}"] = {"parent_size": len(members), **audit}
        next_leaf += int(local.max(initial=-1)) + 1
    if np.any(leaf < 0):
        raise RuntimeError("AtlasNav hierarchy left files without a primary leaf")
    return parent, leaf, {
        "parent": parent_audit,
        "children": child_audits,
        "leaves": next_leaf,
        "child_partition_workers": child_workers,
        "parent_candidate_workers": parent_candidate_workers,
    }


def cross_links(
    layers: Sequence[SparseLayer],
    leaf_membership: np.ndarray,
    *,
    maximum_per_leaf: int = 3,
) -> dict[int, list[dict[str, float | int]]]:
    leaf = np.asarray(leaf_membership, dtype=np.int32)
    sizes = np.bincount(leaf)
    accumulated: dict[tuple[int, int], float] = {}
    leaf_count = len(sizes)
    for layer in layers:
        left = leaf[layer.edges[:, 0]].astype(np.int64, copy=False)
        right = leaf[layer.edges[:, 1]].astype(np.int64, copy=False)
        mask = left != right
        if not np.any(mask):
            continue
        low = np.minimum(left[mask], right[mask])
        high = np.maximum(left[mask], right[mask])
        encoded = low * leaf_count + high
        order = np.argsort(encoded, kind="stable")
        ordered_ids = encoded[order]
        ordered_weights = layer.weights[mask][order].astype(np.float64, copy=False)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(ordered_ids)) + 1))
        unique_ids = ordered_ids[starts]
        totals = np.add.reduceat(ordered_weights, starts)
        for encoded_id, total in zip(unique_ids, totals):
            a, b = divmod(int(encoded_id), leaf_count)
            key = (a, b)
            accumulated[key] = accumulated.get(key, 0.0) + float(total)
    candidates: dict[int, list[tuple[float, int]]] = {index: [] for index in range(len(sizes))}
    for (left, right), weight in accumulated.items():
        normalized = weight / math.sqrt(max(1, int(sizes[left])) * max(1, int(sizes[right])))
        candidates[left].append((normalized, right))
        candidates[right].append((normalized, left))
    return {
        source: [
            {"target_leaf": target, "normalized_weight": weight}
            for weight, target in sorted(rows, key=lambda row: (-row[0], row[1]))[:maximum_per_leaf]
        ]
        for source, rows in candidates.items()
    }
