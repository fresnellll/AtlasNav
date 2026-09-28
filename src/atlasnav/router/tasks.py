"""Construct outcome-blind single/pair/triple Router support tasks."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from atlasnav.atlas.signatures import VIEWS
from atlasnav.io import atomic_json, atomic_jsonl, sha256_file
from atlasnav.text import SENTENCE_RE, clean_field, clean_text


SPLITS = ("train", "validation", "test")
KINDS = ("single", "pair", "triple")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid manifest: {path}")
    return value


def _asset(root: Path, manifest: dict[str, Any], key: str, fallback: str) -> Path:
    value = (manifest.get("arrays") or {}).get(key) or fallback
    path = root / str(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _split_counts(total: int, fractions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(fractions) * total
    result = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - result), kind="stable")[: total - int(result.sum())]:
        result[index] += 1
    return result.tolist()


def _parent_splits(parent: np.ndarray, seed: int) -> dict[int, str]:
    values = sorted(set(map(int, parent)), key=lambda item: _digest(f"{seed}:parent:{item}"))
    counts = _split_counts(len(values), (0.70, 0.15, 0.15))
    boundaries = np.cumsum(counts)
    return {
        value: SPLITS[0 if rank < boundaries[0] else 1 if rank < boundaries[1] else 2]
        for rank, value in enumerate(values)
    }


def _candidate_indexes(leaf: np.ndarray, per_leaf: int, seed: int) -> np.ndarray:
    selected: list[int] = []
    for leaf_id in sorted(set(map(int, leaf))):
        members = np.flatnonzero(leaf == leaf_id)
        ordered = sorted(map(int, members), key=lambda item: _digest(f"{seed}:leaf:{leaf_id}:{item}"))
        selected.extend(ordered[:per_leaf])
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def _load_selected(corpus: Path, selected: np.ndarray, maximum_characters: int = 24_000) -> dict[int, dict[str, str]]:
    manifest = _manifest(corpus)
    parquet = pq.ParquetFile(corpus / str(manifest["documents_file"]))
    requested = set(map(int, selected))
    rows: dict[int, dict[str, str]] = {}
    offset = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["docid", "text", "url"]):
        values = batch.to_pylist()
        for local, value in enumerate(values):
            index = offset + local
            if index in requested:
                rows[index] = {
                    "docid": str(value["docid"]),
                    "text": clean_text(str(value.get("text") or ""))[:maximum_characters],
                    "url": str(value.get("url") or ""),
                }
        offset += len(values)
    if set(rows) != requested:
        raise RuntimeError("failed to load every selected Router support file")
    return rows


def _excerpts(row: dict[str, str], namespace: str) -> list[dict[str, str]]:
    sentences = [
        clean_field(value, 720) for value in SENTENCE_RE.split(row["text"])
        if len(clean_field(value, 720)) >= 45
    ]
    if len(sentences) < 2:
        value = clean_field(row["text"], 720)
        sentences = [value, value]
    ordered = sorted(enumerate(sentences), key=lambda item: _digest(f"{namespace}:{item[0]}:{item[1]}"))
    output = []
    for position, sentence in ordered:
        if sentence in {row["text"] for row in output}:
            continue
        output.append({
            "view": VIEWS[len(output) % len(VIEWS)], "text": sentence,
            "text_sha256": _digest(sentence),
        })
        if len(output) == 2:
            break
    return output


def _edges(atlas: Path, manifest: dict[str, Any], candidates: set[int], parent: np.ndarray,
           parent_splits: dict[int, str]) -> dict[str, list[tuple[int, int, float]]]:
    output: dict[str, list[tuple[int, int, float]]] = {}
    for view in VIEWS:
        graph = np.load(atlas / f"{view}_graph.npz")
        edges = np.asarray(graph["edges"], dtype=np.int32)
        weights = np.asarray(graph["weights"], dtype=np.float32)
        values = []
        for (left, right), weight in zip(edges, weights):
            a, b = int(left), int(right)
            if a not in candidates or b not in candidates:
                continue
            if parent_splits[int(parent[a])] != parent_splits[int(parent[b])]:
                continue
            values.append((a, b, float(weight)))
        output[view] = sorted(values, key=lambda row: (-row[2], row[0], row[1]))
    return output


def build_support_tasks(
    *,
    corpus: Path,
    atlas: Path,
    output: Path,
    tasks: int = 30_000,
    candidate_per_leaf: int = 72,
    single_fraction: float = 0.25,
    pair_fraction: float = 0.50,
    seed: int = 41_200,
) -> dict[str, Any]:
    """Create corpus-only candidates before LLM question synthesis.

    Parent-group splitting occurs before task selection, so no parent region is
    shared by train, validation, and test. Pair/triple positives follow sparse
    Atlas edges. This stage never reads benchmark questions, answers, Qrels,
    trajectories, or correctness.
    """
    corpus, atlas, output = corpus.resolve(), atlas.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace Router task bundle: {output}")
    if (tasks < 3 or not 0 < single_fraction < 1 or not 0 < pair_fraction < 1
            or single_fraction + pair_fraction >= 1):
        raise ValueError("invalid Router task quotas")
    atlas_manifest = _manifest(atlas)
    parent = np.load(_asset(atlas, atlas_manifest, "file_to_parent", "file_to_parent.i32.npy"), mmap_mode="r")
    leaf = np.load(_asset(atlas, atlas_manifest, "file_to_leaf", "file_to_leaf.i32.npy"), mmap_mode="r")
    candidates = _candidate_indexes(leaf, candidate_per_leaf, seed)
    selected = _load_selected(corpus, candidates)
    usable = {index for index, row in selected.items() if len(_excerpts(row, f"eligible:{index}")) == 2}
    parent_split = _parent_splits(parent, seed)
    graph_edges = _edges(atlas, atlas_manifest, usable, parent, parent_split)
    kind_counts = _split_counts(tasks, (single_fraction, pair_fraction, 1.0 - single_fraction - pair_fraction))
    if kind_counts[-1] < 1:
        raise ValueError("triple task fraction must be positive")
    split_counts_by_kind = {
        kind: dict(zip(SPLITS, _split_counts(count, (0.70, 0.15, 0.15))))
        for kind, count in zip(KINDS, kind_counts)
    }
    by_split = {
        split: sorted(
            (index for index in usable if parent_split[int(parent[index])] == split),
            key=lambda index: _digest(f"{seed}:{split}:single:{index}"),
        )
        for split in SPLITS
    }
    pairs_by_split: dict[str, list[tuple[int, int, str, float]]] = defaultdict(list)
    for view, values in graph_edges.items():
        for left, right, weight in values:
            if int(leaf[left]) == int(leaf[right]):
                continue
            split = parent_split[int(parent[left])]
            pairs_by_split[split].append((left, right, view, weight))
    for split in SPLITS:
        pairs_by_split[split] = sorted(
            pairs_by_split[split],
            key=lambda row: (-row[3], _digest(f"{seed}:{split}:{row[0]}:{row[1]}:{row[2]}")),
        )
    tasks_out: list[dict[str, Any]] = []

    def support(index: int, namespace: str) -> dict[str, Any]:
        row = selected[index]
        return {
            "file_index": index, "docid": row["docid"],
            "parent": int(parent[index]), "leaf": int(leaf[index]),
            "excerpts": _excerpts(row, namespace),
        }

    for split in SPLITS:
        count = split_counts_by_kind["single"][split]
        values = by_split[split]
        if not values:
            raise RuntimeError(f"no eligible Router files for split={split}")
        for offset in range(count):
            index = values[offset % len(values)]
            task_id = f"single:{split}:{offset:06d}"
            tasks_out.append({
                "schema": "atlasnav_grounded_support_task_v1", "task_id": task_id,
                "kind": "single", "split": split, "edge_views": [],
                "support_files": [support(index, task_id)],
                "reads_evaluation_artifacts": False,
            })
    for split in SPLITS:
        values = pairs_by_split[split]
        count = split_counts_by_kind["pair"][split]
        if len(values) < count:
            raise RuntimeError(f"insufficient cross-leaf graph pairs for split={split}: {len(values)} < {count}")
        for offset, (left, right, view, _weight) in enumerate(values[:count]):
            task_id = f"pair:{split}:{offset:06d}"
            tasks_out.append({
                "schema": "atlasnav_grounded_support_task_v1", "task_id": task_id,
                "kind": "pair", "split": split, "edge_views": [view],
                "support_files": [support(left, task_id), support(right, task_id)],
                "reads_evaluation_artifacts": False,
            })
    for split in SPLITS:
        target = split_counts_by_kind["triple"][split]
        adjacency: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
        for left, right, view, weight in pairs_by_split[split]:
            adjacency[left].append((right, view, weight))
            adjacency[right].append((left, view, weight))
        chains: list[tuple[float, int, int, int, str, str]] = []
        for middle, neighbours in adjacency.items():
            ordered = sorted(neighbours, key=lambda row: (-row[2], row[0]))[:12]
            for left_index in range(len(ordered)):
                for right_index in range(left_index + 1, len(ordered)):
                    left, left_view, left_weight = ordered[left_index]
                    right, right_view, right_weight = ordered[right_index]
                    if left == right:
                        continue
                    chains.append((min(left_weight, right_weight), left, middle, right, left_view, right_view))
        chains.sort(key=lambda row: (-row[0], _digest(f"{seed}:{split}:{row[1]}:{row[2]}:{row[3]}")))
        seen: set[tuple[int, int, int]] = set()
        accepted = []
        for chain in chains:
            identity = (chain[1], chain[2], chain[3])
            if identity in seen:
                continue
            seen.add(identity)
            accepted.append(chain)
            if len(accepted) == target:
                break
        if len(accepted) < target:
            raise RuntimeError(f"insufficient graph chains for split={split}: {len(accepted)} < {target}")
        for offset, (_score, left, middle, right, left_view, right_view) in enumerate(accepted):
            task_id = f"triple:{split}:{offset:06d}"
            tasks_out.append({
                "schema": "atlasnav_grounded_support_task_v1", "task_id": task_id,
                "kind": "triple", "split": split, "edge_views": [left_view, right_view],
                "support_files": [support(left, task_id), support(middle, task_id), support(right, task_id)],
                "reads_evaluation_artifacts": False,
            })
    tasks_out.sort(key=lambda row: (SPLITS.index(row["split"]), KINDS.index(row["kind"]), row["task_id"]))
    output.mkdir(parents=True)
    task_path = output / "tasks.jsonl"
    atomic_jsonl(task_path, tasks_out)
    manifest = {
        "schema": "atlasnav_grounded_support_tasks_v1", "finalized": True,
        "tasks": len(tasks_out),
        "kind_counts": {kind: sum(row["kind"] == kind for row in tasks_out) for kind in KINDS},
        "split_counts": {split: sum(row["split"] == split for row in tasks_out) for split in SPLITS},
        "candidate_files": len(usable), "candidate_per_leaf": candidate_per_leaf,
        "parent_group_split": True, "reads_evaluation_artifacts": False,
        "corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        "atlas_manifest_sha256": sha256_file(atlas / "manifest.json"),
        "tasks_sha256": sha256_file(task_path), "seed": seed,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
