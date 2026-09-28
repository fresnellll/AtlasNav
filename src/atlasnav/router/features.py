#!/usr/bin/env python3
"""Shared, deterministic feature and inference code for the AtlasNav facet router."""

from __future__ import annotations

import math
import re
from typing import Mapping

import numpy as np

from atlasnav.atlas.signatures import VIEWS


CALIBRATION_FILES = 8_192
SEMANTIC_TOP = 64
RBO_P = 0.90
MECHANICAL_DIMENSIONS = 8
META_PER_VIEW = 10
FEATURE_DIMENSIONS = 4 * 192 + 4 * META_PER_VIEW + MECHANICAL_DIMENSIONS


def calibration_indexes(documents: int = 100_195, seed: int = 41_041) -> np.ndarray:
    """A fixed corpus-only reference sample shared by training and deployment."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(documents, size=min(CALIBRATION_FILES, documents), replace=False)).astype(np.int32)


def mechanical_features(query: str) -> np.ndarray:
    words = re.findall(r"[\w'-]+", query, flags=re.UNICODE)
    upper = re.findall(r"\b[A-Z][A-Za-z'-]+\b", query)
    years = re.findall(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b", query)
    numbers = re.findall(r"(?<!\w)[+-]?(?:\d+[.,]?\d*|\.\d+)(?!\w)", query)
    quotes = re.findall(r"[\"“”‘’'][^\"“”‘’']+[\"“”‘’']", query)
    relations = re.findall(
        r"\b(?:born|died|married|founded|member|parent|child|spouse|before|after|during|between|located|owned|directed|written|created|served)\b",
        query, flags=re.IGNORECASE,
    )
    question_words = re.findall(r"\b(?:who|when|where|which|what|whose|how)\b", query, flags=re.IGNORECASE)
    return np.asarray([
        math.log1p(len(query)) / 8.0,
        math.log1p(len(words)) / 5.0,
        min(len(upper), 8) / 8.0,
        min(len(years), 4) / 4.0,
        min(len(numbers), 6) / 6.0,
        min(len(quotes), 4) / 4.0,
        min(len(relations), 6) / 6.0,
        min(len(question_words), 4) / 4.0,
    ], dtype=np.float32)


def rank_biased_overlap(left: np.ndarray, right: np.ndarray, p: float = RBO_P) -> float:
    right_set: set[int] = set()
    left_set: set[int] = set()
    score = 0.0
    depth = min(len(left), len(right))
    for rank in range(depth):
        left_set.add(int(left[rank]))
        right_set.add(int(right[rank]))
        overlap = len(left_set & right_set)
        score += (1.0 - p) * (p ** rank) * overlap / (rank + 1)
    return float(score)


def graph_coherence(nodes: np.ndarray, sorted_edge_keys: np.ndarray, documents: int) -> float:
    nodes = np.asarray(nodes[:16], dtype=np.int64)
    if len(nodes) < 2:
        return 0.0
    a, b = np.triu_indices(len(nodes), 1)
    low = np.minimum(nodes[a], nodes[b])
    high = np.maximum(nodes[a], nodes[b])
    keys = low * documents + high
    positions = np.searchsorted(sorted_edge_keys, keys)
    present = (positions < len(sorted_edge_keys))
    if np.any(present):
        present[present] &= sorted_edge_keys[positions[present]] == keys[present]
    return float(np.mean(present))


def view_meta_features(
    calibration_similarity: np.ndarray,
    calibration_ids: np.ndarray,
    lexical_ids: np.ndarray,
    lexical_similarity: np.ndarray,
    parent_assignment: np.ndarray,
    leaf_assignment: np.ndarray,
    sorted_edge_keys: np.ndarray,
    documents: int,
) -> np.ndarray:
    """Ten deployable corpus-only statistics for one query/view pair."""
    values = np.asarray(calibration_similarity, dtype=np.float32)
    median = float(np.median(values))
    q95, q99 = np.quantile(values, [0.95, 0.99]).tolist()
    lexical_ids = np.asarray(lexical_ids[:SEMANTIC_TOP], dtype=np.int32)
    lexical_similarity = np.asarray(lexical_similarity[:SEMANTIC_TOP], dtype=np.float32)
    # Compute semantic/lexical overlap on the union of the fixed calibration
    # sample and lexical hits.  Otherwise a lexical hit could overlap only by
    # accidentally belonging to the 8,192-file calibration sample.
    novel = ~np.isin(lexical_ids, calibration_ids, assume_unique=False)
    union_ids = np.concatenate((calibration_ids, lexical_ids[novel]))
    union_values = np.concatenate((values, lexical_similarity[novel]))
    top_count = min(SEMANTIC_TOP, len(union_values))
    local = np.argpartition(union_values, len(union_values) - top_count)[-top_count:]
    order = local[np.lexsort((union_ids[local], -union_values[local]))]
    semantic_ids = union_ids[order]
    top_values = union_values[order]
    top32 = semantic_ids[:32]
    parent_counts = np.unique(parent_assignment[top32], return_counts=True)[1]
    leaf_counts = np.unique(leaf_assignment[top32], return_counts=True)[1]
    denominator = max(q95 - median, 1e-5)
    return np.asarray([
        float(np.std(values)),
        q95,
        q99,
        float(np.mean(top_values[:8])),
        float(np.clip((float(np.mean(top_values[:8])) - q95) / denominator, 0.0, 20.0)),
        float(np.max(parent_counts) / max(len(top32), 1)),
        float(np.max(leaf_counts) / max(len(top32), 1)),
        graph_coherence(top32, sorted_edge_keys, documents),
        float(np.mean(lexical_similarity)) if len(lexical_similarity) else median,
        rank_biased_overlap(semantic_ids, lexical_ids),
    ], dtype=np.float32)


def assemble_features(
    query_vectors: Mapping[str, np.ndarray], meta: Mapping[str, np.ndarray], query: str
) -> np.ndarray:
    feature = np.concatenate(
        [*[np.asarray(query_vectors[view], dtype=np.float32) for view in VIEWS],
         *[np.asarray(meta[view], dtype=np.float32) for view in VIEWS],
         mechanical_features(query)]
    )
    if feature.shape != (FEATURE_DIMENSIONS,):
        raise RuntimeError(f"AtlasNav feature shape mismatch: {feature.shape}")
    return feature


def router_weights(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    rho: float,
    temperature: float,
) -> np.ndarray:
    normalized = np.clip((np.asarray(features, dtype=np.float32) - mean) / scale, -8.0, 8.0)
    logits = normalized @ coefficients.T + intercept
    logits = logits / temperature
    logits -= np.max(logits, axis=-1, keepdims=True)
    soft = np.exp(logits)
    soft /= np.sum(soft, axis=-1, keepdims=True)
    return (1.0 - rho) / len(VIEWS) + rho * soft
